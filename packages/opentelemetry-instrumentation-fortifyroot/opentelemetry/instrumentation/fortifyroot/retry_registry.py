"""§4.7.1 token-based framework-attempt registry.

Shared across all FR fork instrumentations. The flow is:

  - Framework wrappers (LiteLLM / LangChain / LlamaIndex) call
    ``register_framework_attempt()`` at the start of EACH HTTP attempt
    they observe (i.e. once per ``log_pre_api_call`` / ``on_chat_model_start``
    / ``LLMChatStartEvent``). They get back an opaque token.

  - The same wrappers call ``unregister_framework_attempt(token)`` when
    that attempt finishes (success or failure callback).

  - Direct-SDK wrappers (OpenAI / Anthropic / Bedrock) consult
    ``is_framework_owned()`` at every wrapped ``send()`` / event-hook
    invocation. If True, they SUPPRESS their own ``retry_attempt`` span
    emission so the framework's emitter remains the single source of
    truth for that logical call.

The "owned" state is tracked PER OS THREAD because LiteLLM's
``log_pre_api_call`` and the underlying ``httpx.send()`` it triggers
run on the same thread synchronously. Async paths use the same
underlying httpx client, so the thread-local model still applies for
the duration of one attempt. (Re-entrancy across threads is handled by
the per-TID dict.)

Design references:
  - RETRY_LOOP.md §4.7   — suppression discipline rationale.
  - RETRY_LOOP.md §4.7.1 — registry shape, eviction policy, required
                            tests (re-entrancy, stale-cleanup,
                            parent-end cleanup, cap-eviction,
                            thread-ID reuse).
  - phase_st10_retryloop.txt round-3 disposition — replaces an
    earlier ``set[int]`` design that didn't handle re-entrancy.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from typing import Optional

logger = logging.getLogger(__name__)


# Module-level state (per process). Per RETRY_LOOP.md §4.7.1:
#   shape: dict[tid, dict[token, started_at_monotonic_seconds]]
# Keying by tid (thread ID) is what makes per-thread ownership work;
# keying by token within a TID is what makes re-entrancy work (one
# thread can own multiple nested attempts and only suppression-stops
# when ALL tokens have been unregistered).
_REGISTRY: "dict[int, dict[str, float]]" = {}

# Reverse index: token → originating TID. Required so that
# unregister_framework_attempt() works correctly when the framework's
# terminal callback runs on a different OS thread from its start
# callback (e.g. async paths where success/failure callbacks dispatch
# to a worker thread). Without this mapping, unregister would only
# clean up the CURRENT thread's bucket — leaking the original token
# in the original TID's bucket and falsely keeping that TID
# "framework-owned" until TTL eviction.
# (Review-batch-1 Blocker fix 2026-05-10.)
_TOKEN_TO_TID: "dict[str, int]" = {}

_REGISTRY_LOCK = threading.Lock()

# Bounded-size + TTL-based eviction caps. Tunable; chosen to match
# the existing _FR_COMPLETION_SAFETY_MARKERS_MAX shape so all FR
# bookkeeping has consistent memory ceilings.
_REGISTRY_MAX = 4096                 # max ENTRIES (sum of inner-dict sizes) before cap-eviction kicks in
_REGISTRY_EVICT_BATCH = 1024         # number of entries to drop in one cap-eviction pass
_REGISTRY_STALE_TTL_SEC = 60.0       # TTL after which a token is considered stale and evicted

# Rate-limit warning emission to at most one per N evictions per bucket
# (per-TID). Stale-eviction can spike noisily if a framework crashes
# with many leaked tokens; logging every one would flood operator
# dashboards.
_WARN_EVERY_N = 32
_evict_warn_counter: dict[int, int] = {}
_evict_warn_lock = threading.Lock()


def _now() -> float:
    return time.monotonic()


def _evict_stale_for_tid_locked(tid: int, now: float) -> int:
    """Remove tokens older than the TTL for the given TID. Returns
    number of entries evicted. Caller must hold _REGISTRY_LOCK.

    This runs in-band on EVERY ``is_framework_owned()`` call so a
    leaked token (framework crashed without unregistering) cannot
    suppress direct-SDK emission indefinitely. The cost is one O(K)
    pass per call where K = tokens active on this TID — almost always
    0 or 1 in practice.
    """
    bucket = _REGISTRY.get(tid)
    if not bucket:
        return 0
    cutoff = now - _REGISTRY_STALE_TTL_SEC
    stale_tokens = [tok for tok, started_at in bucket.items() if started_at < cutoff]
    for tok in stale_tokens:
        bucket.pop(tok, None)
        _TOKEN_TO_TID.pop(tok, None)
    if not bucket:
        _REGISTRY.pop(tid, None)
    if stale_tokens:
        _maybe_warn_eviction(tid, len(stale_tokens))
    return len(stale_tokens)


def _enforce_max_locked(now: float) -> int:
    """If the total entry count exceeds _REGISTRY_MAX, evict
    _REGISTRY_EVICT_BATCH oldest entries (across all TIDs). Returns
    number evicted. Caller must hold _REGISTRY_LOCK.

    This is a hard cap defending against pathological leak storms.
    Normal operation never triggers it because TTL-eviction keeps the
    registry small.
    """
    total = sum(len(b) for b in _REGISTRY.values())
    if total <= _REGISTRY_MAX:
        return 0
    # Flatten + sort by started_at; drop the oldest _REGISTRY_EVICT_BATCH.
    flat: list[tuple[float, int, str]] = []
    for tid, bucket in _REGISTRY.items():
        for tok, started_at in bucket.items():
            flat.append((started_at, tid, tok))
    flat.sort()  # ascending by started_at — oldest first
    to_evict = flat[:_REGISTRY_EVICT_BATCH]
    for _, tid, tok in to_evict:
        bucket = _REGISTRY.get(tid)
        if bucket is None:
            continue
        bucket.pop(tok, None)
        _TOKEN_TO_TID.pop(tok, None)
        if not bucket:
            _REGISTRY.pop(tid, None)
    logger.warning(
        "fortifyroot retry_registry: cap-evicted %d entries (total exceeded %d); "
        "this indicates leaked tokens from a misbehaving framework wrapper",
        len(to_evict),
        _REGISTRY_MAX,
    )
    return len(to_evict)


def _maybe_warn_eviction(tid: int, evicted: int) -> None:
    with _evict_warn_lock:
        c = _evict_warn_counter.get(tid, 0) + evicted
        if c < _WARN_EVERY_N:
            _evict_warn_counter[tid] = c
            return
        _evict_warn_counter[tid] = 0
    # Outside the lock so logger.warning's I/O can't deadlock with
    # registry operations elsewhere.
    logger.warning(
        "fortifyroot retry_registry: stale-evicted %d+ tokens on tid=%d; "
        "framework wrapper may be leaking attempts (TTL=%.0fs)",
        evicted,
        tid,
        _REGISTRY_STALE_TTL_SEC,
    )


def register_framework_attempt() -> str:
    """Register a new framework-attempt token on the current OS thread.

    Returns the opaque token string. The caller MUST pass this token
    to ``unregister_framework_attempt`` when the attempt completes.

    Idempotency: calling this multiple times on the same thread is
    safe and supported (re-entrancy / nested attempts) — each call
    mints a fresh token.
    """
    token = uuid.uuid4().hex
    tid = threading.get_ident()
    now = _now()
    with _REGISTRY_LOCK:
        bucket = _REGISTRY.get(tid)
        if bucket is None:
            bucket = {}
            _REGISTRY[tid] = bucket
        bucket[token] = now
        _TOKEN_TO_TID[token] = tid
        _enforce_max_locked(now)
    return token


def unregister_framework_attempt(token: Optional[str]) -> None:
    """Remove the given token from its ORIGINATING TID bucket. No-op
    if token is None or unknown (defensive — tolerates double-unregister
    when both sync and async callbacks fire).

    The originating TID is recovered via the ``_TOKEN_TO_TID`` reverse
    index — we deliberately do NOT use ``threading.get_ident()`` here,
    because the framework's terminal callback may run on a different
    OS thread from the start callback (e.g. asyncio worker dispatch).
    Using the current thread's TID would leak the entry in the
    originating TID's bucket and falsely keep that thread
    "framework-owned" until TTL eviction.
    """
    if not token:
        return
    with _REGISTRY_LOCK:
        original_tid = _TOKEN_TO_TID.pop(token, None)
        if original_tid is None:
            # Already unregistered (or unknown). Defensive no-op.
            return
        bucket = _REGISTRY.get(original_tid)
        if bucket is None:
            return
        bucket.pop(token, None)
        if not bucket:
            _REGISTRY.pop(original_tid, None)


def is_framework_owned(tid: Optional[int] = None) -> bool:
    """Return True iff the given TID (default: current thread) has
    at least one live framework-attempt token registered.

    This is the LOAD-BEARING read path consulted by direct-SDK
    wrappers (OpenAI / Anthropic / Bedrock) at every wrapped HTTP
    send. If it returns True, the direct-SDK wrapper SUPPRESSES its
    own retry_attempt span emission — the framework is the source of
    truth for that logical call.

    Performs per-TID stale eviction in-band before answering, so a
    leaked token cannot indefinitely suppress emission (see
    review-round-4 Q2 fix in phase_st10_retryloop.txt).
    """
    if tid is None:
        tid = threading.get_ident()
    now = _now()
    with _REGISTRY_LOCK:
        _evict_stale_for_tid_locked(tid, now)
        bucket = _REGISTRY.get(tid)
        return bool(bucket)


def clear_for_thread(tid: Optional[int] = None) -> int:
    """Drop all tokens for the given TID (default: current thread).
    Returns number of tokens dropped. Used as a parent-span-end
    orphan-cleanup primitive: when a framework wrapper's parent span
    ends, any tokens still registered for that thread are by
    definition orphans (the parent finished without success/failure
    callbacks firing for those attempts).
    """
    if tid is None:
        tid = threading.get_ident()
    with _REGISTRY_LOCK:
        bucket = _REGISTRY.pop(tid, None)
        if bucket:
            for tok in bucket:
                _TOKEN_TO_TID.pop(tok, None)
    return len(bucket) if bucket else 0


def _registry_size_for_test() -> "tuple[int, int]":
    """Test-only helper: returns (num_tids, total_tokens). Not part
    of the public contract; do not import outside tests.
    """
    with _REGISTRY_LOCK:
        return (len(_REGISTRY), sum(len(b) for b in _REGISTRY.values()))


def _reset_for_test() -> None:
    """Test-only helper: wipe registry state. Not part of the public
    contract; do not import outside tests.
    """
    with _REGISTRY_LOCK:
        _REGISTRY.clear()
        _TOKEN_TO_TID.clear()
    with _evict_warn_lock:
        _evict_warn_counter.clear()


__all__ = [
    "register_framework_attempt",
    "unregister_framework_attempt",
    "is_framework_owned",
    "clear_for_thread",
]

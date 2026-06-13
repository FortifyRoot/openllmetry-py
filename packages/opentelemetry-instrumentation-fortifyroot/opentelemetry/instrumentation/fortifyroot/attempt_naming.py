"""Shared naming helpers for FR per-LLM-attempt spans."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

FR_SPAN_ROLE_KEY = "fortifyroot.span.role"
FR_SPAN_ROLE_LLM_ATTEMPT = "llm_attempt"
FR_HAS_ATTEMPT_CHILD_KEY = "fortifyroot.span.has_attempt_child"
FR_ATTEMPT_NUMBER_KEY = "fortifyroot.attempt.number"
FR_ATTEMPT_IS_RETRY_KEY = "fortifyroot.attempt.is_retry"

_COUNTER_LOCK = threading.Lock()
_COUNTERS: dict[str, tuple[int, float]] = {}
_COUNTER_MAX = 4096
_COUNTER_TTL_SEC = 3600.0
_COUNTER_EVICT_BATCH = 1024


def _now() -> float:
    return time.monotonic()


def _parent_key(parent_span: Any) -> str | None:
    try:
        ctx = parent_span.get_span_context()
        if ctx is None or not ctx.is_valid:
            return None
        return f"{ctx.trace_id:032x}:{ctx.span_id:016x}"
    except Exception:
        return None


def _evict_stale_locked(now: float) -> None:
    cutoff = now - _COUNTER_TTL_SEC
    stale = [key for key, (_, seen_at) in _COUNTERS.items() if seen_at < cutoff]
    for key in stale:
        _COUNTERS.pop(key, None)


def _enforce_max_locked() -> None:
    if len(_COUNTERS) <= _COUNTER_MAX:
        return
    items = sorted(_COUNTERS.items(), key=lambda item: item[1][1])
    for key, _ in items[:_COUNTER_EVICT_BATCH]:
        _COUNTERS.pop(key, None)
    logger.warning(
        "fortifyroot attempt_naming: cap-evicted %d parent attempt counters "
        "(max=%d)",
        min(_COUNTER_EVICT_BATCH, len(items)),
        _COUNTER_MAX,
    )


def next_llm_attempt(parent_span: Any, span_name_prefix: str) -> tuple[str, int, bool]:
    """Return the next span name and ordinal for a parent LLM span.

    ``span_name_prefix`` should be the provider/framework prefix without
    an attempt suffix, for example ``fortifyroot.openai``. The returned
    span name is ``<prefix>.attempt_<N>``.

    Use this only when ``parent_span`` is the logical retry-group span.
    Framework integrations that can only see a broad workflow parent should
    use ``first_llm_attempt`` instead, otherwise unrelated LLM calls under the
    same workflow can be mislabeled as retries.
    """
    key = _parent_key(parent_span)
    now = _now()
    if key is None:
        return f"{span_name_prefix}.attempt_1", 1, False

    with _COUNTER_LOCK:
        _evict_stale_locked(now)
        last, _ = _COUNTERS.get(key, (0, now))
        number = last + 1
        _COUNTERS[key] = (number, now)
        _enforce_max_locked()

    return f"{span_name_prefix}.attempt_{number}", number, number > 1


def first_llm_attempt(span_name_prefix: str) -> tuple[str, int, bool]:
    """Return conservative attempt metadata for one observed LLM call.

    Some frameworks expose a broad workflow parent but no stable logical
    retry-group key. In those cases the truthful signal is "this observed
    call is an LLM attempt"; it is not safe to infer retry ordinals.
    """
    return f"{span_name_prefix}.attempt_1", 1, False


def llm_attempt_attributes(number: int, is_retry: bool) -> dict[str, Any]:
    return {
        FR_SPAN_ROLE_KEY: FR_SPAN_ROLE_LLM_ATTEMPT,
        FR_ATTEMPT_NUMBER_KEY: number,
        FR_ATTEMPT_IS_RETRY_KEY: is_retry,
    }


def clear_attempt_counters_for_test() -> None:
    with _COUNTER_LOCK:
        _COUNTERS.clear()

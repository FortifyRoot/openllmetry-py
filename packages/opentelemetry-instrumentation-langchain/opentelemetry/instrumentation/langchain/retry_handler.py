"""Retry-aware emission for LangChain.

Design contract:

  - Hook BOTH ``on_chat_model_start`` (chat models)
    AND ``on_llm_start`` (legacy completion LLMs).
  - Per-HTTP-attempt firing is verified on framework-layer retry
    paths (e.g. ``Runnable.with_retry``); provider-SDK-internal
    retries (e.g. ``ChatOpenAI(max_retries=N)``) fire callbacks
    once per logical call, so the framework coverage limitation applies.
  - Use ``run_id`` as the per-attempt correlation key.
  - Emit ``fortifyroot.langchain.attempt_<N>`` sibling spans
    under the parent_run_id's span (NOT under the per-LLM
    Traceloop span — siblings need to share a parent for
    RetryDetectorProc's grouping to work).
  - Register/unregister framework-attempt tokens so direct-SDK
    wrappers suppress their own emission.
  - Set has_attempt_child=true on the parent AFTER
    the first qualifying llm_attempt successfully starts.
  - Attempt numbering is conservative for LangChain: callbacks expose
    a workflow parent that can contain multiple unrelated LLM calls, so
    this handler emits attempt_1 / is_retry=false for each observed LLM
    call rather than inferring retry ordinals from the shared parent.

This handler is registered ALONGSIDE the existing
TraceloopCallbackHandler (not as a replacement) — it captures
metadata only (model, tokens, status), so its placement is not
safety-critical. The existing handler continues to do prompt /
completion attribute capture; backend dedup correctly
skips its per-attempt spans because they're non-retry siblings of
the llm_attempts emitted here.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any, Optional
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from opentelemetry import trace
from opentelemetry.instrumentation.fortifyroot import (
    FR_HAS_ATTEMPT_CHILD_KEY,
    first_llm_attempt,
    llm_attempt_attributes,
    register_framework_attempt,
    unregister_framework_attempt,
)
from opentelemetry.instrumentation.langchain.span_utils import _message_type_to_role
from opentelemetry.instrumentation.langchain.utils import (
    CallbackFilteredJSONEncoder,
    should_send_prompts,
)
from opentelemetry.instrumentation.langchain.version import __version__
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.trace import SpanKind, Status, StatusCode, set_span_in_context

logger = logging.getLogger(__name__)

# Per-attempt sibling span name + role.
_FR_SPAN_ROLE_KEY = "fortifyroot.span.role"
_FR_LLM_ATTEMPT_SPAN_NAME_PREFIX = "fortifyroot.langchain"
_FR_SPAN_ROLE_LLM_ATTEMPT = "llm_attempt"

# Parent marker.
_FR_HAS_ATTEMPT_CHILD_KEY = FR_HAS_ATTEMPT_CHILD_KEY


# ----------------------------------------------------------------------
# Per-attempt correlation map. Key = LangChain run_id (UUID per attempt
# on framework-layer retry paths). Value =
# {span, started_at_monotonic, framework_token, ended}. Bounded-size +
# TTL eviction defends against framework crashes that leave attempts
# open. Mirrors the LiteLLM map's shape for consistency across wrappers.
# ----------------------------------------------------------------------

_FR_RETRY_ATTEMPT_MAP: dict[UUID, dict] = {}
_FR_RETRY_ATTEMPT_MAP_LOCK = threading.Lock()
_FR_RETRY_ATTEMPT_MAP_MAX = 4096
_FR_RETRY_ATTEMPT_MAP_TTL_SEC = 60.0
_FR_RETRY_ATTEMPT_EVICT_BATCH = 1024
_FR_RETRY_ATTEMPT_EVICT_WARN_EVERY = 32
_FR_RETRY_ATTEMPT_EVICT_WARN_COUNTER = 0


# Per-parent-run-id parent-span memo. The first time we see a given
# parent_run_id, we capture the then-current ambient OTel span as the
# "parent" for retry_attempt sibling-grouping. Subsequent retries
# (which share parent_run_id under Runnable.with_retry) reuse the
# memoised span. This gives RetryDetectorProc a stable parent to group
# by even though each attempt has a fresh run_id.

_FR_PARENT_SPAN_BY_PARENT_RUN_ID: dict[UUID, "trace.Span"] = {}
_FR_PARENT_SPAN_LOCK = threading.Lock()
# Same TTL + cap as the attempt map (parent memo grows alongside it).
_FR_PARENT_SPAN_MAX = 4096
_FR_PARENT_SPAN_TTL_SEC = 60.0
_FR_PARENT_SPAN_INSERT_TIMES: dict[UUID, float] = {}


def _evict_stale_attempts_locked(now: float) -> int:
    """Drop entries older than the TTL. Caller MUST hold the map lock.
    Returns count evicted."""
    cutoff = now - _FR_RETRY_ATTEMPT_MAP_TTL_SEC
    stale = [k for k, v in _FR_RETRY_ATTEMPT_MAP.items() if v["started_at"] < cutoff]
    for k in stale:
        entry = _FR_RETRY_ATTEMPT_MAP.pop(k, None)
        if entry is None:
            continue
        try:
            sp = entry.get("span")
            if sp is not None and not entry.get("ended"):
                sp.set_status(Status(StatusCode.ERROR, "retry_attempt orphaned (framework crashed)"))
                sp.end()
        except Exception:
            pass
        try:
            unregister_framework_attempt(entry.get("framework_token"))
        except Exception:
            pass
    return len(stale)


def _enforce_attempt_max_locked() -> int:
    if len(_FR_RETRY_ATTEMPT_MAP) <= _FR_RETRY_ATTEMPT_MAP_MAX:
        return 0
    items = sorted(_FR_RETRY_ATTEMPT_MAP.items(), key=lambda kv: kv[1]["started_at"])
    to_drop = items[:_FR_RETRY_ATTEMPT_EVICT_BATCH]
    for k, entry in to_drop:
        _FR_RETRY_ATTEMPT_MAP.pop(k, None)
        try:
            sp = entry.get("span")
            if sp is not None and not entry.get("ended"):
                sp.set_status(Status(StatusCode.ERROR, "retry_attempt cap-evicted"))
                sp.end()
        except Exception:
            pass
        try:
            unregister_framework_attempt(entry.get("framework_token"))
        except Exception:
            pass
    return len(to_drop)


def _maybe_warn_attempt_eviction(evicted: int) -> None:
    global _FR_RETRY_ATTEMPT_EVICT_WARN_COUNTER
    _FR_RETRY_ATTEMPT_EVICT_WARN_COUNTER += evicted
    if _FR_RETRY_ATTEMPT_EVICT_WARN_COUNTER < _FR_RETRY_ATTEMPT_EVICT_WARN_EVERY:
        return
    _FR_RETRY_ATTEMPT_EVICT_WARN_COUNTER = 0
    logger.warning(
        "fortifyroot langchain retry_attempt map: evicted %d+ stale/over-cap entries; "
        "framework may be leaking attempts (TTL=%.0fs, max=%d)",
        evicted,
        _FR_RETRY_ATTEMPT_MAP_TTL_SEC,
        _FR_RETRY_ATTEMPT_MAP_MAX,
    )


def _evict_stale_parent_spans_locked(now: float) -> int:
    """Drop parent-span memos older than the TTL. Caller holds lock."""
    cutoff = now - _FR_PARENT_SPAN_TTL_SEC
    stale = [k for k, v in _FR_PARENT_SPAN_INSERT_TIMES.items() if v < cutoff]
    for k in stale:
        _FR_PARENT_SPAN_BY_PARENT_RUN_ID.pop(k, None)
        _FR_PARENT_SPAN_INSERT_TIMES.pop(k, None)
    return len(stale)


def _enforce_parent_span_max_locked() -> int:
    if len(_FR_PARENT_SPAN_BY_PARENT_RUN_ID) <= _FR_PARENT_SPAN_MAX:
        return 0
    items = sorted(_FR_PARENT_SPAN_INSERT_TIMES.items(), key=lambda kv: kv[1])
    to_drop = items[:_FR_RETRY_ATTEMPT_EVICT_BATCH]
    for k, _ in to_drop:
        _FR_PARENT_SPAN_BY_PARENT_RUN_ID.pop(k, None)
        _FR_PARENT_SPAN_INSERT_TIMES.pop(k, None)
    return len(to_drop)


def _resolve_parent_span(
    parent_run_id: Optional[UUID],
    traceloop_handler: Optional[Any] = None,
) -> Optional["trace.Span"]:
    """Resolve the OTel parent span for a retry_attempt under the given
    parent_run_id. Multi-attempt retries share parent_run_id, so the
    retry_attempts must share one OTel parent → RetryDetectorProc can
    group them.

    Resolution strategy:

    Strategy A — TRACELOOP SPANS-DICT LOOKUP (load-bearing when
    parent_run_id is set):
      The TraceloopCallbackHandler maintains a ``spans`` dict keyed
      by run_id → SpanHolder. When parent_run_id is set, we look up
      the parent's SpanHolder there and use its OTel span as the
      workflow parent. This gives the correct answer even when
      Traceloop has already attached its per-LLM span as the OTel
      ambient (which would otherwise be the visible "current span"
      and would break sibling-grouping across retries).

      The ``traceloop_handler`` is passed as a DIRECT reference at
      ``_FortifyRootRetryHandler`` construction time (NOT via a
      mutable shared back-reference to BaseCallbackManager). This
      avoids a single shared
      ``_FortifyRootRetryHandler`` instance was being mutated
      per-BaseCallbackManager-init, racing concurrent callbacks
      onto the wrong manager's spans dict.

    Strategy B — AMBIENT FALLBACK (root invocation OR no-Traceloop
    standalone use):
      Used when:
        (a) parent_run_id is None (root invocation with no enclosing
            chain / runnable), OR
        (b) no traceloop_handler reference was provided to the
            retry handler (standalone / test scenario — production
            ALWAYS wires this via LangchainInstrumentor).
      In both cases there is no risk of Traceloop having attached
      its per-LLM span as the OTel ambient, so ambient is the
      correct workflow parent.

    No-emission policy:
      If parent_run_id IS set AND traceloop_handler is provided
      (production wiring) BUT Strategy A's Traceloop lookup fails
      (parent_run_id not in its spans dict — e.g. evicted, stale),
      DO NOT fall back to ambient. Ambient at that moment is likely
      Traceloop's per-LLM span (handler-order is Traceloop-first)
      and parenting under it would break sibling-grouping. Return
      None; the caller skips emission with a debug log. Backend
      backend dedup degrades gracefully (no retry_attempt → parent
      stays canonical → single LLMUsageEvent per call).
    """
    # Strategy A: lookup Traceloop's parent SpanHolder via parent_run_id.
    if parent_run_id is not None and traceloop_handler is not None:
        try:
            span_holder = getattr(traceloop_handler, "spans", {}).get(parent_run_id)
            if span_holder is not None:
                span = getattr(span_holder, "span", None)
                if span is not None:
                    return span
        except Exception:
            logger.debug("Traceloop spans-dict lookup failed", exc_info=True)
        # If parent_run_id was set AND traceloop_handler was provided
        # but lookup failed, refuse to fall back to ambient — see
        # "No-emission policy" above.
        return None

    # Strategy B: ambient fallback.
    #   - parent_run_id is None (legitimate root), OR
    #   - traceloop_handler is None (standalone / test).
    current = trace.get_current_span()
    ctx = current.get_span_context() if current is not None else None
    if ctx is None or not ctx.is_valid:
        return None
    return current


def _resolve_routed_provider(serialized: Optional[dict], invocation_params: Optional[dict]) -> Optional[str]:
    """Best-effort: derive the routed provider for the gen_ai.system
    attribute. Falls back to inspecting the serialized handler /
    invocation_params dicts; returns None if undeterminable.

    LangChain's serialized dict typically contains 'id' = list of
    module path components, e.g. ['langchain_openai', 'chat_models',
    'base', 'ChatOpenAI']. The first segment ('langchain_openai',
    'langchain_anthropic', 'langchain_aws') maps to a provider name.
    """
    if isinstance(serialized, dict):
        ids = serialized.get("id")
        if isinstance(ids, list) and ids:
            first = str(ids[0]).lower()
            if first.startswith("langchain_"):
                provider = first[len("langchain_"):]
                # Common provider mappings.
                mapping = {
                    "openai": "openai",
                    "anthropic": "anthropic",
                    "aws": "AWS",  # Bedrock SDK
                    "google_genai": "google",
                    "google_vertexai": "google",
                }
                return mapping.get(provider, provider)
    if isinstance(invocation_params, dict):
        # Some langchain integrations expose an explicit provider field.
        cand = invocation_params.get("_type") or invocation_params.get("model_provider")
        if isinstance(cand, str):
            return cand
    return None


def _resolve_model(serialized: Optional[dict], invocation_params: Optional[dict]) -> Optional[str]:
    if isinstance(invocation_params, dict):
        for key in ("model", "model_name", "deployment_name"):
            v = invocation_params.get(key)
            if isinstance(v, str) and v:
                return v
    if isinstance(serialized, dict):
        kwargs = serialized.get("kwargs") or {}
        for key in ("model", "model_name", "deployment_name"):
            v = kwargs.get(key)
            if isinstance(v, str) and v:
                return v
    return None


def _content_to_string(content: Any) -> str:
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, cls=CallbackFilteredJSONEncoder)
    except Exception:
        return str(content)


def _add_prompt_attrs(
    attrs: dict[str, Any],
    *,
    messages: Optional[list[list[Any]]] = None,
    prompts: Optional[list[str]] = None,
) -> None:
    """Copy LangChain's request content onto the retry_attempt span.

    Backend dedup makes retry_attempt the canonical LLMUsageEvent span
    when it exists. Safety E2E tests and customers looking up the
    canonical event therefore still need the same prompt content that
    Traceloop's normal LLM span carries. The callback receives prompts
    after FR prompt-safety masking, so these attributes preserve the
    existing masked/plaintext semantics.
    """
    try:
        if not should_send_prompts():
            return
    except Exception:
        return

    if prompts is not None:
        for i, prompt in enumerate(prompts):
            if not isinstance(prompt, str):
                continue
            attrs[f"{GenAIAttributes.GEN_AI_PROMPT}.{i}.role"] = "user"
            attrs[f"{GenAIAttributes.GEN_AI_PROMPT}.{i}.content"] = prompt
        return

    if messages is None:
        return

    i = 0
    for message_group in messages:
        for msg in message_group:
            msg_type = getattr(msg, "type", None)
            if isinstance(msg_type, str):
                attrs[f"{GenAIAttributes.GEN_AI_PROMPT}.{i}.role"] = _message_type_to_role(msg_type)
            content = getattr(msg, "content", None)
            if content is not None:
                attrs[f"{GenAIAttributes.GEN_AI_PROMPT}.{i}.content"] = _content_to_string(content)
            i += 1


def _start_retry_attempt(
    run_id: UUID,
    parent_run_id: Optional[UUID],
    serialized: Optional[dict],
    invocation_params: Optional[dict],
    traceloop_handler: Optional[Any] = None,
    messages: Optional[list[list[Any]]] = None,
    prompts: Optional[list[str]] = None,
) -> None:
    """Open the retry_attempt sibling span and register state.
    Idempotent if called twice for the same run_id (defensive — the
    callback contract should fire it exactly once, but we don't crash
    on duplicates)."""
    if run_id is None:
        return
    parent_span = _resolve_parent_span(parent_run_id, traceloop_handler=traceloop_handler)
    if parent_span is None:
        # No ambient parent → orphan retry_attempt would have no place
        # in the trace tree. Skip emission. Backend dedup
        # degrades gracefully when no retry_attempt exists.
        logger.debug(
            "no ambient parent span for langchain retry_attempt; skipping emission "
            "(run_id=%s, parent_run_id=%s)", run_id, parent_run_id,
        )
        return

    routed_provider = _resolve_routed_provider(serialized, invocation_params)
    model = _resolve_model(serialized, invocation_params)
    span_name, attempt_number, is_retry = first_llm_attempt(
        _FR_LLM_ATTEMPT_SPAN_NAME_PREFIX,
    )

    attrs: dict[str, Any] = {
        "gen_ai.operation.name": "chat",
    }
    attrs.update(llm_attempt_attributes(attempt_number, is_retry))
    if routed_provider:
        attrs["gen_ai.system"] = routed_provider
    if model:
        attrs["gen_ai.request.model"] = model
    _add_prompt_attrs(attrs, messages=messages, prompts=prompts)

    tracer = trace.get_tracer(__name__, __version__)
    parent_ctx = set_span_in_context(parent_span)
    span = tracer.start_span(
        span_name,
        kind=SpanKind.CLIENT,
        attributes=attrs,
        context=parent_ctx,
    )

    try:
        framework_token = register_framework_attempt()
    except Exception:
        framework_token = None
        logger.debug("failed to register framework attempt token", exc_info=True)

    now = time.monotonic()
    with _FR_RETRY_ATTEMPT_MAP_LOCK:
        evicted = _evict_stale_attempts_locked(now)
        evicted += _enforce_attempt_max_locked()
        if evicted:
            _maybe_warn_attempt_eviction(evicted)
        # Defensive: if a duplicate run_id slips through, end the
        # previous span as ERROR (orphaned by the duplicate) and
        # replace.
        prev = _FR_RETRY_ATTEMPT_MAP.get(run_id)
        if prev is not None and not prev.get("ended"):
            try:
                prev["span"].set_status(Status(StatusCode.ERROR, "duplicate run_id; superseded"))
                prev["span"].end()
            except Exception:
                pass
            try:
                unregister_framework_attempt(prev.get("framework_token"))
            except Exception:
                pass
        _FR_RETRY_ATTEMPT_MAP[run_id] = {
            "span": span,
            "started_at": now,
            "framework_token": framework_token,
            "ended": False,
        }

    # Parent-marker timing: set AFTER the first qualifying llm_attempt
    # has successfully started under this parent. Idempotent — setting
    # the attribute twice on the same parent is a no-op.
    try:
        parent_span.set_attribute(_FR_HAS_ATTEMPT_CHILD_KEY, True)
    except Exception:
        logger.debug("failed to set has_attempt_child on parent", exc_info=True)


def _finalize_retry_attempt(
    run_id: UUID,
    *,
    success: bool,
    response: Any = None,
    error: Optional[BaseException] = None,
) -> None:
    """End the retry_attempt span associated with run_id. Idempotent."""
    if run_id is None:
        return
    with _FR_RETRY_ATTEMPT_MAP_LOCK:
        entry = _FR_RETRY_ATTEMPT_MAP.pop(run_id, None)
        if entry is None or entry.get("ended"):
            return
        entry["ended"] = True
        span = entry["span"]
        framework_token = entry.get("framework_token")

    try:
        if success:
            # Best-effort: pull model + token usage from LLMResult.
            try:
                llm_output = getattr(response, "llm_output", None) or {}
                if isinstance(llm_output, dict):
                    model_name = llm_output.get("model_name") or llm_output.get("model")
                    if isinstance(model_name, str) and model_name:
                        span.set_attribute("gen_ai.response.model", model_name)
                    usage = llm_output.get("token_usage") or llm_output.get("usage") or {}
                    if isinstance(usage, dict):
                        prompt_tokens = usage.get("prompt_tokens") or usage.get("input_tokens")
                        completion_tokens = usage.get("completion_tokens") or usage.get("output_tokens")
                        if isinstance(prompt_tokens, int):
                            span.set_attribute("gen_ai.usage.input_tokens", prompt_tokens)
                        if isinstance(completion_tokens, int):
                            span.set_attribute("gen_ai.usage.output_tokens", completion_tokens)
                # generations[0][0].generation_info often has model + finish reason.
                generations = getattr(response, "generations", None)
                if generations and generations[0]:
                    gen0 = generations[0][0]
                    gen_info = getattr(gen0, "generation_info", None) or {}
                    if isinstance(gen_info, dict):
                        rid = gen_info.get("response_id") or gen_info.get("id")
                        if isinstance(rid, str) and rid:
                            span.set_attribute("gen_ai.response.id", rid)
            except Exception:
                logger.debug("failed extracting response attrs", exc_info=True)
            span.set_status(Status(StatusCode.OK))
        else:
            if error is not None:
                error_type = type(error).__name__
                span.set_attribute("error.type", error_type)
                # Best-effort: pull HTTP status from the error if it
                # carries one (e.g. langchain-openai wraps openai's
                # APIStatusError, which has .status_code).
                status_code = getattr(error, "status_code", None) or getattr(
                    getattr(error, "response", None), "status_code", None
                )
                if isinstance(status_code, int):
                    span.set_attribute("http.status_code", status_code)
                try:
                    span.record_exception(error)
                except Exception:
                    pass
                span.set_status(Status(StatusCode.ERROR, str(error)))
            else:
                span.set_status(Status(StatusCode.ERROR, "retry_attempt failed"))
    finally:
        try:
            span.end()
        except Exception:
            logger.debug("failed to end langchain retry_attempt span", exc_info=True)
        try:
            unregister_framework_attempt(framework_token)
        except Exception:
            pass


class _FortifyRootRetryHandler(BaseCallbackHandler):
    """LangChain BaseCallbackHandler that emits one
    fortifyroot.langchain.attempt_<N> sibling span per LLM-start
    callback invocation. Hooks BOTH on_chat_model_start (chat models —
    observed chat-model path) AND on_llm_start (legacy
    completion LLMs).

    Key correlation: ``run_id`` (UUID minted per attempt by LangChain).
    The retry_attempt span is created as a CHILD of the parent_run_id's
    span (memoised on first sighting), so multiple attempts under the
    same parent_run_id become SIBLINGS that RetryDetectorProc can
    group on (parent_span_id, provider, model).
    """

    # LangChain's BaseCallbackHandler is class-based (not pydantic)
    # in langchain-core >=1.0; instance state is fine here. We use a
    # MODULE-level map for span tracking so that multiple callback
    # manager instances (one per Runnable invocation) all funnel
    # through the same correlation table — the LangChain wrapper
    # registers ONE _FortifyRootRetryHandler per Runnable thanks to
    # _BaseCallbackManagerInitWrapper, but module-level state survives
    # any handler-instance churn cleanly.

    raise_error: bool = False
    run_inline: bool = True

    # Set by ``_BaseCallbackManagerInitWrapper`` at construction time
    # (NOT mutated per-callback-manager-init). Direct reference to the sibling Traceloop handler whose
    # ``spans`` dict we look up by run_id to resolve the workflow
    # parent for sibling-grouping across multi-attempt retries. See
    # ``_resolve_parent_span``.
    _traceloop_handler: Any = None

    def __init__(self, traceloop_handler: Optional[Any] = None) -> None:
        super().__init__()
        self._traceloop_handler = traceloop_handler

    def on_chat_model_start(  # type: ignore[override]
        self,
        serialized,
        messages,
        *,
        run_id,
        parent_run_id=None,
        tags=None,
        metadata=None,
        invocation_params=None,
        **kwargs,
    ):
        try:
            _start_retry_attempt(
                run_id, parent_run_id, serialized, invocation_params,
                traceloop_handler=self._traceloop_handler,
                messages=messages,
            )
        except Exception:
            logger.debug("on_chat_model_start retry-attempt-start failed", exc_info=True)

    def on_llm_start(  # type: ignore[override]
        self,
        serialized,
        prompts,
        *,
        run_id,
        parent_run_id=None,
        tags=None,
        metadata=None,
        invocation_params=None,
        **kwargs,
    ):
        try:
            _start_retry_attempt(
                run_id, parent_run_id, serialized, invocation_params,
                traceloop_handler=self._traceloop_handler,
                prompts=prompts,
            )
        except Exception:
            logger.debug("on_llm_start retry-attempt-start failed", exc_info=True)

    def on_llm_end(  # type: ignore[override]
        self,
        response,
        *,
        run_id,
        parent_run_id=None,
        **kwargs,
    ):
        try:
            _finalize_retry_attempt(run_id, success=True, response=response)
        except Exception:
            logger.debug("on_llm_end retry-attempt-finalize failed", exc_info=True)

    def on_llm_error(  # type: ignore[override]
        self,
        error,
        *,
        run_id,
        parent_run_id=None,
        **kwargs,
    ):
        try:
            _finalize_retry_attempt(run_id, success=False, error=error)
        except Exception:
            logger.debug("on_llm_error retry-attempt-finalize failed", exc_info=True)


def _reset_state_for_test() -> None:
    """Test-only helper: clear all module state. Not part of the
    public contract; do not import outside tests."""
    with _FR_RETRY_ATTEMPT_MAP_LOCK:
        for entry in _FR_RETRY_ATTEMPT_MAP.values():
            try:
                if not entry.get("ended"):
                    entry["span"].end()
            except Exception:
                pass
        _FR_RETRY_ATTEMPT_MAP.clear()
    with _FR_PARENT_SPAN_LOCK:
        _FR_PARENT_SPAN_BY_PARENT_RUN_ID.clear()
        _FR_PARENT_SPAN_INSERT_TIMES.clear()


__all__ = [
    "_FortifyRootRetryHandler",
    "_FR_LLM_ATTEMPT_SPAN_NAME_PREFIX",
    "_FR_HAS_ATTEMPT_CHILD_KEY",
]

"""ST-10.3 retry-aware emission for LlamaIndex.

Per RETRY_LOOP.md §4.4 LlamaIndex row + §4.4.2 coverage limitation +
ST-10.0 C3 empirical verification (POC findings F4-F6 in
phase_st10_retryloop.txt):

  - LlamaIndex's ``OpenAI.chat()`` (and equivalents) fires the
    dispatcher span TWICE per HTTP attempt — once on the public
    ``chat()`` method and once on the inner ``_chat()`` (F4).
    A naive event-handler approach emits 2x retry_attempt children
    per attempt. ST-10.3 hooks the SPAN HANDLER (not the event
    handler) and filters to OUTER public methods only via a
    method-name whitelist + ``BaseLLM`` instance check.
  - Per-attempt firing is verified empirically on framework-layer
    retry paths (e.g. tenacity wrapping at the application layer).
    Provider-SDK-internal retries (e.g.
    ``OpenAI(max_retries=N).chat(...)``) fire dispatcher spans ONCE
    per logical call → §4.4.2 coverage limitation applies (same
    pattern as LiteLLM C1 / LangChain C2).
  - ``span_enter``/``span_exit``/``span_drop`` lifecycle hooks
    correlate cleanly: each enter has a matching exit OR drop, so
    we don't need separate event-side de-dup logic.
  - Use ambient OTel span at first attempt as the parent for
    sibling-grouping (RetryDetectorProc requires retry_attempts to
    be siblings under one parent_span_id).
  - Attempt numbering is conservative for LlamaIndex: dispatcher hooks
    expose a broad ambient workflow parent that can contain unrelated LLM
    calls, so this handler emits attempt_1 / is_retry=false for each
    observed outer LLM call rather than inferring retry ordinals from the
    shared parent.

This handler is registered alongside the existing
``OpenLLMetrySpanHandler`` — they capture orthogonal data
(OpenLLMetry: full per-call telemetry; FR retry-handler: per-attempt
sibling spans for retry detection only).
"""

from __future__ import annotations

import logging
import re
import threading
import time
from typing import Any, Optional

from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.instrumentation.span_handlers.base import BaseSpanHandler
from opentelemetry import trace
from opentelemetry.instrumentation.fortifyroot import (
    FR_HAS_ATTEMPT_CHILD_KEY,
    first_llm_attempt,
    llm_attempt_attributes,
    register_framework_attempt,
    unregister_framework_attempt,
)
from opentelemetry.instrumentation.llamaindex.version import __version__
from opentelemetry.trace import SpanKind, Status, StatusCode, set_span_in_context

logger = logging.getLogger(__name__)

# ST-10 §4.4: per-attempt sibling span name + role.
_FR_SPAN_ROLE_KEY = "fortifyroot.span.role"
_FR_LLM_ATTEMPT_SPAN_NAME_PREFIX = "fortifyroot.llamaindex"
_FR_SPAN_ROLE_LLM_ATTEMPT = "llm_attempt"

# ST-10 §4.5 parent marker.
_FR_HAS_ATTEMPT_CHILD_KEY = FR_HAS_ATTEMPT_CHILD_KEY

# Dispatcher span IDs follow the pattern "ClassName.method-uuid".
# Capture the method name so we can filter outer (public) calls
# from inner (underscore-prefixed) ones — F4 finding from POC.
_DISPATCHER_ID_RE = re.compile(r"^([A-Za-z][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)-")

# Whitelist of OUTER public LlamaIndex BaseLLM methods we want to
# track. The inner private methods (``_chat``, ``_complete``, etc.)
# do NOT appear here, so they're filtered out — the F4 de-dup.
# Sourced from llama_index.core.llms.llm.LLM public surface:
#   chat / achat / stream_chat / astream_chat
#   complete / acomplete / stream_complete / astream_complete
#   predict / apredict
#   structured_predict / astructured_predict /
#   stream_structured_predict / astream_structured_predict
_OUTER_LLM_METHODS = {
    "chat", "achat", "stream_chat", "astream_chat",
    "complete", "acomplete", "stream_complete", "astream_complete",
    "predict", "apredict",
    "structured_predict", "astructured_predict",
    "stream_structured_predict", "astream_structured_predict",
}


# ----------------------------------------------------------------------
# Per-attempt correlation map. Key = dispatcher span id_ (the OUTER
# span's id_, post-filter). Value = {span, started_at, framework_token,
# ended}. Bounded-size + TTL eviction defends against framework
# crashes.
# ----------------------------------------------------------------------

_FR_RETRY_ATTEMPT_MAP: dict[str, dict] = {}
_FR_RETRY_ATTEMPT_MAP_LOCK = threading.Lock()
_FR_RETRY_ATTEMPT_MAP_MAX = 4096
_FR_RETRY_ATTEMPT_MAP_TTL_SEC = 60.0
_FR_RETRY_ATTEMPT_EVICT_BATCH = 1024
_FR_RETRY_ATTEMPT_EVICT_WARN_EVERY = 32
_FR_RETRY_ATTEMPT_EVICT_WARN_COUNTER = 0


def _evict_stale_attempts_locked(now: float) -> int:
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


def _maybe_warn_eviction(evicted: int) -> None:
    global _FR_RETRY_ATTEMPT_EVICT_WARN_COUNTER
    _FR_RETRY_ATTEMPT_EVICT_WARN_COUNTER += evicted
    if _FR_RETRY_ATTEMPT_EVICT_WARN_COUNTER < _FR_RETRY_ATTEMPT_EVICT_WARN_EVERY:
        return
    _FR_RETRY_ATTEMPT_EVICT_WARN_COUNTER = 0
    logger.warning(
        "fortifyroot llamaindex retry_attempt map: evicted %d+ stale/over-cap entries; "
        "framework may be leaking attempts (TTL=%.0fs, max=%d)",
        evicted,
        _FR_RETRY_ATTEMPT_MAP_TTL_SEC,
        _FR_RETRY_ATTEMPT_MAP_MAX,
    )


def _is_outer_llm_method(id_: str, instance: Any) -> bool:
    """F4 de-dup filter: only emit retry_attempt for OUTER public
    LlamaIndex LLM methods. Returns False for:
      - non-LLM dispatcher spans (e.g. workflow / chain spans)
      - inner private methods (``_chat``, ``_complete``, etc.)
      - non-BaseLLM instances
    """
    if not isinstance(instance, BaseLLM):
        return False
    m = _DISPATCHER_ID_RE.match(id_)
    if m is None:
        return False
    method = m.group(2)
    return method in _OUTER_LLM_METHODS


def _resolve_routed_provider(instance: Any) -> Optional[str]:
    """Best-effort: derive the routed provider for gen_ai.system from
    the instance's class module path."""
    cls = type(instance)
    module_name = (cls.__module__ or "").lower()
    # Common shapes: ``llama_index.llms.openai.base``,
    # ``llama_index_llms_openai`` (newer), etc.
    if "openai" in module_name:
        return "openai"
    if "anthropic" in module_name:
        return "anthropic"
    if "bedrock" in module_name or "aws" in module_name:
        return "AWS"
    if "google" in module_name or "gemini" in module_name or "vertex" in module_name:
        return "google"
    return None


def _resolve_model(instance: Any) -> Optional[str]:
    for attr in ("model", "model_name", "deployment_name"):
        v = getattr(instance, attr, None)
        if isinstance(v, str) and v:
            return v
    return None


def _content_to_string(content: Any) -> str:
    if isinstance(content, str):
        return content
    try:
        import json
        return json.dumps(content, default=str)
    except Exception:
        return str(content)


def _add_prompt_attrs(attrs: dict[str, Any], bound_args: Any) -> None:
    """Copy request prompt content onto the retry_attempt span.

    Backend §4.5 makes retry_attempt the canonical LLMUsageEvent span
    when it exists, so safety correlation still needs prompt content on
    this span. LlamaIndex safety wrappers have already processed the
    bound arguments by the time dispatcher span handlers see them.
    """
    arguments = getattr(bound_args, "arguments", None)
    if not isinstance(arguments, dict):
        return

    prompt = arguments.get("prompt")
    if isinstance(prompt, str):
        attrs["gen_ai.prompt.0.role"] = "user"
        attrs["gen_ai.prompt.0.content"] = prompt
        return

    messages = arguments.get("messages")
    if not isinstance(messages, list):
        return
    for i, msg in enumerate(messages):
        role = getattr(msg, "role", None)
        if role is None and isinstance(msg, dict):
            role = msg.get("role")
        content = getattr(msg, "content", None)
        if content is None and isinstance(msg, dict):
            content = msg.get("content")
        if role is not None:
            attrs[f"gen_ai.prompt.{i}.role"] = str(role)
        if content is not None:
            attrs[f"gen_ai.prompt.{i}.content"] = _content_to_string(content)


def _start_retry_attempt(id_: str, instance: Any, bound_args: Any = None) -> None:
    parent_span = trace.get_current_span()
    if parent_span is None or not parent_span.get_span_context().is_valid:
        # No ambient parent → orphan retry_attempt would have no place
        # in the trace tree. Skip emission. Same graceful-degradation
        # contract as LiteLLM/LangChain.
        logger.debug(
            "no ambient parent span for llamaindex retry_attempt; skipping (id_=%s)",
            id_,
        )
        return

    routed_provider = _resolve_routed_provider(instance)
    model = _resolve_model(instance)
    operation = "chat"
    m = _DISPATCHER_ID_RE.match(id_)
    if m is not None:
        method = m.group(2)
        if "complete" in method:
            operation = "text_completion"
        elif "predict" in method:
            operation = "chat"  # treat as chat-shaped

    span_name, attempt_number, is_retry = first_llm_attempt(
        _FR_LLM_ATTEMPT_SPAN_NAME_PREFIX,
    )
    attrs: dict[str, Any] = {
        "gen_ai.operation.name": operation,
    }
    attrs.update(llm_attempt_attributes(attempt_number, is_retry))
    if routed_provider:
        attrs["gen_ai.system"] = routed_provider
    if model:
        attrs["gen_ai.request.model"] = model
    _add_prompt_attrs(attrs, bound_args)

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
            _maybe_warn_eviction(evicted)
        # Defensive: replace any prior entry with same id_ (shouldn't
        # happen — dispatcher span ids are UUID-suffixed — but
        # tolerated to avoid leaks if it does).
        prev = _FR_RETRY_ATTEMPT_MAP.get(id_)
        if prev is not None and not prev.get("ended"):
            try:
                prev["span"].set_status(Status(StatusCode.ERROR, "duplicate id_; superseded"))
                prev["span"].end()
            except Exception:
                pass
            try:
                unregister_framework_attempt(prev.get("framework_token"))
            except Exception:
                pass
        _FR_RETRY_ATTEMPT_MAP[id_] = {
            "span": span,
            "started_at": now,
            "framework_token": framework_token,
            "ended": False,
        }

    # §4.5 marker timing: set on parent AFTER the first qualifying
    # llm_attempt has successfully started under it.
    try:
        parent_span.set_attribute(_FR_HAS_ATTEMPT_CHILD_KEY, True)
    except Exception:
        logger.debug("failed to set has_attempt_child on parent", exc_info=True)


def _finalize_retry_attempt(
    id_: str,
    *,
    success: bool,
    result: Any = None,
    err: Optional[BaseException] = None,
) -> None:
    with _FR_RETRY_ATTEMPT_MAP_LOCK:
        entry = _FR_RETRY_ATTEMPT_MAP.pop(id_, None)
        if entry is None or entry.get("ended"):
            return
        entry["ended"] = True
        span = entry["span"]
        framework_token = entry.get("framework_token")

    try:
        if success:
            try:
                # Best-effort attribute extraction. ChatResponse / CompletionResponse
                # both expose .raw (provider-native), .message, .usage.
                raw = getattr(result, "raw", None)
                if raw is not None:
                    rid = getattr(raw, "id", None)
                    if isinstance(rid, str) and rid:
                        span.set_attribute("gen_ai.response.id", rid)
                    rmodel = getattr(raw, "model", None)
                    if isinstance(rmodel, str) and rmodel:
                        span.set_attribute("gen_ai.response.model", rmodel)
                # Usage may live on .raw.usage or .additional_kwargs.
                usage = getattr(raw, "usage", None) if raw is not None else None
                if usage is not None:
                    pt = getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", None)
                    ct = getattr(usage, "completion_tokens", None) or getattr(usage, "output_tokens", None)
                    if isinstance(pt, int):
                        span.set_attribute("gen_ai.usage.input_tokens", pt)
                    if isinstance(ct, int):
                        span.set_attribute("gen_ai.usage.output_tokens", ct)
            except Exception:
                logger.debug("failed extracting llamaindex result attrs", exc_info=True)
            span.set_status(Status(StatusCode.OK))
        else:
            if err is not None:
                error_type = type(err).__name__
                span.set_attribute("error.type", error_type)
                status_code = getattr(err, "status_code", None) or getattr(
                    getattr(err, "response", None), "status_code", None
                )
                if isinstance(status_code, int):
                    span.set_attribute("http.status_code", status_code)
                try:
                    span.record_exception(err)
                except Exception:
                    pass
                span.set_status(Status(StatusCode.ERROR, str(err)))
            else:
                span.set_status(Status(StatusCode.ERROR, "retry_attempt failed"))
    finally:
        try:
            span.end()
        except Exception:
            logger.debug("failed to end llamaindex retry_attempt span", exc_info=True)
        try:
            unregister_framework_attempt(framework_token)
        except Exception:
            pass


class _FortifyRootRetryHandler(BaseSpanHandler):
    """LlamaIndex SpanHandler that emits one
    ``fortifyroot.llamaindex.attempt_<N>`` sibling span per OUTER
    public LLM method invocation (chat/achat/complete/acomplete/...).

    De-dup vs the inner ``_chat``/``_complete`` spans is enforced by
    a method-name whitelist in ``_is_outer_llm_method`` (F4 finding
    from ST-10.0 C3 POC). So one HTTP attempt = exactly one
    retry_attempt span — even though LlamaIndex internally fires
    dispatcher spans on both the outer and inner methods.

    Registered ALONGSIDE the existing ``OpenLLMetrySpanHandler`` (not
    as a replacement). Order matters: this handler MUST be registered
    BEFORE OpenLLMetrySpanHandler so its ``span_enter`` fires while
    the OTel ambient context is still the user's enclosing span (and
    NOT yet the SpanHolder's per-call OTel span). This ensures all
    retry_attempts under one logical user-call become SIBLINGS under
    one OTel parent — the structural invariant RetryDetectorProc
    needs.
    """

    @classmethod
    def class_name(cls) -> str:
        return "FortifyRootRetryHandler"

    def new_span(  # type: ignore[override]
        self,
        id_: str,
        bound_args,
        instance: Optional[Any] = None,
        parent_span_id: Optional[str] = None,
        tags: Optional[dict] = None,
        **kwargs,
    ) -> None:
        # We don't store SpanHolder objects (the existing
        # OpenLLMetrySpanHandler does). We just open a retry_attempt
        # OTel span and remember the id_ → otel_span mapping in
        # _FR_RETRY_ATTEMPT_MAP. Returning None signals BaseSpanHandler
        # to not store the result.
        try:
            if not _is_outer_llm_method(id_, instance):
                return None
            _start_retry_attempt(id_, instance, bound_args)
        except Exception:
            logger.debug("new_span retry-attempt-start failed", exc_info=True)
        return None

    def prepare_to_exit_span(  # type: ignore[override]
        self,
        id_: str,
        bound_args,
        instance: Optional[Any] = None,
        result: Optional[Any] = None,
        **kwargs,
    ) -> None:
        try:
            if id_ in _FR_RETRY_ATTEMPT_MAP:
                _finalize_retry_attempt(id_, success=True, result=result)
        except Exception:
            logger.debug("prepare_to_exit_span retry-attempt-finalize failed", exc_info=True)
        return None

    def prepare_to_drop_span(  # type: ignore[override]
        self,
        id_: str,
        bound_args,
        instance: Optional[Any] = None,
        err: Optional[BaseException] = None,
        **kwargs,
    ) -> None:
        try:
            if id_ in _FR_RETRY_ATTEMPT_MAP:
                _finalize_retry_attempt(id_, success=False, err=err)
        except Exception:
            logger.debug("prepare_to_drop_span retry-attempt-finalize failed", exc_info=True)
        return None


def _reset_state_for_test() -> None:
    """Test-only helper: clear all module state."""
    with _FR_RETRY_ATTEMPT_MAP_LOCK:
        for entry in _FR_RETRY_ATTEMPT_MAP.values():
            try:
                if not entry.get("ended"):
                    entry["span"].end()
            except Exception:
                pass
        _FR_RETRY_ATTEMPT_MAP.clear()


__all__ = [
    "_FortifyRootRetryHandler",
    "_FR_LLM_ATTEMPT_SPAN_NAME_PREFIX",
    "_FR_HAS_ATTEMPT_CHILD_KEY",
    "_OUTER_LLM_METHODS",
]

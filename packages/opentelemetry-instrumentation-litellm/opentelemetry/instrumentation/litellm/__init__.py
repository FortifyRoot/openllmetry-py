"""OpenTelemetry LiteLLM instrumentation."""

import asyncio  # FR: async safety
import inspect
import logging
import threading
import time
from typing import Collection, Optional

from opentelemetry import context as context_api
from opentelemetry import trace
from opentelemetry.instrumentation.fortifyroot import (
    get_object_value,
    register_framework_attempt,
    unregister_framework_attempt,
)
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.litellm.safety import (
    apply_completion_safety,
    apply_prompt_safety,
    extract_prompt_texts,
    extract_text_content,
)
from opentelemetry.instrumentation.litellm.streaming_safety import (
    is_async_streaming_response,
    is_sync_streaming_response,
    wrap_async_streaming_response,
    wrap_sync_streaming_response,
)
from opentelemetry.instrumentation.litellm.version import __version__
from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY, unwrap
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import (
    LLMRequestTypeValues,
    SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
    SpanAttributes,
)
from opentelemetry.trace import SpanKind, Status, StatusCode, get_tracer, set_span_in_context
from wrapt import wrap_function_wrapper

logger = logging.getLogger(__name__)

_instruments = ("litellm >= 1.71.2, < 2",)

# FR safety wrapper span name and role attribute
_FR_SAFETY_SPAN_NAME = "fortifyroot.litellm.safety"
_FR_SPAN_ROLE_KEY = "fortifyroot.span.role"
_FR_SPAN_ROLE_VALUE = "safety_wrapper"

# Marker set on the safety_wrapper span when LiteLLM's native OTel callback
# will emit a separate ``litellm_request`` span. The FR backend's
# LLMUsageExtractor uses this marker to skip the safety_wrapper for dedup
# WITHOUT needing to see the child span in the same OTLP batch — important
# because ``disable_batch=True`` (SimpleSpanProcessor) sends each span in its
# own ResourceSpans and the existing parent→child correlation in
# ``buildSafetyWrapperDedupeSet`` cannot observe siblings across batches.
_FR_HAS_NATIVE_OTEL_CHILD_KEY = "fortifyroot.span.has_native_otel_child"

# ST-10 §4.5: marker on parent set AFTER the first qualifying retry_attempt
# child has started, so the FR backend's LLMUsageExtractor can dedup the
# parent + non-retry siblings cross-batch (mirrors has_native_otel_child).
_FR_HAS_RETRY_ATTEMPT_CHILD_KEY = "fortifyroot.span.has_retry_attempt_child"

# ST-10 §4.4: per-attempt sibling span emitted by _FortifyRootRetryEmitter
# under the safety_wrapper parent.
_FR_RETRY_ATTEMPT_SPAN_NAME = "fortifyroot.litellm.retry_attempt"
_FR_SPAN_ROLE_RETRY_ATTEMPT = "retry_attempt"
_FR_COMPLETION_SAFETY_MARKER = "_fortifyroot_completion_safety_applied"
_FR_COMPLETION_SAFETY_FALLBACK_MARKERS: set[tuple[int, type]] = set()
_FR_COMPLETION_SAFETY_MARKERS_LOCK = threading.Lock()
_FR_COMPLETION_SAFETY_MARKERS_MAX = 4096


def _native_otel_callback_active() -> bool:
    """Return True iff LiteLLM's native ``OpenTelemetry`` callback will
    emit a separate ``litellm_request`` span.

    LiteLLM's callback does not create that primary span when an ambient
    parent span already exists unless ``USE_OTEL_LITELLM_REQUEST_SPAN`` is
    enabled. FR always attaches its safety span as the ambient parent, so a
    registered callback alone is not enough to mark the safety span for
    backend dedup.
    """
    try:
        import litellm
        # LiteLLM's OpenTelemetry callback class lives at a known path.
        from litellm.integrations.opentelemetry import OpenTelemetry as _LiteLLMNativeOTel  # noqa: N814
        from litellm.secret_managers.main import get_secret_bool
    except ImportError:
        return False
    callbacks = getattr(litellm, "callbacks", None) or []
    try:
        has_native_callback = any(
            isinstance(cb, _LiteLLMNativeOTel) for cb in callbacks
        )
        if not has_native_callback:
            return False
        return bool(get_secret_bool("USE_OTEL_LITELLM_REQUEST_SPAN", False))
    except Exception:
        return False


def _completion_safety_already_applied(response) -> bool:
    try:
        if bool(getattr(response, _FR_COMPLETION_SAFETY_MARKER, False)):
            return True
    except Exception:
        pass

    key = (id(response), type(response))
    with _FR_COMPLETION_SAFETY_MARKERS_LOCK:
        return key in _FR_COMPLETION_SAFETY_FALLBACK_MARKERS


def _mark_completion_safety_applied(response) -> None:
    try:
        setattr(response, _FR_COMPLETION_SAFETY_MARKER, True)
        return
    except Exception:
        pass

    key = (id(response), type(response))
    with _FR_COMPLETION_SAFETY_MARKERS_LOCK:
        if len(_FR_COMPLETION_SAFETY_FALLBACK_MARKERS) >= _FR_COMPLETION_SAFETY_MARKERS_MAX:
            _FR_COMPLETION_SAFETY_FALLBACK_MARKERS.clear()
        _FR_COMPLETION_SAFETY_FALLBACK_MARKERS.add(key)


def _ensure_completion_safety_applied(span, response, request_type, span_name) -> None:
    """Apply completion safety once for a concrete LiteLLM response object.

    LiteLLM async callbacks run through a global worker and may execute after
    the wrapper has returned, after the event loop has changed, or during
    process teardown. Running completion safety during FR span finalization
    makes the customer-visible response deterministic. The callback path still
    calls this helper first when it wins the race, preserving the
    "FR logger before native OTel" ordering while preventing duplicate finding
    events and ended-span writes when the worker runs late.
    """
    if response is None or _completion_safety_already_applied(response):
        return
    try:
        apply_completion_safety(span, response, request_type, span_name)
    finally:
        _mark_completion_safety_applied(response)


async def _ensure_completion_safety_applied_async(
    span, response, request_type, span_name
) -> None:
    if response is None or _completion_safety_already_applied(response):
        return
    try:
        await asyncio.to_thread(
            apply_completion_safety, span, response, request_type, span_name
        )
    finally:
        _mark_completion_safety_applied(response)


_WRAPPED_METHODS = [
    ("litellm", "completion", False, False),
    ("litellm", "acompletion", True, False),
    ("litellm", "text_completion", False, True),
    ("litellm", "atext_completion", True, True),
    ("litellm.main", "completion", False, False),
    ("litellm.main", "acompletion", True, False),
    ("litellm.main", "text_completion", False, True),
    ("litellm.main", "atext_completion", True, True),
]


# Lazy-import LiteLLM's CustomLogger base for inheritance. We MUST inherit
# (not duck-type) because LiteLLM's dispatch loop gates every callback hook
# on ``isinstance(callback, CustomLogger)`` — see
# litellm_logging.py:1015 (log_pre_api_call), :1216, :1267, :2303
# (log_success_event), :2613 (async_log_success_event), etc. A duck-typed
# class is silently SKIPPED by the dispatch, so the retry emitter never
# fires and no retry_attempt spans get emitted.
#
# The fallback to ``object`` keeps the module importable even when litellm
# isn't installed (the instrumentor's ``instrumentation_dependencies`` check
# guards actual use).
#
# Discovered end-to-end during ST-10 review-batch-1 re-verification 2026-05-10
# after local vendoring. The pre-existing _FortifyRootCompletionLogger is
# also duck-typed (same latent bug) but its primary safety-masking path is
# synchronous inside _finalize_response, so its callback never firing is
# masked in production. The retry emitter has no such backup — purely
# callback-driven — which made this bug observable.
try:
    from litellm.integrations.custom_logger import CustomLogger as _LiteLLMCustomLoggerBase
except ImportError:
    _LiteLLMCustomLoggerBase = object  # type: ignore[assignment,misc]


# ----------------------------------------------------------------------
# ST-10 §4.4 / §4.3: retry-attempt sibling-span emission via a second
# LiteLLM CustomLogger.
# ----------------------------------------------------------------------
#
# The emitter opens one ``fortifyroot.litellm.retry_attempt`` sibling span
# per attempt-start callback fired by LiteLLM, and ends it on the matching
# success/failure callback. Per ST-10.0 C1 source-verified findings, this
# fires per-attempt only on the ``completion_with_retries(num_retries=N)``
# and ``Router(...)`` retry surfaces — see RETRY_LOOP.md §4.4.2 for the
# documented coverage limitation on the ``completion(num_retries=N)``
# path (where retries delegate to the underlying provider SDK and are
# invisible to LiteLLM's callback layer).

# Per-attempt correlation map (§4.3). Key = LiteLLM's per-call
# ``litellm_call_id`` (sufficient because each attempt at the
# observable surfaces gets a fresh ID — verified ST-10.0 C2 POC). Value =
# {span, started_at_monotonic, parent_span, framework_token, ended}.
# Bounded-size + TTL eviction defends against framework crashes that
# leave attempts open.
_FR_RETRY_ATTEMPT_MAP: dict[str, dict] = {}
_FR_RETRY_ATTEMPT_MAP_LOCK = threading.Lock()
_FR_RETRY_ATTEMPT_MAP_MAX = 4096
_FR_RETRY_ATTEMPT_MAP_TTL_SEC = 60.0
_FR_RETRY_ATTEMPT_EVICT_BATCH = 1024
# Rate-limit eviction warnings to avoid log floods on pathological leaks.
_FR_RETRY_ATTEMPT_EVICT_WARN_EVERY = 32
_FR_RETRY_ATTEMPT_EVICT_WARN_COUNTER = 0


def _evict_stale_retry_attempts_locked(now: float) -> int:
    """Drop entries older than the TTL. Caller MUST hold the map lock.
    Returns count evicted."""
    cutoff = now - _FR_RETRY_ATTEMPT_MAP_TTL_SEC
    stale = [k for k, v in _FR_RETRY_ATTEMPT_MAP.items() if v["started_at"] < cutoff]
    for k in stale:
        entry = _FR_RETRY_ATTEMPT_MAP.pop(k, None)
        if entry is None:
            continue
        # Best-effort: end the leaked span and unregister its framework token.
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


def _enforce_retry_attempt_max_locked() -> int:
    """Cap-evict the oldest _FR_RETRY_ATTEMPT_EVICT_BATCH entries when
    the map exceeds _FR_RETRY_ATTEMPT_MAP_MAX. Caller MUST hold the
    map lock. Returns count evicted."""
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


def _maybe_warn_retry_attempt_eviction(evicted: int) -> None:
    global _FR_RETRY_ATTEMPT_EVICT_WARN_COUNTER
    _FR_RETRY_ATTEMPT_EVICT_WARN_COUNTER += evicted
    if _FR_RETRY_ATTEMPT_EVICT_WARN_COUNTER < _FR_RETRY_ATTEMPT_EVICT_WARN_EVERY:
        return
    _FR_RETRY_ATTEMPT_EVICT_WARN_COUNTER = 0
    logger.warning(
        "fortifyroot litellm retry_attempt map: evicted %d+ stale/over-cap entries; "
        "framework may be leaking attempts (TTL=%.0fs, max=%d)",
        evicted,
        _FR_RETRY_ATTEMPT_MAP_TTL_SEC,
        _FR_RETRY_ATTEMPT_MAP_MAX,
    )


def _resolve_routed_provider(kwargs) -> Optional[str]:
    """Best-effort: derive the ROUTED provider (e.g. ``openai``) from
    LiteLLM kwargs for the retry_attempt span's ``gen_ai.system``
    attribute. Per RETRY_LOOP.md §4.2, this is the routed provider
    (NOT the framework name). Falls back to ``litellm`` if undetermined.
    """
    candidate = (
        kwargs.get("custom_llm_provider")
        or kwargs.get("provider")
        or (kwargs.get("model_response_object") and get_object_value(kwargs["model_response_object"], "model"))
    )
    raw: Optional[str] = None
    if isinstance(candidate, str) and "/" in candidate:
        # e.g. "openai/gpt-4o-mini" — take the prefix.
        raw = candidate.split("/", 1)[0]
    elif isinstance(candidate, str) and candidate:
        raw = candidate
    else:
        model = kwargs.get("model")
        if isinstance(model, str):
            model_lower = model.lower()
            if "/" in model:
                raw = model.split("/", 1)[0]
            elif model_lower.startswith("claude-"):
                # LiteLLM callbacks can pass Anthropic-routed calls as
                # bare Claude model ids (for example
                # ``claude-4-sonnet-20250514``) even when the public
                # call used ``anthropic/<model>``. Without this inference
                # retry_attempt spans fall back to gen_ai.system="litellm",
                # so the backend stores the canonical ST-10 event under the
                # framework rather than the routed provider.
                raw = "anthropic"
            elif model and "." in model:
                # No explicit provider AND no slash, but the model
                # carries a Bedrock-style prefix like
                # ``amazon.nova-lite-v1:0`` or
                # ``anthropic.claude-3-5-sonnet`` — pass to the
                # normaliser so it can recognise the prefix and
                # map to "AWS". Bare model strings without "." or
                # "/" (e.g. ``gpt-4o-mini``) carry no provider
                # signal; we return None so the wrapper falls
                # back to ``"litellm"`` rather than guessing.
                raw = model
    if raw is None:
        return None
    return _normalize_routed_provider(raw)


# Normalisation for the ``gen_ai.system`` attribute.
#
# Per RETRY_LOOP.md §4.2, the value MUST be the ROUTED provider, NOT
# the framework, AND it MUST be the canonical OTel-semconv form (e.g.
# Bedrock = ``"AWS"``). LiteLLM's ``custom_llm_provider`` field uses
# its own taxonomy (``"bedrock"``, ``"bedrock_converse"``,
# ``"sagemaker"``, ...), so we map those to the §4.2 canonical
# values and leave already-canonical values untouched. (Review-batch-1
# Minor 4 fix 2026-05-10 — keeps cross-wrapper consistency with
# LangChain's _resolve_routed_provider which already normalises
# langchain_aws → ``"AWS"``.)
_LITELLM_PROVIDER_NORMALISATION = {
    # AWS Bedrock variants → "AWS"
    "bedrock": "AWS",
    "bedrock_converse": "AWS",
    "amazon": "AWS",
    "aws": "AWS",
    # Google variants → "google" (gemini, vertex, ...)
    "gemini": "google",
    "vertex_ai": "google",
    "vertex": "google",
    "google_genai": "google",
    "google_generativeai": "google",
}


def _normalize_routed_provider(raw: str) -> str:
    """Map LiteLLM's provider taxonomy to RETRY_LOOP.md §4.2's
    routed-provider form. If no mapping applies, return the raw
    value lower-cased (matches OpenAI / Anthropic which already
    use the canonical form).
    """
    lower = raw.lower()
    if lower in _LITELLM_PROVIDER_NORMALISATION:
        return _LITELLM_PROVIDER_NORMALISATION[lower]
    # Bedrock model-prefix detection: LiteLLM model strings like
    # "anthropic.claude-3-5-sonnet-20241022-v2:0" or "amazon.titan..."
    # routed via Bedrock surface as ``custom_llm_provider="bedrock"``,
    # but defensive pattern matching catches edge cases.
    if lower.startswith(("amazon.", "anthropic.", "meta.", "ai21.", "cohere.", "mistral.")):
        # These prefixes appear on Bedrock model IDs.
        return "AWS"
    return lower


def _httpx_status_code(response_obj) -> Optional[int]:
    """Best-effort extract HTTP status code from a LiteLLM exception or response."""
    if response_obj is None:
        return None
    for attr in ("status_code", "http_status", "code"):
        v = getattr(response_obj, attr, None)
        if isinstance(v, int):
            return v
    response = getattr(response_obj, "response", None)
    if response is not None:
        return _httpx_status_code(response)
    return None


def _server_address(kwargs) -> Optional[str]:
    api_base = kwargs.get("api_base")
    if not isinstance(api_base, str) or not api_base:
        return None
    # api_base is typically like "https://api.openai.com/v1" — strip scheme/path.
    try:
        from urllib.parse import urlparse
        parsed = urlparse(api_base)
        return parsed.hostname
    except Exception:
        return None


def _add_retry_attempt_prompt_attrs(
    attrs: dict,
    kwargs: dict,
    *,
    is_text_completion: bool = False,
) -> None:
    """Copy request prompt content onto the retry_attempt span.

    Backend §4.5 makes retry_attempt the canonical LLMUsageEvent span
    when it exists. Safety correlation and masking assertions therefore
    need the same request content on the retry_attempt span that the
    safety_wrapper parent carries.
    """
    operation_is_text = is_text_completion or kwargs.get("text_completion")
    if operation_is_text:
        prompt = kwargs.get("prompt")
        for index, text in enumerate(extract_prompt_texts(prompt)):
            attrs[f"{SpanAttributes.LLM_PROMPTS}.{index}.role"] = "user"
            attrs[f"{SpanAttributes.LLM_PROMPTS}.{index}.content"] = text
        return

    messages = kwargs.get("messages")
    if not isinstance(messages, list):
        return
    for index, message in enumerate(messages):
        role = get_object_value(message, "role")
        content = extract_text_content(get_object_value(message, "content"))
        if role is not None:
            attrs[f"{SpanAttributes.LLM_PROMPTS}.{index}.role"] = str(role)
        if content:
            attrs[f"{SpanAttributes.LLM_PROMPTS}.{index}.content"] = content


def _start_retry_attempt_span(kwargs, *, is_text_completion: bool = False) -> None:
    """Open a retry_attempt sibling span under the current ambient FR
    parent span and register it in the correlation map. Idempotent:
    if a span already exists for this litellm_call_id, replace it
    (defensive — shouldn't happen on the per-attempt observable
    surfaces, but tolerated to handle edge cases without losing the
    parent-marker side effect)."""
    call_id = kwargs.get("litellm_call_id") if isinstance(kwargs, dict) else None
    if not call_id:
        # No correlation key → cannot match success/failure later. Skip.
        return

    parent = trace.get_current_span()
    if parent is None or not parent.get_span_context().is_valid:
        # No ambient FR parent — likely the FR safety_wrapper context
        # was never attached (e.g. user invoked LiteLLM's logger
        # directly without going through the wrapped completion).
        # Skip — parent-orphan retry_attempts have no meaningful place
        # in the trace tree.
        return

    routed_provider = _resolve_routed_provider(kwargs) or "litellm"
    model = kwargs.get("model")
    operation = "text_completion" if (is_text_completion or kwargs.get("text_completion")) else "chat"
    attrs = {
        GenAIAttributes.GEN_AI_SYSTEM: routed_provider,
        GenAIAttributes.GEN_AI_OPERATION_NAME: operation,
        _FR_SPAN_ROLE_KEY: _FR_SPAN_ROLE_RETRY_ATTEMPT,
    }
    if model is not None:
        attrs[GenAIAttributes.GEN_AI_REQUEST_MODEL] = str(model)
    server = _server_address(kwargs)
    if server:
        # Per RETRY_LOOP.md §4.2, attribute name is the OTel-standard
        # ``server.address`` (network namespace) — using the literal
        # key string for forward-compat across semconv lib changes.
        attrs["server.address"] = server
    _add_retry_attempt_prompt_attrs(attrs, kwargs, is_text_completion=is_text_completion)

    tracer = trace.get_tracer(__name__, __version__)
    # Set the parent context explicitly so the retry_attempt span is a
    # CHILD of the FR safety_wrapper, not a child of whatever happens
    # to be ambient (which IS the safety_wrapper here, but explicit is
    # safer for future refactoring).
    parent_ctx = set_span_in_context(parent)
    span = tracer.start_span(
        _FR_RETRY_ATTEMPT_SPAN_NAME,
        kind=SpanKind.CLIENT,
        attributes=attrs,
        context=parent_ctx,
    )

    # §4.7.1: register a framework-attempt token so direct-SDK
    # wrappers (OpenAI/Anthropic/Bedrock) suppress their own emission
    # while this attempt is in flight.
    try:
        framework_token = register_framework_attempt()
    except Exception:
        framework_token = None
        logger.debug("Failed to register framework attempt token", exc_info=True)

    now = time.monotonic()
    with _FR_RETRY_ATTEMPT_MAP_LOCK:
        # Defensive: TTL + cap eviction on every insert path.
        evicted = _evict_stale_retry_attempts_locked(now)
        evicted += _enforce_retry_attempt_max_locked()
        if evicted:
            _maybe_warn_retry_attempt_eviction(evicted)
        _FR_RETRY_ATTEMPT_MAP[call_id] = {
            "span": span,
            "parent": parent,
            "started_at": now,
            "framework_token": framework_token,
            "ended": False,
        }

    # §4.5 marker timing: set has_retry_attempt_child=true on the
    # parent ONLY AFTER the first qualifying retry_attempt has
    # successfully started. We just succeeded; mark the parent now.
    # Idempotent: setting the attribute twice on the same parent is a
    # no-op (OTel deduplicates).
    try:
        parent.set_attribute(_FR_HAS_RETRY_ATTEMPT_CHILD_KEY, True)
    except Exception:
        # Some span impls (e.g. a NonRecordingSpan during shutdown)
        # may reject set_attribute. Non-fatal; the in-batch dedup
        # path still fires from the in-batch retry_attempt children.
        logger.debug("Failed to set has_retry_attempt_child on parent", exc_info=True)


def _finalize_retry_attempt_span(
    kwargs,
    response_obj,
    *,
    success: bool,
) -> None:
    """End the retry_attempt span associated with this kwargs's
    ``litellm_call_id``. Idempotent: a no-op if the entry is already
    ended (handles the sync+async race where both
    ``log_success_event`` and ``async_log_success_event`` may fire)."""
    call_id = kwargs.get("litellm_call_id") if isinstance(kwargs, dict) else None
    if not call_id:
        return
    with _FR_RETRY_ATTEMPT_MAP_LOCK:
        entry = _FR_RETRY_ATTEMPT_MAP.get(call_id)
        if entry is None or entry.get("ended"):
            return
        entry["ended"] = True
        # Keep the entry around briefly — TTL sweep will clean it up,
        # OR the success/failure callback's symmetric counterpart can
        # short-circuit on the "ended" flag.
        # Actually pop now: the "ended" sentinel is only useful within
        # this critical section; outside it, popping is cleaner.
        _FR_RETRY_ATTEMPT_MAP.pop(call_id, None)
        span = entry["span"]
        framework_token = entry.get("framework_token")

    try:
        if success:
            response_model = get_object_value(response_obj, "model")
            if response_model:
                span.set_attribute(GenAIAttributes.GEN_AI_RESPONSE_MODEL, str(response_model))
            response_id = get_object_value(response_obj, "id")
            if response_id:
                span.set_attribute(GenAIAttributes.GEN_AI_RESPONSE_ID, str(response_id))
            usage = get_object_value(response_obj, "usage")
            input_tokens = get_object_value(usage, "prompt_tokens")
            if input_tokens is None:
                input_tokens = get_object_value(usage, "input_tokens")
            output_tokens = get_object_value(usage, "completion_tokens")
            if output_tokens is None:
                output_tokens = get_object_value(usage, "output_tokens")
            # Per §4.2 token-usage rule: SET when known, OMIT when unknown.
            if input_tokens is not None:
                span.set_attribute(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS, int(input_tokens))
            if output_tokens is not None:
                span.set_attribute(GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS, int(output_tokens))
            span.set_status(Status(StatusCode.OK))
        else:
            exception = kwargs.get("exception") if isinstance(kwargs, dict) else None
            status_code = _httpx_status_code(exception) or _httpx_status_code(response_obj)
            if status_code is not None:
                # Per RETRY_LOOP.md §4.2 the attribute name is the
                # OTel-standard ``http.status_code`` (legacy semconv;
                # backend extractor reads the literal key, not a
                # python-binding constant).
                span.set_attribute("http.status_code", int(status_code))
            if exception is not None:
                error_type = type(exception).__name__
                span.set_attribute("error.type", error_type)
                try:
                    span.record_exception(exception)
                except Exception:
                    pass
                span.set_status(Status(StatusCode.ERROR, str(exception)))
            else:
                span.set_status(Status(StatusCode.ERROR, "retry_attempt failed"))
    finally:
        try:
            span.end()
        except Exception:
            logger.debug("Failed to end retry_attempt span", exc_info=True)
        try:
            unregister_framework_attempt(framework_token)
        except Exception:
            pass


class _FortifyRootRetryEmitter(_LiteLLMCustomLoggerBase):
    """Second LiteLLM CustomLogger. Emits per-HTTP-attempt sibling
    spans (``fortifyroot.litellm.retry_attempt``) under the FR
    safety_wrapper parent.

    Registered AFTER ``_FortifyRootCompletionLogger`` in the LiteLLM
    callbacks list so that completion-safety masking still runs
    BEFORE retry_attempt finalization (the retry_attempt span captures
    request prompt content at start, then model/tokens/status at end).

    MUST inherit from ``litellm.integrations.custom_logger.CustomLogger``
    because LiteLLM's dispatch loop gates every callback hook on
    ``isinstance(callback, CustomLogger)``. A duck-typed class is
    silently skipped — see _LiteLLMCustomLoggerBase docstring above.
    """

    def log_pre_api_call(self, model, messages, kwargs):
        try:
            _start_retry_attempt_span(kwargs)
        except Exception:
            logger.debug("retry-attempt span start failed", exc_info=True)

    async def async_log_pre_api_call(self, model, messages, kwargs):
        # Defensive: LiteLLM's dispatch can fire either log_pre_api_call
        # OR async_log_pre_api_call depending on the call path (sync vs
        # async, callback registration timing). The _start helper is
        # idempotent on (litellm_call_id) — defensive replace path
        # tolerates duplicate fires — so making BOTH hooks the entry
        # point avoids missed attempts on async-only paths.
        # Review-round-2 Blocker 4 (2026-05-11): we previously only
        # implemented the sync hook; some LiteLLM call paths (acompletion
        # via certain routers) skipped the sync pre-call hook and our
        # retry_attempt span never opened, leaving the failure callback
        # with no entry to finalise.
        try:
            _start_retry_attempt_span(kwargs)
        except Exception:
            logger.debug("retry-attempt span async start failed", exc_info=True)

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        try:
            _finalize_retry_attempt_span(kwargs, response_obj, success=True)
        except Exception:
            logger.debug("retry-attempt span finalize-success failed", exc_info=True)

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        try:
            _finalize_retry_attempt_span(kwargs, response_obj, success=True)
        except Exception:
            logger.debug("retry-attempt span async finalize-success failed", exc_info=True)

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        try:
            _finalize_retry_attempt_span(kwargs, response_obj, success=False)
        except Exception:
            logger.debug("retry-attempt span finalize-failure failed", exc_info=True)

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        try:
            _finalize_retry_attempt_span(kwargs, response_obj, success=False)
        except Exception:
            logger.debug("retry-attempt span async finalize-failure failed", exc_info=True)


class _FortifyRootCompletionLogger:
    """LiteLLM duck-typed CustomLogger that masks completions before native OTel fires.

    Registered at litellm.callbacks[0] so it fires before LiteLLM's own OTel
    callback. Masks ``response_obj`` in-place for non-streaming calls only;
    streaming safety is applied per-chunk by streaming_safety.py.

    The FR safety span is current (via span_token) when this fires, so
    ``trace.get_current_span()`` returns FR's parent span for finding emission.
    """

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        # Streaming: per-chunk safety handled in streaming_safety.py; skip here.
        if kwargs.get("stream"):
            return
        is_tc = bool(kwargs.get("text_completion"))
        span_name = _span_name(kwargs, is_tc)
        request_type = _request_type(kwargs, is_tc)
        span = trace.get_current_span()
        try:
            _ensure_completion_safety_applied(
                span, response_obj, request_type, span_name
            )
        except Exception:
            pass

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        if kwargs.get("stream"):
            return
        is_tc = bool(kwargs.get("text_completion"))
        span_name = _span_name(kwargs, is_tc)
        request_type = _request_type(kwargs, is_tc)
        span = trace.get_current_span()
        try:
            await _ensure_completion_safety_applied_async(
                span, response_obj, request_type, span_name
            )
        except Exception:
            pass

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        pass

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        pass


class LiteLLMInstrumentor(BaseInstrumentor):
    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(__name__, __version__, tracer_provider)

        # Register FR's completion logger at position 0 so it fires before
        # LiteLLM's native OTel callback and masks response_obj in-place.
        try:
            import litellm
            if not isinstance(getattr(litellm, "callbacks", None), list):
                litellm.callbacks = []
            self._fr_logger = _FortifyRootCompletionLogger()
            litellm.callbacks.insert(0, self._fr_logger)
            # ST-10.1: register the retry-attempt emitter immediately
            # AFTER the completion logger so completion-safety masking
            # still runs first. The retry emitter captures metadata
            # only (model, tokens, status), so its placement is not
            # safety-critical. Insert at index 1 to keep ordering
            # deterministic regardless of customer-registered
            # callbacks.
            self._fr_retry_emitter = _FortifyRootRetryEmitter()
            litellm.callbacks.insert(1, self._fr_retry_emitter)
        except Exception:
            logger.debug("Failed to register FR LiteLLM callbacks")
            self._fr_logger = None
            self._fr_retry_emitter = None

        for module_name, func_name, is_async, is_text_completion in _WRAPPED_METHODS:
            wrapper = (
                _build_async_wrapper(tracer, is_text_completion)
                if is_async
                else _build_sync_wrapper(tracer, is_text_completion)
            )
            wrap_function_wrapper(module_name, func_name, wrapper)

    def _uninstrument(self, **kwargs):
        # Remove FR's loggers from litellm.callbacks. Order: retry
        # emitter first so it can no longer pin tokens, THEN the
        # completion logger.
        fr_retry_emitter = getattr(self, "_fr_retry_emitter", None)
        if fr_retry_emitter is not None:
            try:
                import litellm
                if isinstance(getattr(litellm, "callbacks", None), list):
                    litellm.callbacks.remove(fr_retry_emitter)
            except Exception:
                pass
            self._fr_retry_emitter = None

        fr_logger = getattr(self, "_fr_logger", None)
        if fr_logger is not None:
            try:
                import litellm
                if isinstance(getattr(litellm, "callbacks", None), list):
                    litellm.callbacks.remove(fr_logger)
            except Exception:
                pass
            self._fr_logger = None

        for module_name, func_name, _, _ in _WRAPPED_METHODS:
            try:
                unwrap(module_name, func_name)
            except Exception:
                logger.debug("Failed to unwrap %s.%s", module_name, func_name)


def _build_sync_wrapper(tracer, is_text_completion):
    def wrapper(wrapped, instance, args, kwargs):
        return _invoke_completion(
            tracer,
            wrapped,
            args,
            kwargs,
            is_text_completion=is_text_completion,
        )

    return wrapper


def _build_async_wrapper(tracer, is_text_completion):
    async def wrapper(wrapped, instance, args, kwargs):
        return await _invoke_acompletion(
            tracer,
            wrapped,
            args,
            kwargs,
            is_text_completion=is_text_completion,
        )

    return wrapper


def _invoke_completion(tracer, wrapped, args, kwargs, *, is_text_completion=False):
    if _should_skip_instrumentation(kwargs):
        return wrapped(*args, **kwargs)

    span_name = _span_name(kwargs, is_text_completion)
    request_type = _request_type(kwargs, is_text_completion)

    span_attrs = {
        GenAIAttributes.GEN_AI_SYSTEM: "litellm",
        SpanAttributes.LLM_REQUEST_TYPE: request_type,
        SpanAttributes.LLM_IS_STREAMING: bool(kwargs.get("stream")),
        _FR_SPAN_ROLE_KEY: _FR_SPAN_ROLE_VALUE,
    }
    if _native_otel_callback_active():
        # Hint to the FR backend that this safety_wrapper WILL have a
        # sibling litellm_request child emitted by LiteLLM's native OTel
        # callback — enables single-pass dedup in proc_llm_extractor.go
        # without needing to see the child in the same OTLP batch.
        span_attrs[_FR_HAS_NATIVE_OTEL_CHILD_KEY] = True
    span = tracer.start_span(
        _FR_SAFETY_SPAN_NAME,
        kind=SpanKind.CLIENT,
        attributes=span_attrs,
    )
    # Attach FR span as ambient OTel context so LiteLLM's native OTel callback
    # creates its litellm_request span as a child of this span.
    # Also set SUPPRESS key to prevent FR re-entry via wrapt.
    ctx = set_span_in_context(span)
    ctx = context_api.set_value(SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY, True, ctx)
    span_token = context_api.attach(ctx)

    try:
        _set_request_attributes(span, args, kwargs, is_text_completion)
        updated_args, updated_kwargs = apply_prompt_safety(
            span, args, kwargs, request_type, span_name
        )
        _set_prompt_attributes(span, updated_args, updated_kwargs, request_type, is_text_completion)
        response = wrapped(*updated_args, **updated_kwargs)
    except Exception as exc:
        context_api.detach(span_token)
        _record_span_error(span, exc)
        span.end()
        raise

    if inspect.isawaitable(response):
        context_api.detach(span_token)
        return _finalize_awaitable_response(span, response, request_type, span_name)

    if is_sync_streaming_response(updated_kwargs, response):
        # Keep span_token active during stream iteration so LiteLLM callbacks
        # (including native OTel) fire with FR's span as the ambient context.
        # The streaming wrapper detaches span_token in its finally block.
        return wrap_sync_streaming_response(
            span,
            response,
            request_type,
            span_name,
            _set_response_attributes,
            token=span_token,
        )

    context_api.detach(span_token)
    return _finalize_response(span, response, request_type, span_name)


async def _invoke_acompletion(
    tracer,
    wrapped,
    args,
    kwargs,
    *,
    is_text_completion=False,
):
    if _should_skip_instrumentation(kwargs):
        return await wrapped(*args, **kwargs)

    span_name = _span_name(kwargs, is_text_completion)
    request_type = _request_type(kwargs, is_text_completion)

    span_attrs = {
        GenAIAttributes.GEN_AI_SYSTEM: "litellm",
        SpanAttributes.LLM_REQUEST_TYPE: request_type,
        SpanAttributes.LLM_IS_STREAMING: bool(kwargs.get("stream")),
        _FR_SPAN_ROLE_KEY: _FR_SPAN_ROLE_VALUE,
    }
    if _native_otel_callback_active():
        span_attrs[_FR_HAS_NATIVE_OTEL_CHILD_KEY] = True
    span = tracer.start_span(
        _FR_SAFETY_SPAN_NAME,
        kind=SpanKind.CLIENT,
        attributes=span_attrs,
    )
    ctx = set_span_in_context(span)
    ctx = context_api.set_value(SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY, True, ctx)
    span_token = context_api.attach(ctx)

    try:
        _set_request_attributes(span, args, kwargs, is_text_completion)
        updated_args, updated_kwargs = await asyncio.to_thread(  # FR: async safety
            apply_prompt_safety,
            span, args, kwargs, request_type, span_name
        )
        _set_prompt_attributes(span, updated_args, updated_kwargs, request_type, is_text_completion)
        response = await wrapped(*updated_args, **updated_kwargs)
    except Exception as exc:
        context_api.detach(span_token)
        _record_span_error(span, exc)
        span.end()
        raise

    if is_async_streaming_response(updated_kwargs, response):
        # Keep span_token active during async stream iteration.
        return wrap_async_streaming_response(
            span,
            response,
            request_type,
            span_name,
            _set_response_attributes,
            token=span_token,
        )

    context_api.detach(span_token)
    return await _async_finalize_response(span, response, request_type, span_name)  # FR: async safety


def _should_skip_instrumentation(kwargs):
    return bool(
        context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY)
        or context_api.get_value(SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY)
    )


def _span_name(kwargs, is_text_completion):
    if is_text_completion or kwargs.get("text_completion"):
        return "litellm.text_completion"
    return "litellm.completion"


def _request_type(kwargs, is_text_completion):
    if is_text_completion or kwargs.get("text_completion"):
        return LLMRequestTypeValues.COMPLETION.value
    return LLMRequestTypeValues.CHAT.value


def _set_request_attributes(span, args, kwargs, is_text_completion):
    model = kwargs.get("model")
    if model is None and args:
        if is_text_completion:
            model = args[1] if len(args) > 1 else None
        else:
            model = args[0]
    if model is not None:
        span.set_attribute(GenAIAttributes.GEN_AI_REQUEST_MODEL, str(model))

    user = kwargs.get("user")
    if user is not None:
        span.set_attribute(SpanAttributes.LLM_USER, str(user))

    custom_provider = kwargs.get("custom_llm_provider")
    if custom_provider is not None:
        span.set_attribute("litellm.request.provider", str(custom_provider))


def _set_prompt_attributes(span, args, kwargs, request_type, is_text_completion):
    if request_type == LLMRequestTypeValues.COMPLETION.value:
        prompt = kwargs.get("prompt")
        if prompt is None and args:
            prompt = args[0]

        prompt_texts = extract_prompt_texts(prompt)
        for index, text in enumerate(prompt_texts):
            span.set_attribute(f"{SpanAttributes.LLM_PROMPTS}.{index}.role", "user")
            span.set_attribute(f"{SpanAttributes.LLM_PROMPTS}.{index}.content", text)
        return

    messages = kwargs.get("messages")
    if messages is None and len(args) > 1:
        messages = args[1]
    if not isinstance(messages, list):
        return

    for index, message in enumerate(messages):
        role = get_object_value(message, "role")
        content = extract_text_content(get_object_value(message, "content"))
        if role is not None:
            span.set_attribute(f"{SpanAttributes.LLM_PROMPTS}.{index}.role", str(role))
        if content:
            span.set_attribute(
                f"{SpanAttributes.LLM_PROMPTS}.{index}.content",
                content,
            )


def _set_response_attributes(span, response):
    response_model = get_object_value(response, "model")
    if response_model is not None:
        span.set_attribute(GenAIAttributes.GEN_AI_RESPONSE_MODEL, str(response_model))

    usage = get_object_value(response, "usage")
    input_tokens = get_object_value(usage, "prompt_tokens")
    output_tokens = get_object_value(usage, "completion_tokens")
    total_tokens = get_object_value(usage, "total_tokens")

    if input_tokens is not None:
        span.set_attribute(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS, int(input_tokens))
    if output_tokens is not None:
        span.set_attribute(
            GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS,
            int(output_tokens),
        )
    if total_tokens is not None:
        span.set_attribute(SpanAttributes.LLM_USAGE_TOTAL_TOKENS, int(total_tokens))

    choices = get_object_value(response, "choices") or []
    for index, choice in enumerate(choices):
        finish_reason = get_object_value(choice, "finish_reason")
        if finish_reason is not None:
            span.set_attribute(
                f"{SpanAttributes.LLM_COMPLETIONS}.{index}.finish_reason",
                str(finish_reason),
            )

        message = get_object_value(choice, "message")
        role = get_object_value(message, "role")
        content = extract_text_content(get_object_value(message, "content"))
        if role is not None:
            span.set_attribute(
                f"{SpanAttributes.LLM_COMPLETIONS}.{index}.role",
                str(role),
            )
        if content:
            span.set_attribute(
                f"{SpanAttributes.LLM_COMPLETIONS}.{index}.content",
                content,
            )
            continue

        text = get_object_value(choice, "text")
        if isinstance(text, str) and text:
            span.set_attribute(
                f"{SpanAttributes.LLM_COMPLETIONS}.{index}.content",
                text,
            )


async def _finalize_awaitable_response(
    span,
    response,
    request_type,
    span_name,
):
    # Re-attach FR span as ambient context AND suppress re-entry while awaiting,
    # so LiteLLM callbacks (including the completion logger) fire with the
    # correct span as current, enabling accurate finding emission.
    ctx = set_span_in_context(span)
    ctx = context_api.set_value(SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY, True, ctx)
    token = context_api.attach(ctx)
    try:
        awaited_response = await response
    except Exception as exc:
        _record_span_error(span, exc)
        span.end()
        raise
    finally:
        context_api.detach(token)

    return await _async_finalize_response(span, awaited_response, request_type, span_name)  # FR: async safety


async def _async_finalize_response(span, response, request_type, span_name):  # FR: async safety
    """Async variant of _finalize_response with deterministic completion safety."""  # FR: async safety
    try:  # FR: async safety
        _ensure_completion_safety_applied(span, response, request_type, span_name)  # FR: async safety
        _set_response_attributes(span, response)  # FR: async safety
        span.set_status(Status(StatusCode.OK))  # FR: async safety
        return response  # FR: async safety
    finally:  # FR: async safety
        span.end()  # FR: async safety


def _finalize_response(span, response, request_type, span_name):
    """Finalize a non-streaming response with deterministic completion safety."""
    try:
        _ensure_completion_safety_applied(span, response, request_type, span_name)
        _set_response_attributes(span, response)
        span.set_status(Status(StatusCode.OK))
        return response
    finally:
        span.end()


def _record_span_error(span, exc):
    span.record_exception(exc)
    span.set_status(Status(StatusCode.ERROR, str(exc)))


__all__ = [
    "LiteLLMInstrumentor",
    "_invoke_acompletion",
    "_invoke_completion",
]

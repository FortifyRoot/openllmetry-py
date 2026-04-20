"""OpenTelemetry LiteLLM instrumentation."""

import asyncio  # FR: async safety
import inspect
import logging
import threading
from typing import Collection

from opentelemetry import context as context_api
from opentelemetry import trace
from opentelemetry.instrumentation.fortifyroot import get_object_value
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
        except Exception:
            logger.debug("Failed to register _FortifyRootCompletionLogger")
            self._fr_logger = None

        for module_name, func_name, is_async, is_text_completion in _WRAPPED_METHODS:
            wrapper = (
                _build_async_wrapper(tracer, is_text_completion)
                if is_async
                else _build_sync_wrapper(tracer, is_text_completion)
            )
            wrap_function_wrapper(module_name, func_name, wrapper)

    def _uninstrument(self, **kwargs):
        # Remove FR's logger from litellm.callbacks
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

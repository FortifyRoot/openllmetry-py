from __future__ import annotations

import inspect
import time
from types import SimpleNamespace

from opentelemetry import context as context_api
from opentelemetry.trace.status import Status, StatusCode

from opentelemetry.instrumentation.fortifyroot import get_object_value, set_object_value
from opentelemetry.instrumentation.fortifyroot.text_streaming import (
    CompletionTextStreamGroup,
)
from opentelemetry.instrumentation.litellm.safety import PROVIDER, extract_text_content

FR_STREAMING_TIME_TO_FIRST_TOKEN_MS = "fortifyroot.llm.streaming.time_to_first_token_ms"
FR_STREAMING_TIME_TO_GENERATE_MS = "fortifyroot.llm.streaming.time_to_generate_ms"


def _elapsed_seconds(start_time, end_time):
    return max(0, end_time - start_time)


def is_sync_streaming_response(kwargs, response) -> bool:
    """Check if this is a streaming response."""
    if kwargs.get("stream"):
        return (
            not inspect.iscoroutine(response)
            and not inspect.isasyncgen(response)
        )
    return False


def is_async_streaming_response(kwargs, response) -> bool:
    """Check if this is an async streaming response."""
    if kwargs.get("stream"):
        return (
            inspect.iscoroutine(response)
            or inspect.isasyncgen(response)
            or hasattr(response, "__aiter__")
        )
    return False


def wrap_sync_streaming_response(
    span,
    response,
    request_type,
    span_name,
    set_response_attributes,
    token=None,
    set_canonical_span_attribute=None,
):
    """Wrap a sync streaming response with per-chunk safety and span lifecycle.

    ``token`` is the OTel context token established by ``_invoke_completion``
    that keeps FR's safety span as the ambient span during stream iteration.
    It is detached here in the ``finally`` block, after all LiteLLM callbacks
    (including native OTel) have fired for the last chunk.
    """
    streams = CompletionTextStreamGroup(
        span=span,
        provider=PROVIDER,
        span_name=span_name,
        request_type=request_type,
    )
    complete_response = {"choices": [], "model": None, "usage": None}
    stream_started_at = time.perf_counter()
    first_token_at = None

    try:
        for chunk in response:
            _mask_streaming_chunk(streams, chunk)
            _accumulate_streaming_chunk(complete_response, chunk)
            if first_token_at is None and _chunk_has_output_text(chunk):
                first_token_at = time.perf_counter()
                _set_streaming_latency_attr(
                    span,
                    FR_STREAMING_TIME_TO_FIRST_TOKEN_MS,
                    _elapsed_seconds(stream_started_at, first_token_at),
                    set_canonical_span_attribute,
                )
            yield chunk
        if first_token_at is not None:
            _set_streaming_latency_attr(
                span,
                FR_STREAMING_TIME_TO_GENERATE_MS,
                _elapsed_seconds(first_token_at, time.perf_counter()),
                set_canonical_span_attribute,
            )
        _finalize_streaming_span(span, complete_response, set_response_attributes)
    except Exception as exc:
        _close_streaming_response(response)
        _record_span_error(span, exc)
        raise
    finally:
        if token is not None:
            context_api.detach(token)
        if span.is_recording():
            span.end()


async def wrap_async_streaming_response(
    span,
    response,
    request_type,
    span_name,
    set_response_attributes,
    token=None,
    set_canonical_span_attribute=None,
):
    """Wrap an async streaming response with per-chunk safety and span lifecycle.

    ``token`` is detached in the ``finally`` block after the stream is
    exhausted, keeping FR's safety span as ambient context until that point.
    """
    streams = CompletionTextStreamGroup(
        span=span,
        provider=PROVIDER,
        span_name=span_name,
        request_type=request_type,
    )
    complete_response = {"choices": [], "model": None, "usage": None}
    stream_started_at = time.perf_counter()
    first_token_at = None

    try:
        async for chunk in response:
            _mask_streaming_chunk(streams, chunk)
            _accumulate_streaming_chunk(complete_response, chunk)
            if first_token_at is None and _chunk_has_output_text(chunk):
                first_token_at = time.perf_counter()
                _set_streaming_latency_attr(
                    span,
                    FR_STREAMING_TIME_TO_FIRST_TOKEN_MS,
                    _elapsed_seconds(stream_started_at, first_token_at),
                    set_canonical_span_attribute,
                )
            yield chunk
        if first_token_at is not None:
            _set_streaming_latency_attr(
                span,
                FR_STREAMING_TIME_TO_GENERATE_MS,
                _elapsed_seconds(first_token_at, time.perf_counter()),
                set_canonical_span_attribute,
            )
        _finalize_streaming_span(span, complete_response, set_response_attributes)
    except Exception as exc:
        await _aclose_streaming_response(response)
        _record_span_error(span, exc)
        raise
    finally:
        if token is not None:
            context_api.detach(token)
        if span.is_recording():
            span.end()


def _mask_streaming_chunk(streams, chunk):
    for index, choice in enumerate(get_object_value(chunk, "choices") or []):
        finish_reason = get_object_value(choice, "finish_reason")
        message = get_object_value(choice, "message")
        content = extract_text_content(get_object_value(message, "content")) if message is not None else None
        if isinstance(content, str):
            masked = streams.process(
                ("choice", index),
                content,
                segment_index=index,
                segment_role="assistant",
            )
            set_object_value(message, "content", masked)
            if get_object_value(choice, "text") is not None:
                set_object_value(choice, "text", masked)
        else:
            text = get_object_value(choice, "text")
            if isinstance(text, str):
                masked = streams.process(
                    ("choice", index),
                    text,
                    segment_index=index,
                    segment_role="assistant",
                )
                set_object_value(choice, "text", masked)
        if finish_reason:
            tail = streams.flush(("choice", index))
            if tail:
                if message is not None:
                    current_content = extract_text_content(get_object_value(message, "content"))
                    if current_content is None:
                        current_content = ""
                    set_object_value(message, "content", f"{current_content}{tail}")
                if message is None or get_object_value(choice, "text") is not None:
                    current_text = get_object_value(choice, "text") or ""
                    set_object_value(choice, "text", f"{current_text}{tail}")


def _accumulate_streaming_chunk(complete_response, chunk):
    model = get_object_value(chunk, "model")
    if model is not None:
        complete_response["model"] = model
    usage = get_object_value(chunk, "usage")
    if usage is not None:
        complete_response["usage"] = usage

    for index, choice in enumerate(get_object_value(chunk, "choices") or []):
        while len(complete_response["choices"]) <= index:
            complete_response["choices"].append(
                {"message": {"role": "assistant", "content": ""}, "text": ""}
            )
        aggregate = complete_response["choices"][index]
        finish_reason = get_object_value(choice, "finish_reason")
        if finish_reason is not None:
            aggregate["finish_reason"] = finish_reason

        message = get_object_value(choice, "message")
        if message is not None:
            role = get_object_value(message, "role")
            if role is not None:
                aggregate["message"]["role"] = role
            content = extract_text_content(get_object_value(message, "content"))
            if isinstance(content, str):
                aggregate["message"]["content"] += content
                aggregate["text"] += content
                continue

        text = get_object_value(choice, "text")
        if isinstance(text, str):
            aggregate["text"] += text


def _finalize_streaming_span(span, complete_response, set_response_attributes):
    response = SimpleNamespace(
        model=complete_response["model"],
        usage=complete_response["usage"],
        choices=[
            SimpleNamespace(
                finish_reason=choice.get("finish_reason"),
                message=SimpleNamespace(
                    role=get_object_value(choice.get("message"), "role") or "assistant",
                    content=get_object_value(choice.get("message"), "content"),
                ),
                text=choice.get("text"),
            )
            for choice in complete_response["choices"]
        ],
    )
    set_response_attributes(span, response)
    span.set_status(Status(StatusCode.OK))


def _record_span_error(span, exc):
    span.record_exception(exc)
    span.set_status(Status(StatusCode.ERROR, str(exc)))


def _close_streaming_response(response) -> None:
    close = getattr(response, "close", None)
    if callable(close):
        close()


async def _aclose_streaming_response(response) -> None:
    aclose = getattr(response, "aclose", None)
    if callable(aclose):
        await aclose()


def _set_streaming_latency_attr(
    span,
    key: str,
    seconds: float,
    set_canonical_span_attribute=None,
) -> None:
    value = int(round(seconds * 1000))
    if span.is_recording():
        span.set_attribute(key, value)
    if set_canonical_span_attribute is not None:
        set_canonical_span_attribute(span, key, value)


def _chunk_has_output_text(chunk) -> bool:
    for choice in get_object_value(chunk, "choices") or []:
        if _value_has_text(get_object_value(choice, "text")):
            return True
        if _value_has_text(get_object_value(choice, "content")):
            return True
        message = get_object_value(choice, "message")
        if message is not None and _value_has_text(get_object_value(message, "content")):
            return True
        delta = get_object_value(choice, "delta")
        if delta is not None and _value_has_text(get_object_value(delta, "content")):
            return True
    return False


def _value_has_text(value) -> bool:
    text = extract_text_content(value)
    return isinstance(text, str) and text != ""

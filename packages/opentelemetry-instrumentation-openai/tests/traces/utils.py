import httpx
from opentelemetry.sdk.trace import Span
from opentelemetry.trace import StatusCode
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from opentelemetry.trace.propagation import get_current_span
from unittest.mock import MagicMock


# from: https://stackoverflow.com/a/41599695/2749989
def spy_decorator(method_to_decorate):
    mock = MagicMock()

    def wrapper(self, *args, **kwargs):
        mock(*args, **kwargs)
        return method_to_decorate(self, *args, **kwargs)

    wrapper.mock = mock
    return wrapper


def assert_request_contains_tracecontext(request: httpx.Request, expected_span: Span):
    assert TraceContextTextMapPropagator._TRACEPARENT_HEADER_NAME in request.headers
    ctx = TraceContextTextMapPropagator().extract(request.headers)
    request_span_context = get_current_span(ctx).get_span_context()
    expected_span_context = expected_span.get_span_context()

    assert request_span_context.trace_id == expected_span_context.trace_id
    assert request_span_context.span_id == expected_span_context.span_id


def assert_openai_exception_span(span: Span):
    assert span.status.status_code == StatusCode.ERROR
    assert span.status.description

    events = span.events
    assert len(events) == 1

    event = events[0]
    assert event.name == "exception"

    exception_type = event.attributes["exception.type"]
    assert exception_type in {
        "openai.AuthenticationError",
        "openai.APIConnectionError",
    }

    assert event.attributes["exception.message"] == span.status.description

    error_type = span.attributes.get("error.type")
    assert error_type in {"AuthenticationError", "APIConnectionError"}
    assert error_type == exception_type.split(".")[-1]

    stacktrace = event.attributes["exception.stacktrace"]
    assert "Traceback (most recent call last):" in stacktrace
    assert exception_type in stacktrace

    if exception_type == "openai.AuthenticationError":
        assert "invalid_api_key" in stacktrace

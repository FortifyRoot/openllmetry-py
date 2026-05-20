"""FR: Tests for _FortifyRootCompletionLogger.

Verifies that the duck-typed CustomLogger:
- Applies completion safety for non-streaming calls
- Skips completion safety for streaming calls (handled per-chunk elsewhere)
- Uses the current OTel span for finding emission
- Non-text content types (image, audio) are a no-op
- async variant offloads safety to a thread
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from opentelemetry import context as context_api
from opentelemetry import trace
from opentelemetry.instrumentation.fortifyroot import (
    SafetyFinding,
    SafetyLocation,
    SafetyResult,
    clear_safety_handlers,
    register_completion_safety_handler,
)
from opentelemetry.instrumentation.litellm import (
    LiteLLMInstrumentor,
    _FortifyRootCompletionLogger,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import set_span_in_context
from opentelemetry.semconv_ai import SpanAttributes

pytestmark = pytest.mark.fr


def setup_function():
    clear_safety_handlers()


def teardown_function():
    clear_safety_handlers()


def _make_tracer():
    exp = InMemorySpanExporter()
    prov = TracerProvider()
    prov.add_span_processor(SimpleSpanProcessor(exp))
    return exp, prov.get_tracer(__name__)


def _completion_handler(masked_text, context):
    return SafetyResult(
        text=masked_text,
        overall_action="MASK",
        findings=[
            SafetyFinding(
                category="SECRET",
                severity="HIGH",
                action="MASK",
                rule_name="SECRET.token",
                start=0,
                end=len(context.text),
            )
        ],
    )


# ---------------------------------------------------------------------------
# Logger skips streaming calls
# ---------------------------------------------------------------------------

def test_logger_skips_streaming():
    """log_success_event must be a no-op when stream=True."""
    register_completion_safety_handler(
        lambda context: _completion_handler("[MASKED]", context)
    )
    logger = _FortifyRootCompletionLogger()
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="secret"), text="secret")]
    )
    logger.log_success_event({"stream": True}, response, None, None)
    # Safety must NOT have been applied (streaming is handled per-chunk).
    assert response.choices[0].message.content == "secret"


@pytest.mark.asyncio
async def test_async_logger_skips_streaming():
    """async_log_success_event must be a no-op when stream=True."""
    register_completion_safety_handler(
        lambda context: _completion_handler("[MASKED]", context)
    )
    logger = _FortifyRootCompletionLogger()
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="secret"), text="secret")]
    )
    await logger.async_log_success_event({"stream": True}, response, None, None)
    assert response.choices[0].message.content == "secret"


# ---------------------------------------------------------------------------
# Logger applies completion safety for non-streaming calls
# ---------------------------------------------------------------------------

def test_logger_masks_non_streaming_completion():
    """log_success_event masks response_obj in-place for non-streaming calls."""
    register_completion_safety_handler(
        lambda context: _completion_handler("[SECRET.token]", context)
        if context.text == "token-abc"
        else None
    )
    logger = _FortifyRootCompletionLogger()
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="token-abc"),
                text="token-abc",
            )
        ]
    )
    logger.log_success_event({}, response, None, None)
    assert response.choices[0].message.content == "[SECRET.token]"


@pytest.mark.asyncio
async def test_async_logger_masks_non_streaming_completion():
    """async_log_success_event masks response_obj in-place for non-streaming calls."""
    register_completion_safety_handler(
        lambda context: _completion_handler("[SECRET.token]", context)
        if context.text == "token-abc"
        else None
    )
    logger = _FortifyRootCompletionLogger()
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="token-abc"),
                text="token-abc",
            )
        ]
    )
    await logger.async_log_success_event({}, response, None, None)
    assert response.choices[0].message.content == "[SECRET.token]"


@pytest.mark.asyncio
async def test_async_logger_skips_response_already_processed_by_wrapper():
    """Late LiteLLM worker callbacks must not re-run safety on a response
    already handled by FR finalization."""
    from opentelemetry.instrumentation.litellm import _invoke_acompletion

    exp, tracer = _make_tracer()
    calls = []

    def handler(context):
        calls.append(context.text)
        if context.text == "token-abc":
            return _completion_handler("[SECRET.token]", context)
        return None

    register_completion_safety_handler(handler)

    async def wrapped(*args, **kwargs):
        return SimpleNamespace(
            model="gpt-4o-mini",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="token-abc"),
                    text="token-abc",
                    finish_reason="stop",
                )
            ],
        )

    response = await _invoke_acompletion(
        tracer,
        wrapped,
        (),
        {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.choices[0].message.content == "[SECRET.token]"
    assert calls == ["token-abc"]

    # Simulate LiteLLM's async logging worker flushing after FR already ended
    # the span. The marker should make this a no-op.
    logger = _FortifyRootCompletionLogger()
    await logger.async_log_success_event({}, response, None, None)

    assert calls == ["token-abc"]
    spans = exp.get_finished_spans()
    assert len(spans) == 1
    assert len(spans[0].events) == 1


# ---------------------------------------------------------------------------
# Logger uses current OTel span for finding emission
# ---------------------------------------------------------------------------

def test_logger_emits_findings_on_current_span():
    """Findings from completion safety are emitted on whatever span is current
    when log_success_event fires (i.e., FR's safety span in production)."""
    exp, tracer = _make_tracer()
    register_completion_safety_handler(
        lambda context: _completion_handler("[MASKED]", context)
        if context.text == "secret"
        else None
    )
    logger = _FortifyRootCompletionLogger()
    span = tracer.start_span("test.span")
    token = context_api.attach(set_span_in_context(span))
    try:
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="secret"), text="secret")]
        )
        logger.log_success_event({}, response, None, None)
    finally:
        context_api.detach(token)
        span.end()

    finished = exp.get_finished_spans()
    assert len(finished) == 1
    assert len(finished[0].events) == 1  # one completion finding event


# ---------------------------------------------------------------------------
# Logger: failure event callbacks are no-ops (must not raise)
# ---------------------------------------------------------------------------

def test_logger_failure_events_are_noop():
    logger = _FortifyRootCompletionLogger()
    logger.log_failure_event({}, None, None, None)


@pytest.mark.asyncio
async def test_async_logger_failure_events_are_noop():
    logger = _FortifyRootCompletionLogger()
    await logger.async_log_failure_event({}, None, None, None)


# ---------------------------------------------------------------------------
# Logger: non-text content (image/audio/binary) is a no-op
# ---------------------------------------------------------------------------

def test_logger_non_text_content_is_noop():
    """Image/audio/binary content blocks must pass through unmodified."""
    register_completion_safety_handler(
        lambda context: _completion_handler("[MASKED]", context)
    )
    logger = _FortifyRootCompletionLogger()
    # Image URL block — type is not "text"/"output_text"/None
    image_block = {"type": "image_url", "url": "https://example.com/img.png"}
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=[image_block]))]
    )
    original_url = image_block["url"]
    logger.log_success_event({}, response, None, None)
    assert response.choices[0].message.content[0]["url"] == original_url


# ---------------------------------------------------------------------------
# Instrumentor registers / deregisters logger in litellm.callbacks
# ---------------------------------------------------------------------------

def test_instrumentor_registers_logger_at_position_zero():
    """_FortifyRootCompletionLogger must be at index 0, and (post-ST-10.1)
    _FortifyRootRetryEmitter must be at index 1, with any pre-existing
    customer callbacks pushed to index 2+ after instrument()."""
    import litellm
    from opentelemetry.instrumentation.litellm import _FortifyRootRetryEmitter
    original_callbacks = list(getattr(litellm, "callbacks", []))
    try:
        litellm.callbacks = ["existing_cb"]
        instrumentor = LiteLLMInstrumentor()
        with patch("opentelemetry.instrumentation.litellm.wrap_function_wrapper"), \
             patch("opentelemetry.instrumentation.litellm.unwrap"):
            instrumentor._instrument()
            assert isinstance(litellm.callbacks[0], _FortifyRootCompletionLogger)
            assert isinstance(litellm.callbacks[1], _FortifyRootRetryEmitter)
            assert litellm.callbacks[2] == "existing_cb"
            instrumentor._uninstrument()
            # After uninstrument, BOTH FR callbacks removed.
            assert not any(isinstance(cb, _FortifyRootCompletionLogger) for cb in litellm.callbacks)
            assert not any(isinstance(cb, _FortifyRootRetryEmitter) for cb in litellm.callbacks)
    finally:
        litellm.callbacks = original_callbacks


def test_instrumentor_handles_non_list_callbacks():
    """If litellm.callbacks is not a list, _instrument() converts it."""
    import litellm
    original_callbacks = getattr(litellm, "callbacks", [])
    try:
        litellm.callbacks = None
        instrumentor = LiteLLMInstrumentor()
        with patch("opentelemetry.instrumentation.litellm.wrap_function_wrapper"), \
             patch("opentelemetry.instrumentation.litellm.unwrap"):
            instrumentor._instrument()
            assert isinstance(litellm.callbacks, list)
            assert isinstance(litellm.callbacks[0], _FortifyRootCompletionLogger)
            instrumentor._uninstrument()
    finally:
        litellm.callbacks = original_callbacks


# ---------------------------------------------------------------------------
# span.role attribute is set on FR safety span
# ---------------------------------------------------------------------------

def test_fr_span_has_safety_wrapper_role():
    """FR's span must carry fortifyroot.span.role = 'safety_wrapper'."""
    from opentelemetry.instrumentation.litellm import _invoke_completion

    exp, tracer = _make_tracer()

    def wrapped(*args, **kwargs):
        return SimpleNamespace(
            model="gpt-4o",
            usage=None,
            choices=[SimpleNamespace(message=SimpleNamespace(content="reply"), text="reply", finish_reason="stop")],
        )

    _invoke_completion(tracer, wrapped, (), {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]})

    spans = exp.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "fortifyroot.litellm.safety"
    assert spans[0].attributes["fortifyroot.span.role"] == "safety_wrapper"
    assert spans[0].attributes["gen_ai.system"] == "litellm"


def test_native_otel_marker_requires_litellm_request_span_flag(monkeypatch):
    """Do not mark the safety span unless LiteLLM will emit litellm_request."""
    from opentelemetry.instrumentation.litellm import _invoke_completion
    import litellm
    import litellm.integrations.opentelemetry as native_otel

    class DummyOpenTelemetry:
        pass

    original_callbacks = list(getattr(litellm, "callbacks", []))
    monkeypatch.setattr(native_otel, "OpenTelemetry", DummyOpenTelemetry)
    monkeypatch.delenv("USE_OTEL_LITELLM_REQUEST_SPAN", raising=False)

    try:
        litellm.callbacks = [DummyOpenTelemetry()]
        exp, tracer = _make_tracer()

        def wrapped(*args, **kwargs):
            return SimpleNamespace(
                model="gpt-4o",
                usage=None,
                choices=[SimpleNamespace(message=SimpleNamespace(content="reply"), text="reply")],
            )

        _invoke_completion(
            tracer,
            wrapped,
            (),
            {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
        )

        span = exp.get_finished_spans()[0]
        assert "fortifyroot.span.has_native_otel_child" not in span.attributes
    finally:
        litellm.callbacks = original_callbacks


def test_native_otel_marker_set_when_litellm_request_span_enabled(monkeypatch):
    """Mark the safety span when LiteLLM is configured to emit litellm_request."""
    from opentelemetry.instrumentation.litellm import _invoke_completion
    import litellm
    import litellm.integrations.opentelemetry as native_otel

    class DummyOpenTelemetry:
        pass

    original_callbacks = list(getattr(litellm, "callbacks", []))
    monkeypatch.setattr(native_otel, "OpenTelemetry", DummyOpenTelemetry)
    monkeypatch.setenv("USE_OTEL_LITELLM_REQUEST_SPAN", "true")

    try:
        litellm.callbacks = [DummyOpenTelemetry()]
        exp, tracer = _make_tracer()

        def wrapped(*args, **kwargs):
            return SimpleNamespace(
                model="gpt-4o",
                usage=None,
                choices=[SimpleNamespace(message=SimpleNamespace(content="reply"), text="reply")],
            )

        _invoke_completion(
            tracer,
            wrapped,
            (),
            {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
        )

        span = exp.get_finished_spans()[0]
        assert span.attributes["fortifyroot.span.has_native_otel_child"] is True
    finally:
        litellm.callbacks = original_callbacks

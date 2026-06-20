from types import SimpleNamespace
from uuid import uuid4

from langchain_core.outputs import Generation, LLMResult
from opentelemetry import trace
from opentelemetry.instrumentation.langchain import callback_handler
from opentelemetry.instrumentation.langchain.callback_handler import (
    FR_STREAMING_TIME_TO_FIRST_TOKEN_MS,
    FR_STREAMING_TIME_TO_GENERATE_MS,
    TraceloopCallbackHandler,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)


class _StubHistogram:
    def __init__(self):
        self.records = []

    def record(self, *args, **kwargs):
        self.records.append((args, kwargs))


def _install_noop_fortifyroot(monkeypatch):
    monkeypatch.setitem(
        __import__("sys").modules,
        "opentelemetry.instrumentation.fortifyroot",
        SimpleNamespace(emit_deferred_findings=lambda span: None),
    )


def _handler_and_exporter():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    handler = TraceloopCallbackHandler(
        provider.get_tracer("test"),
        _StubHistogram(),
        _StubHistogram(),
    )
    return handler, exporter


def _llm_result():
    return LLMResult(
        generations=[[Generation(text="hello")]],
        llm_output={
            "model_name": "gpt-4o-mini",
            "token_usage": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
            },
        },
    )


def test_streaming_callbacks_set_ttft_and_sttg_span_attrs(monkeypatch):
    _install_noop_fortifyroot(monkeypatch)
    handler, exporter = _handler_and_exporter()
    run_id = uuid4()

    handler.on_llm_start(
        serialized={"id": ["langchain_openai", "llms", "base", "OpenAI"]},
        prompts=["hello"],
        run_id=run_id,
        invocation_params={"model": "gpt-4o-mini"},
    )
    handler.spans[run_id].start_time = 100.0

    times = [101.25, 103.0]

    def fake_time():
        if times:
            return times.pop(0)
        return 103.0

    monkeypatch.setattr(callback_handler.time, "perf_counter", fake_time)

    handler.on_llm_new_token("hello", run_id=run_id)
    handler.on_llm_end(_llm_result(), run_id=run_id)

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    attrs = spans[0].attributes
    assert attrs[FR_STREAMING_TIME_TO_FIRST_TOKEN_MS] == 1250
    assert attrs[FR_STREAMING_TIME_TO_GENERATE_MS] == 1750


def test_streaming_callbacks_clamp_negative_latency_attrs(monkeypatch):
    _install_noop_fortifyroot(monkeypatch)
    handler, exporter = _handler_and_exporter()
    run_id = uuid4()

    handler.on_llm_start(
        serialized={"id": ["langchain_openai", "llms", "base", "OpenAI"]},
        prompts=["hello"],
        run_id=run_id,
        invocation_params={"model": "gpt-4o-mini"},
    )
    handler.spans[run_id].start_time = 100.0

    times = [99.0, 98.0]

    def fake_time():
        if times:
            return times.pop(0)
        return 98.0

    monkeypatch.setattr(callback_handler.time, "perf_counter", fake_time)

    handler.on_llm_new_token("hello", run_id=run_id)
    handler.on_llm_end(_llm_result(), run_id=run_id)

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    attrs = spans[0].attributes
    assert attrs[FR_STREAMING_TIME_TO_FIRST_TOKEN_MS] == 0
    assert attrs[FR_STREAMING_TIME_TO_GENERATE_MS] == 0


def test_non_streaming_llm_end_does_not_set_streaming_latency_attrs(monkeypatch):
    _install_noop_fortifyroot(monkeypatch)
    handler, exporter = _handler_and_exporter()
    run_id = uuid4()

    handler.on_llm_start(
        serialized={"id": ["langchain_openai", "llms", "base", "OpenAI"]},
        prompts=["hello"],
        run_id=run_id,
        invocation_params={"model": "gpt-4o-mini"},
    )
    handler.spans[run_id].start_time = 100.0
    monkeypatch.setattr(callback_handler.time, "perf_counter", lambda: 103.0)

    handler.on_llm_end(_llm_result(), run_id=run_id)

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    attrs = spans[0].attributes
    assert FR_STREAMING_TIME_TO_FIRST_TOKEN_MS not in attrs
    assert FR_STREAMING_TIME_TO_GENERATE_MS not in attrs

# NOTE:
# This file has been added by FortifyRoot.
#
# Unit test for the FortifyRoot streaming-latency span attributes added to the
# OpenAI streaming instrumentation:
#   - fortifyroot.llm.streaming.time_to_first_token_ms  (TTFT)
#   - fortifyroot.llm.streaming.time_to_generate_ms     (STTG)
#
# These are the RDS-extractable span counterparts of the existing Mimir
# streaming histograms (which are left unchanged). It exercises the v0 streaming
# generator directly with fake chunks + stub histograms, so it needs NO OpenAI
# API key and NO VCR cassette and is fully deterministic.
#
# Contract: streaming latency contract
import time

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from opentelemetry.instrumentation.openai.shared import chat_wrappers as cw

TTFT = "fortifyroot.llm.streaming.time_to_first_token_ms"
STTG = "fortifyroot.llm.streaming.time_to_generate_ms"


class _StubHistogram:
    """Truthy histogram stub so the wrapper's record()/attr branches run."""

    def __init__(self):
        self.records = []

    def record(self, value, attributes=None):
        self.records.append(value)


class _StubStreamingSafety:
    def __init__(self, span, span_name):
        pass

    def process_chunk(self, item):
        return item


def _neutralize_helpers(monkeypatch):
    """Isolate the streaming loop + new span-attr logic from the OpenAI-typed
    accumulation/response helpers so the generator runs on fake chunks."""
    monkeypatch.setattr(cw, "_accumulate_stream_items", lambda *a, **k: None)
    monkeypatch.setattr(cw, "_set_streaming_token_metrics", lambda *a, **k: None)
    monkeypatch.setattr(cw, "_set_response_attributes", lambda *a, **k: None)
    monkeypatch.setattr(cw, "_set_completions", lambda *a, **k: None)
    monkeypatch.setattr(cw, "_get_openai_base_url", lambda *a, **k: "")
    monkeypatch.setattr(cw, "metric_shared_attributes", lambda *a, **k: {})
    monkeypatch.setattr(cw, "should_emit_events", lambda: False)
    monkeypatch.setattr(cw, "should_send_prompts", lambda: False)
    monkeypatch.setattr(cw, "OpenAIChatStreamingSafety", _StubStreamingSafety)


def _recording_span():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    span = provider.get_tracer("fr-streaming-latency-test").start_span("openai.chat")
    return span, exporter


def test_elapsed_ms_clamps_negative_duration():
    assert cw._elapsed_ms(10.0, 9.0) == 0


def test_streaming_sets_positive_integer_latency_attrs(monkeypatch):
    """A streaming response that yields tokens sets both attributes as
    non-negative integer milliseconds on the single LLM span."""
    _neutralize_helpers(monkeypatch)
    span, exporter = _recording_span()

    start = time.perf_counter() - 0.05  # 50ms ago => TTFT/STTG are clearly > 0
    ttft_hist, sttg_hist = _StubHistogram(), _StubHistogram()
    chunks = [object(), object(), object()]

    consumed = list(
        cw._build_from_streaming_response(
            span,
            iter(chunks),
            streaming_time_to_first_token=ttft_hist,
            streaming_time_to_generate=sttg_hist,
            start_time=start,
        )
    )

    assert len(consumed) == len(chunks)
    finished = exporter.get_finished_spans()
    assert len(finished) == 1
    attrs = finished[0].attributes

    assert isinstance(attrs.get(TTFT), int) and attrs[TTFT] >= 0
    assert isinstance(attrs.get(STTG), int) and attrs[STTG] >= 0
    # The existing Mimir histograms still record exactly once each (unchanged).
    assert len(ttft_hist.records) == 1
    assert len(sttg_hist.records) == 1


def test_streaming_latency_attrs_do_not_require_metrics_histograms(monkeypatch):
    """Tracing/RDS span attrs must not depend on the Mimir histogram objects."""
    _neutralize_helpers(monkeypatch)
    span, exporter = _recording_span()

    consumed = list(
        cw._build_from_streaming_response(
            span,
            iter([object(), object()]),
            streaming_time_to_first_token=None,
            streaming_time_to_generate=None,
            start_time=time.perf_counter() - 0.05,
        )
    )

    assert len(consumed) == 2
    finished = exporter.get_finished_spans()
    assert len(finished) == 1
    attrs = finished[0].attributes
    assert isinstance(attrs.get(TTFT), int) and attrs[TTFT] >= 0
    assert isinstance(attrs.get(STTG), int) and attrs[STTG] >= 0


def test_empty_stream_sets_no_latency_attrs(monkeypatch):
    """No chunk => no first token => neither attribute is set, so the backend
    leaves the RDS columns NULL (they are nullable by design)."""
    _neutralize_helpers(monkeypatch)
    span, exporter = _recording_span()

    consumed = list(
        cw._build_from_streaming_response(
            span,
            iter([]),
            streaming_time_to_first_token=_StubHistogram(),
            streaming_time_to_generate=_StubHistogram(),
            start_time=time.perf_counter(),
        )
    )

    assert consumed == []
    finished = exporter.get_finished_spans()
    assert len(finished) == 1
    attrs = finished[0].attributes
    assert TTFT not in attrs
    assert STTG not in attrs

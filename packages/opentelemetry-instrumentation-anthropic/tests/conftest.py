"""Unit tests configuration module."""

import os

import pytest
from anthropic import Anthropic, AsyncAnthropic
from opentelemetry.instrumentation.anthropic import AnthropicInstrumentor
from opentelemetry.instrumentation.anthropic.utils import TRACELOOP_TRACE_CONTENT
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import (
    InMemoryLogExporter,
    SimpleLogRecordProcessor,
)
from opentelemetry.sdk.metrics import Counter, Histogram, MeterProvider
from opentelemetry.sdk.metrics.export import (
    AggregationTemporality,
    InMemoryMetricReader,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

pytest_plugins = []


class _NoFortifyRootSpanExporter(InMemorySpanExporter):
    """ST-10.4 (2026-05-17): filter out any span whose name starts
    with ``fortifyroot.`` from the upstream-test span exporter.

    ST-10.4 added per-attempt ``fortifyroot.anthropic.retry_attempt``
    sibling spans under every anthropic logical call (once the
    Anthropic ``_wrap`` started using ``trace.use_span`` so the retry
    handler can find the parent span). Upstream / legacy Anthropic
    tests assert exact span-name lists (e.g.
    ``all(span.name == "anthropic.chat" for span in spans)``); without
    a filter, every such assertion would now fail because the
    retry_attempt sibling is also exported.

    Mirrors the OpenAI test conftest pattern (added 2026-05-16) and
    the LangChain CI-hardening pattern (2026-05-15). ST-10.4 unit
    tests in ``tests/test_retry_attempt_emission.py`` use their own
    ``fresh_tracer`` fixture (not this one), so they continue to see
    retry_attempt spans and aren't affected by the filter.
    """

    def get_finished_spans(self):  # type: ignore[override]
        # Filter by role rather than name prefix so legitimate
        # fortifyroot.*.safety / .llm_wrapper / etc. spans remain
        # visible to tests that inspect them.
        return tuple(
            s for s in super().get_finished_spans()
            if (s.attributes or {}).get("fortifyroot.span.role") != "retry_attempt"
        )


@pytest.fixture(scope="function", name="span_exporter")
def fixture_span_exporter():
    exporter = _NoFortifyRootSpanExporter()
    yield exporter


@pytest.fixture(scope="function", name="tracer_provider")
def fixture_tracer_provider(span_exporter):
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    return provider


@pytest.fixture(scope="function", name="log_exporter")
def fixture_log_exporter():
    exporter = InMemoryLogExporter()
    yield exporter


@pytest.fixture(scope="function", name="logger_provider")
def fixture_logger_provider(log_exporter):
    provider = LoggerProvider()
    provider.add_log_record_processor(SimpleLogRecordProcessor(log_exporter))
    return provider


@pytest.fixture(scope="function", name="reader")
def fixture_reader():
    reader = InMemoryMetricReader(
        {Counter: AggregationTemporality.DELTA, Histogram: AggregationTemporality.DELTA}
    )
    return reader


@pytest.fixture(scope="function", name="meter_provider")
def fixture_meter_provider(reader):
    resource = Resource.create()
    meter_provider = MeterProvider(metric_readers=[reader], resource=resource)

    return meter_provider


@pytest.fixture
def anthropic_client():
    return Anthropic()


@pytest.fixture
def async_anthropic_client():
    return AsyncAnthropic()


@pytest.fixture(scope="function")
def instrument_legacy(reader, tracer_provider, meter_provider):
    async def upload_base64_image(*args):
        return "/some/url"

    instrumentor = AnthropicInstrumentor(
        enrich_token_usage=True,
        upload_base64_image=upload_base64_image,
    )
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
    )

    yield instrumentor

    instrumentor.uninstrument()


@pytest.fixture(scope="function")
def instrument_with_content(
    reader, tracer_provider, logger_provider, meter_provider
):
    os.environ.update({TRACELOOP_TRACE_CONTENT: "True"})

    async def upload_base64_image(*args):
        return "/some/url"

    instrumentor = AnthropicInstrumentor(
        use_legacy_attributes=False,
        enrich_token_usage=True,
        upload_base64_image=upload_base64_image,
    )
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    )

    yield instrumentor

    os.environ.pop(TRACELOOP_TRACE_CONTENT, None)
    instrumentor.uninstrument()


@pytest.fixture(scope="function")
def instrument_with_no_content(
    reader, tracer_provider, logger_provider, meter_provider
):
    os.environ.update({TRACELOOP_TRACE_CONTENT: "False"})

    async def upload_base64_image(*args):
        return "/some/url"

    instrumentor = AnthropicInstrumentor(
        use_legacy_attributes=False,
        enrich_token_usage=True,
        upload_base64_image=upload_base64_image,
    )
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    )

    yield instrumentor

    os.environ.pop(TRACELOOP_TRACE_CONTENT, None)
    instrumentor.uninstrument()


@pytest.fixture(autouse=True)
def environment():
    if "ANTHROPIC_API_KEY" not in os.environ:
        os.environ["ANTHROPIC_API_KEY"] = "test_api_key"


@pytest.fixture(scope="module")
def vcr_config():
    return {"filter_headers": ["x-api-key"]}

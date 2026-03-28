"""
VCR conftest template for fr-openllmetry-py instrumentation packages.

Copy this file to your package's tests/ directory and customize the
provider-specific sections (marked with TODO). The VCR configuration,
OTel exporter setup, and environment fixtures follow the standard
pattern used across all FR fork packages.

Usage:
    cp scripts/templates/conftest_vcr.py packages/<your-package>/tests/conftest.py

Then customize:
    1. PROVIDER_ENV_VARS — environment variables your provider SDK needs
    2. PROVIDER_FILTER_HEADERS — headers to strip from cassettes
    3. Provider client fixtures — the SDK client your tests use
    4. Instrumentor fixtures — your package's OpenTelemetry instrumentor
"""

import os

import pytest
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

# ---------------------------------------------------------------------------
# TODO: Provider-specific configuration
# ---------------------------------------------------------------------------

# Environment variables required by the provider SDK.
# These are set to dummy values so cassette replay works without real keys.
PROVIDER_ENV_VARS = {
    # "OPENAI_API_KEY": "test_api_key",
    # "ANTHROPIC_API_KEY": "test_api_key",
}

# Headers to strip from recorded cassettes (prevents leaking secrets).
# Common values: "authorization", "x-api-key", "api-key"
PROVIDER_FILTER_HEADERS = [
    "authorization",
    "x-api-key",
    "api-key",
]

# Query parameters to strip from recorded cassettes.
PROVIDER_FILTER_QUERY_PARAMS = [
    "api_key",
]


# ---------------------------------------------------------------------------
# Environment — set dummy keys so provider SDKs don't fail on import
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def environment():
    for key, default in PROVIDER_ENV_VARS.items():
        if key not in os.environ:
            os.environ[key] = default


# ---------------------------------------------------------------------------
# OpenTelemetry exporters — capture spans, metrics, and logs in memory
# ---------------------------------------------------------------------------


@pytest.fixture(scope="function", name="span_exporter")
def fixture_span_exporter():
    exporter = InMemorySpanExporter()
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
    return MeterProvider(metric_readers=[reader], resource=resource)


# ---------------------------------------------------------------------------
# VCR cassette configuration
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def vcr_config():
    """Standard VCR config for cassette recording and replay.

    - filter_headers: strips auth headers so secrets are never persisted
    - filter_query_parameters: strips API keys from URLs
    - match_on: deterministic matching for replay stability
    - record_mode: controlled by --record-mode pytest flag (default: none in CI)
    """
    return {
        "filter_headers": PROVIDER_FILTER_HEADERS,
        "filter_query_parameters": PROVIDER_FILTER_QUERY_PARAMS,
        "match_on": ["method", "scheme", "host", "port", "path", "query"],
    }


@pytest.fixture(autouse=True)
def clear_exporter(span_exporter):
    """Clear captured spans before each test for isolation."""
    span_exporter.clear()


# ---------------------------------------------------------------------------
# TODO: Provider client fixtures
# ---------------------------------------------------------------------------

# @pytest.fixture
# def provider_client():
#     """Create provider SDK client for testing."""
#     from <provider> import Client
#     return Client()

# @pytest.fixture
# def async_provider_client():
#     from <provider> import AsyncClient
#     return AsyncClient()


# ---------------------------------------------------------------------------
# TODO: Instrumentor fixtures
# ---------------------------------------------------------------------------

# @pytest.fixture(scope="function")
# def instrument_with_content(reader, tracer_provider, logger_provider, meter_provider):
#     """Instrument provider with content tracing enabled."""
#     from opentelemetry.instrumentation.<provider> import <Provider>Instrumentor
#
#     instrumentor = <Provider>Instrumentor()
#     instrumentor.instrument(
#         tracer_provider=tracer_provider,
#         logger_provider=logger_provider,
#         meter_provider=meter_provider,
#     )
#     yield instrumentor
#     instrumentor.uninstrument()

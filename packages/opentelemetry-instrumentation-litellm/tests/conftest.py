import os

import pytest
from opentelemetry.instrumentation.litellm import LiteLLMInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)


@pytest.fixture(scope="session")
def span_exporter():
    return InMemorySpanExporter()


@pytest.fixture(scope="session")
def tracer_provider(span_exporter):
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    return provider


@pytest.fixture(scope="session")
def instrument(tracer_provider):
    instrumentor = LiteLLMInstrumentor()
    instrumentor.instrument(tracer_provider=tracer_provider)
    yield instrumentor
    instrumentor.uninstrument()


@pytest.fixture(autouse=True)
def clear_exporter(span_exporter):
    span_exporter.clear()


@pytest.fixture(autouse=True)
def environment():
    env_defaults = {
        "OPENAI_API_KEY": "test_api_key",
    }
    for key, default in env_defaults.items():
        if key not in os.environ:
            os.environ[key] = default


@pytest.fixture(scope="module")
def vcr_config():
    return {
        "filter_headers": ["authorization", "api-key", "x-api-key"],
        "filter_query_parameters": ["api_key"],
        "match_on": ["method", "scheme", "host", "port", "path", "query"],
    }


# Backward compat: existing mock-based tests use test_tracer fixture
@pytest.fixture
def test_tracer():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer(__name__)
    return exporter, tracer

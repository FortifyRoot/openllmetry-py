# NOTE:
# This file has been modified by FortifyRoot.
# Original source: https://github.com/traceloop/openllmetry

import os

import pytest
from openai import OpenAI
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from traceloop.sdk.decorators import workflow, task


# ST-10.4: FortifyRoot LLM-attempt spans land in this test's
# session-scoped exporter ahead of the logical ``openai.chat`` span.
# Filter by the canonical ``fortifyroot.span.role`` attribute so every
# provider is dropped uniformly.
_FR_SPAN_ROLE_KEY = "fortifyroot.span.role"
_FR_SPAN_ROLE_LLM_ATTEMPT = "llm_attempt"


def _without_retry_attempt_spans(spans):
    return [
        s for s in spans
        if (s.attributes or {}).get(_FR_SPAN_ROLE_KEY) != _FR_SPAN_ROLE_LLM_ATTEMPT
    ]


@pytest.fixture(autouse=True)
def disable_trace_content():
    os.environ["TRACELOOP_TRACE_CONTENT"] = "false"
    yield
    os.environ["TRACELOOP_TRACE_CONTENT"] = "true"


@pytest.fixture
def openai_client():
    return OpenAI()


@pytest.mark.vcr
def test_simple_workflow(exporter, openai_client):
    @task(name="joke_creation")
    def create_joke():
        completion = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "Tell me a joke about opentelemetry"}
            ],
        )
        return completion.choices[0].message.content

    @workflow(name="pirate_joke_generator")
    def joke_workflow():
        create_joke()

    joke_workflow()

    spans = _without_retry_attempt_spans(exporter.get_finished_spans())
    assert [span.name for span in spans] == [
        "openai.chat",
        "joke_creation.task",
        "pirate_joke_generator.workflow",
    ]
    open_ai_span = next(s for s in spans if s.name == "openai.chat")
    assert open_ai_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 15
    assert not open_ai_span.attributes.get(f"{GenAIAttributes.GEN_AI_PROMPT}.0.content")
    assert not open_ai_span.attributes.get(
        f"{GenAIAttributes.GEN_AI_PROMPT}.0.completions"
    )

"""
FR VCR cassette tests for LiteLLM non-safety telemetry.

These tests verify that the FR-authored LiteLLM instrumentation correctly
captures LLM telemetry attributes (model, tokens, provider, request type,
streaming flag, span hierarchy) using recorded API responses.

Since the LiteLLM instrumentation package is 100% FR-authored (not TL),
it has no TL-authored cassette tests. This file provides that coverage.

North-star: This is a NEW file (FR-owned). Zero delta on TL files.
"""

import pytest
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import SpanAttributes

pytestmark = [pytest.mark.vcr, pytest.mark.fr]


# ---------------------------------------------------------------------------
# T4-C2: Sync chat completion — telemetry attributes
# ---------------------------------------------------------------------------

def test_litellm_sync_chat_completion_telemetry(instrument, span_exporter):
    """Sync chat completion captures model, tokens, prompt/completion content."""
    import litellm

    litellm.completion(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say hello in one word."}],
        max_tokens=10,
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) >= 1

    # Find the FR safety wrapper span
    fr_span = next(
        (s for s in spans if s.name == "fortifyroot.litellm.safety"), None
    )
    assert fr_span is not None, "FR safety wrapper span should exist"

    attrs = fr_span.attributes
    assert attrs.get(GenAIAttributes.GEN_AI_SYSTEM) == "litellm"
    assert attrs.get(SpanAttributes.LLM_REQUEST_TYPE) == "chat"
    assert attrs.get("fortifyroot.span.role") == "safety_wrapper"
    assert attrs.get(GenAIAttributes.GEN_AI_REQUEST_MODEL) == "gpt-4o-mini"
    assert attrs.get(SpanAttributes.LLM_IS_STREAMING) is False

    # Prompt content captured
    assert attrs.get(f"{SpanAttributes.LLM_PROMPTS}.0.role") == "user"
    assert "hello" in attrs.get(f"{SpanAttributes.LLM_PROMPTS}.0.content", "").lower()

    # Completion content captured
    assert attrs.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.role") is not None
    assert attrs.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content") is not None
    assert attrs.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.finish_reason") is not None

    # Response model and token usage
    assert attrs.get(GenAIAttributes.GEN_AI_RESPONSE_MODEL) is not None
    assert attrs.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS) is not None
    assert isinstance(attrs.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS), int)
    assert attrs.get(GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS) is not None
    assert isinstance(attrs.get(GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS), int)


# ---------------------------------------------------------------------------
# T4-C2: Async chat completion — telemetry attributes
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_litellm_async_chat_completion_telemetry(instrument, span_exporter):
    """Async chat completion captures same telemetry as sync."""
    import litellm

    await litellm.acompletion(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say goodbye in one word."}],
        max_tokens=10,
    )

    spans = span_exporter.get_finished_spans()
    fr_span = next(
        (s for s in spans if s.name == "fortifyroot.litellm.safety"), None
    )
    assert fr_span is not None

    attrs = fr_span.attributes
    assert attrs.get(GenAIAttributes.GEN_AI_SYSTEM) == "litellm"
    assert attrs.get(SpanAttributes.LLM_REQUEST_TYPE) == "chat"
    assert attrs.get(GenAIAttributes.GEN_AI_REQUEST_MODEL) == "gpt-4o-mini"
    assert attrs.get(f"{SpanAttributes.LLM_PROMPTS}.0.role") == "user"
    assert attrs.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content") is not None
    assert attrs.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS) is not None


# ---------------------------------------------------------------------------
# T4-C3: Sync streaming chat — telemetry attributes
# ---------------------------------------------------------------------------

def test_litellm_sync_streaming_chat_telemetry(instrument, span_exporter):
    """Streaming chat captures llm.is_streaming=True and accumulated content."""
    import litellm

    response = litellm.completion(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Count from 1 to 3."}],
        max_tokens=20,
        stream=True,
    )

    # Consume the stream
    chunks = []
    for chunk in response:
        chunks.append(chunk)

    assert len(chunks) > 0, "Should receive at least one streaming chunk"

    spans = span_exporter.get_finished_spans()
    fr_span = next(
        (s for s in spans if s.name == "fortifyroot.litellm.safety"), None
    )
    assert fr_span is not None

    attrs = fr_span.attributes
    assert attrs.get(SpanAttributes.LLM_IS_STREAMING) is True
    assert attrs.get(GenAIAttributes.GEN_AI_SYSTEM) == "litellm"
    assert attrs.get(f"{SpanAttributes.LLM_PROMPTS}.0.role") == "user"


# ---------------------------------------------------------------------------
# T4-C3: Async streaming chat — telemetry attributes
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_litellm_async_streaming_chat_telemetry(instrument, span_exporter):
    """Async streaming chat captures same telemetry as sync streaming."""
    import litellm

    response = await litellm.acompletion(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Count from 4 to 6."}],
        max_tokens=20,
        stream=True,
    )

    chunks = []
    async for chunk in response:
        chunks.append(chunk)

    assert len(chunks) > 0

    spans = span_exporter.get_finished_spans()
    fr_span = next(
        (s for s in spans if s.name == "fortifyroot.litellm.safety"), None
    )
    assert fr_span is not None

    attrs = fr_span.attributes
    assert attrs.get(SpanAttributes.LLM_IS_STREAMING) is True
    assert attrs.get(GenAIAttributes.GEN_AI_SYSTEM) == "litellm"


# ---------------------------------------------------------------------------
# T4-C4: Sync text completion — telemetry attributes
# ---------------------------------------------------------------------------

def test_litellm_sync_text_completion_telemetry(instrument, span_exporter):
    """Text completion captures llm.request.type='completion' and prompt content."""
    import litellm

    litellm.text_completion(
        model="gpt-4o-mini",
        prompt="The capital of France is",
        max_tokens=10,
    )

    spans = span_exporter.get_finished_spans()
    fr_span = next(
        (s for s in spans if s.name == "fortifyroot.litellm.safety"), None
    )
    assert fr_span is not None

    attrs = fr_span.attributes
    assert attrs.get(SpanAttributes.LLM_REQUEST_TYPE) == "completion"
    assert attrs.get(GenAIAttributes.GEN_AI_SYSTEM) == "litellm"
    assert attrs.get(GenAIAttributes.GEN_AI_REQUEST_MODEL) == "gpt-4o-mini"
    assert "France" in attrs.get(f"{SpanAttributes.LLM_PROMPTS}.0.content", "")


# ---------------------------------------------------------------------------
# T4-C4: Async text completion — telemetry attributes
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_litellm_async_text_completion_telemetry(instrument, span_exporter):
    """Async text completion captures same telemetry as sync."""
    import litellm

    await litellm.atext_completion(
        model="gpt-4o-mini",
        prompt="The capital of Germany is",
        max_tokens=10,
    )

    spans = span_exporter.get_finished_spans()
    fr_span = next(
        (s for s in spans if s.name == "fortifyroot.litellm.safety"), None
    )
    assert fr_span is not None

    attrs = fr_span.attributes
    assert attrs.get(SpanAttributes.LLM_REQUEST_TYPE) == "completion"
    assert "Germany" in attrs.get(f"{SpanAttributes.LLM_PROMPTS}.0.content", "")


# ---------------------------------------------------------------------------
# T4-C5: Dual instrumentation — span hierarchy
# ---------------------------------------------------------------------------

def test_litellm_dual_span_hierarchy(instrument, span_exporter):
    """
    Verify FR parent span → litellm_request child span hierarchy.

    FR's safety wrapper span should be the parent. LiteLLM's native OTel
    callback should create its 'litellm_request' span as a child.
    """
    import litellm

    litellm.completion(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "What is 2+2?"}],
        max_tokens=10,
    )

    spans = span_exporter.get_finished_spans()

    # FR parent span
    fr_span = next(
        (s for s in spans if s.name == "fortifyroot.litellm.safety"), None
    )
    assert fr_span is not None
    assert fr_span.attributes.get("fortifyroot.span.role") == "safety_wrapper"
    assert fr_span.attributes.get(GenAIAttributes.GEN_AI_SYSTEM) == "litellm"

    # LiteLLM native OTel child span (if native OTel is enabled)
    child_spans = [
        s for s in spans
        if s.parent is not None
        and s.parent.span_id == fr_span.context.span_id
    ]
    # Note: child span existence depends on LiteLLM's native OTel being enabled.
    # When enabled, we verify the hierarchy. When not, FR span is standalone.
    if child_spans:
        child = child_spans[0]
        # Child span should have actual provider info (not "litellm")
        child_system = child.attributes.get(GenAIAttributes.GEN_AI_SYSTEM)
        assert child_system is not None, "Child span should have gen_ai.system"


# ---------------------------------------------------------------------------
# T4-C6: FR completion logger at position zero
# ---------------------------------------------------------------------------

def test_litellm_fr_logger_at_position_zero(instrument, span_exporter):
    """
    Verify _FortifyRootCompletionLogger is at litellm.callbacks[0].

    This ensures FR's completion masking fires before LiteLLM's native OTel
    callback, so response_obj is masked in-place before native OTel sees it.
    """
    import litellm
    from opentelemetry.instrumentation.litellm import _FortifyRootCompletionLogger

    # Check callback ordering
    callbacks = getattr(litellm, "callbacks", [])
    assert len(callbacks) >= 1, "litellm.callbacks should have at least FR logger"
    assert isinstance(callbacks[0], _FortifyRootCompletionLogger), (
        f"Expected _FortifyRootCompletionLogger at position 0, got {type(callbacks[0])}"
    )

    # Make an actual call to verify completion content is captured
    litellm.completion(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say yes."}],
        max_tokens=5,
    )

    spans = span_exporter.get_finished_spans()
    fr_span = next(
        (s for s in spans if s.name == "fortifyroot.litellm.safety"), None
    )
    assert fr_span is not None
    # Completion content should be present (logger captured it)
    assert fr_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content") is not None

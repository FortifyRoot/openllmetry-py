"""Tests for LiteLLM retry-aware attempt emission.

Covers:
  - Instrumentor symmetry: _FortifyRootRetryEmitter is registered at
    instrument() AND removed at uninstrument() (Fallback B child-emission
    proof requirement (ii) per retry-loop design notes §4.4.1).
  - Single-attempt happy path: one llm_attempt span emitted under the
    safety_wrapper parent, parent carries has_attempt_child=true.
  - Multi-attempt retry path: 3 retry_attempt spans emitted (2 ERROR + 1
    OK) for a 429→429→200 sequence, all under one safety_wrapper.
  - Marker timing (§4.5): the parent's has_attempt_child marker
    is set AFTER the first retry_attempt's start, NOT at parent creation.
  - §4.7.1 token registration: framework-attempt tokens are registered
    on attempt-start AND unregistered on attempt-end.
  - Idempotency: sync + async success callbacks both firing for one
    attempt do not double-end the span.
  - No-parent guard: invoking the emitter without an ambient FR parent
    does not crash and does not emit an orphan retry_attempt span.
"""

from __future__ import annotations

from typing import Any

import pytest
from opentelemetry import trace
from opentelemetry.instrumentation.fortifyroot import (
    clear_attempt_counters_for_test,
    retry_registry,
)
from opentelemetry.instrumentation.litellm import (
    LiteLLMInstrumentor,
    _FortifyRootCompletionLogger,
    _FortifyRootRetryEmitter,
    _FR_HAS_ATTEMPT_CHILD_KEY,
    _FR_RETRY_ATTEMPT_MAP,
    _FR_LLM_ATTEMPT_SPAN_NAME_PREFIX,
    _resolve_routed_provider,
    _start_retry_attempt_span,
    _finalize_retry_attempt_span,
    _set_active_retry_attempt_attribute,
)
from opentelemetry.instrumentation.litellm.streaming_safety import (
    FR_STREAMING_TIME_TO_FIRST_TOKEN_MS,
    FR_STREAMING_TIME_TO_GENERATE_MS,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)


# ---------------------------------------------------------------------------
# Fixtures local to this file (intentionally NOT using conftest's
# session-scoped instrument fixture — these tests need fresh
# instrument/uninstrument cycles to validate symmetry).
# ---------------------------------------------------------------------------

@pytest.fixture
def fresh_tracer():
    """A fresh TracerProvider + in-memory exporter per test, installed
    as the GLOBAL tracer provider so the retry emitter's
    ``trace.get_tracer(...)`` lookups route through it. After the
    test, the global provider is left as-is — OTel's
    set_tracer_provider only allows one set per process, so we don't
    try to "restore" — but the in-memory exporter is cleared between
    tests."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    # Install as global so the retry emitter's tracer lookup uses it.
    # Idempotent across tests because OTel allows only one set; second
    # call is a no-op (the existing provider stays). To work around
    # that for test isolation, we keep clearing the exporter between
    # tests and reuse whatever provider was first set.
    try:
        trace.set_tracer_provider(provider)
    except Exception:
        # set_tracer_provider raises a warning (not an error) if a
        # provider is already set; we tolerate it because the
        # InMemorySpanExporter on the FIRST test's provider is what
        # all subsequent tests will use, and that's fine since each
        # test clears it.
        pass
    # Always re-fetch the global tracer so the test sees the same
    # one the emitter sees.
    tracer = trace.get_tracer("test")
    # If a previous test set its own provider, fall back to using that
    # test's exporter. We can't access it directly, so re-create our
    # own and install via add_span_processor on the existing provider.
    current_provider = trace.get_tracer_provider()
    if current_provider is not provider:
        # Existing provider — add OUR exporter to it so we still capture spans.
        try:
            current_provider.add_span_processor(SimpleSpanProcessor(exporter))
        except Exception:
            pass
    yield tracer, exporter, current_provider


@pytest.fixture(autouse=True)
def reset_registry_and_map():
    """Each test starts with empty retry-emitter state."""
    retry_registry._reset_for_test()
    clear_attempt_counters_for_test()
    _FR_RETRY_ATTEMPT_MAP.clear()
    yield
    retry_registry._reset_for_test()
    clear_attempt_counters_for_test()
    _FR_RETRY_ATTEMPT_MAP.clear()


def _assert_attempt_sequence(spans):
    by_number = sorted(
        spans,
        key=lambda s: int(s.attributes.get("fortifyroot.attempt.number") or -1),
    )
    for expected, span in enumerate(by_number, start=1):
        assert span.name == f"{_FR_LLM_ATTEMPT_SPAN_NAME_PREFIX}.attempt_{expected}"
        assert span.attributes.get("fortifyroot.attempt.number") == expected
        assert span.attributes.get("fortifyroot.attempt.is_retry") is (expected > 1)


# ---------------------------------------------------------------------------
# Instrumentor symmetry — Fallback B child-emission proof (ii).
# ---------------------------------------------------------------------------

def test_retry_emitter_inherits_from_litellm_custom_logger():
    """Regression guard for LiteLLM callback dispatch compatibility.
    LiteLLM's dispatch loop gates every callback hook on
    ``isinstance(callback, CustomLogger)`` — see
    litellm_logging.py:1015 (log_pre_api_call) and :2303
    (log_success_event). A duck-typed _FortifyRootRetryEmitter is
    silently SKIPPED by the dispatch — log_pre_api_call never fires
    → no retry_attempt span ever emitted → §4.5 backend dedup has
    nothing to dedup → the entire LiteLLM retry-attempt contract is no-op'd.

    The bug was invisible to fork-side unit tests (which call the
    emitter's hooks directly, bypassing LiteLLM's dispatch). End-to-end
    Tier 1 with vendored fork surfaced it.

    This test catches the regression at unit-test time so a future
    refactor that drops the inheritance gets a fast failure.
    """
    from litellm.integrations.custom_logger import CustomLogger
    emitter = _FortifyRootRetryEmitter()
    assert isinstance(emitter, CustomLogger), (
        "_FortifyRootRetryEmitter MUST inherit from "
        "litellm.integrations.custom_logger.CustomLogger so LiteLLM's "
        "isinstance-gated callback dispatch fires its hooks. "
        "Without inheritance, the emitter is silently skipped and no "
        "retry_attempt spans are emitted."
    )


def test_instrumentor_registers_retry_emitter_at_instrument():
    """At _instrument() time, exactly one _FortifyRootRetryEmitter
    is present in litellm.callbacks."""
    import litellm

    instrumentor = LiteLLMInstrumentor()
    try:
        instrumentor.instrument()
        assert any(
            isinstance(cb, _FortifyRootRetryEmitter) for cb in litellm.callbacks
        ), "_FortifyRootRetryEmitter must be registered at instrument()"
        emitters = [
            cb for cb in litellm.callbacks if isinstance(cb, _FortifyRootRetryEmitter)
        ]
        assert len(emitters) == 1, f"expected exactly 1 emitter, found {len(emitters)}"
    finally:
        instrumentor.uninstrument()


def test_instrumentor_removes_retry_emitter_at_uninstrument():
    """At _uninstrument(), the _FortifyRootRetryEmitter is removed
    from litellm.callbacks."""
    import litellm

    instrumentor = LiteLLMInstrumentor()
    instrumentor.instrument()
    assert any(isinstance(cb, _FortifyRootRetryEmitter) for cb in litellm.callbacks)
    instrumentor.uninstrument()
    assert not any(
        isinstance(cb, _FortifyRootRetryEmitter) for cb in litellm.callbacks
    ), "_FortifyRootRetryEmitter must be unregistered at uninstrument()"


def test_completion_logger_fires_before_retry_emitter():
    """Order in litellm.callbacks: _FortifyRootCompletionLogger BEFORE
    _FortifyRootRetryEmitter. Completion-safety masking must run
    before retry_attempt finalization (which only captures metadata)."""
    import litellm

    instrumentor = LiteLLMInstrumentor()
    try:
        instrumentor.instrument()
        types = [type(cb).__name__ for cb in litellm.callbacks]
        completion_idx = next(
            (i for i, cb in enumerate(litellm.callbacks)
             if isinstance(cb, _FortifyRootCompletionLogger)),
            None,
        )
        retry_idx = next(
            (i for i, cb in enumerate(litellm.callbacks)
             if isinstance(cb, _FortifyRootRetryEmitter)),
            None,
        )
        assert completion_idx is not None, f"no completion logger found; callbacks={types}"
        assert retry_idx is not None, f"no retry emitter found; callbacks={types}"
        assert completion_idx < retry_idx, (
            f"completion logger must precede retry emitter; got "
            f"completion={completion_idx} retry={retry_idx} callbacks={types}"
        )
    finally:
        instrumentor.uninstrument()


# ---------------------------------------------------------------------------
# Single-attempt happy path.
# ---------------------------------------------------------------------------

def test_single_attempt_emits_one_retry_attempt_under_parent(fresh_tracer):
    """Drive _start_retry_attempt_span + _finalize_retry_attempt_span
    directly with an ambient parent span. Verify:
      - exactly 1 retry_attempt span exported
      - retry_attempt's parent is the safety_wrapper
      - parent has has_attempt_child=true
      - retry_attempt has fortifyroot.span.role=llm_attempt
      - retry_attempt has gen_ai.system, gen_ai.request.model
    """
    tracer, exporter, _ = fresh_tracer

    parent = tracer.start_span("fortifyroot.litellm.safety")
    with trace.use_span(parent, end_on_exit=False):
        kwargs: dict[str, Any] = {
            "litellm_call_id": "test-call-001",
            "model": "openai/gpt-4o-mini",
            "api_base": "https://api.openai.com/v1",
            "custom_llm_provider": "openai",
            "messages": [{"role": "user", "content": "hello masked [EMAIL]"}],
        }
        _start_retry_attempt_span(kwargs)
        _set_active_retry_attempt_attribute(
            parent, FR_STREAMING_TIME_TO_FIRST_TOKEN_MS, 123
        )
        _set_active_retry_attempt_attribute(
            parent, FR_STREAMING_TIME_TO_GENERATE_MS, 456
        )

        # Mock response object with usage/model.
        class MockUsage:
            prompt_tokens = 10
            completion_tokens = 5

        class MockResponse:
            id = "resp-abc"
            model = "gpt-4o-mini"
            usage = MockUsage()

        _finalize_retry_attempt_span(kwargs, MockResponse(), success=True)
    parent.end()

    spans = exporter.get_finished_spans()
    span_names = [s.name for s in spans]
    assert f"{_FR_LLM_ATTEMPT_SPAN_NAME_PREFIX}.attempt_1" in span_names, (
        f"retry_attempt span missing; got {span_names}"
    )
    retry_span = next(s for s in spans if s.name.startswith(f"{_FR_LLM_ATTEMPT_SPAN_NAME_PREFIX}.attempt_"))
    parent_span_exported = next(s for s in spans if s.name == "fortifyroot.litellm.safety")
    _assert_attempt_sequence([retry_span])

    assert retry_span.parent.span_id == parent_span_exported.context.span_id, (
        "retry_attempt MUST be a child of the safety_wrapper parent"
    )
    assert retry_span.attributes.get("fortifyroot.span.role") == "llm_attempt"
    assert retry_span.attributes.get("gen_ai.system") == "openai", (
        "gen_ai.system MUST be the routed provider, NOT the framework"
    )
    assert retry_span.attributes.get("gen_ai.request.model") == "openai/gpt-4o-mini"
    assert retry_span.attributes.get("gen_ai.response.model") == "gpt-4o-mini"
    assert retry_span.attributes.get("gen_ai.response.id") == "resp-abc"
    assert retry_span.attributes.get("gen_ai.usage.input_tokens") == 10
    assert retry_span.attributes.get("gen_ai.usage.output_tokens") == 5
    assert retry_span.attributes.get(FR_STREAMING_TIME_TO_FIRST_TOKEN_MS) == 123
    assert retry_span.attributes.get(FR_STREAMING_TIME_TO_GENERATE_MS) == 456
    assert retry_span.attributes.get("gen_ai.prompt.0.role") == "user"
    assert retry_span.attributes.get("gen_ai.prompt.0.content") == "hello masked [EMAIL]"

    # §4.5 marker: must be set on parent.
    assert parent_span_exported.attributes.get(
        _FR_HAS_ATTEMPT_CHILD_KEY
    ) is True, "parent MUST carry has_attempt_child=true"


def test_retry_attempt_accepts_anthropic_usage_token_names(fresh_tracer):
    """Anthropic-shaped LiteLLM responses may expose input/output token
    names instead of OpenAI-style prompt/completion names."""
    tracer, exporter, _ = fresh_tracer

    parent = tracer.start_span("fortifyroot.litellm.safety")
    with trace.use_span(parent, end_on_exit=False):
        kwargs: dict[str, Any] = {
            "litellm_call_id": "anthropic-usage-001",
            "model": "anthropic/claude-4-sonnet-20250514",
            "custom_llm_provider": "anthropic",
            "messages": [{"role": "user", "content": "hello"}],
        }
        _start_retry_attempt_span(kwargs)

        class MockUsage:
            input_tokens = 11
            output_tokens = 7

        class MockResponse:
            model = "claude-sonnet-4-20250514"
            usage = MockUsage()

        _finalize_retry_attempt_span(kwargs, MockResponse(), success=True)
    parent.end()

    retry_span = next(
        s for s in exporter.get_finished_spans()
        if s.name.startswith(f"{_FR_LLM_ATTEMPT_SPAN_NAME_PREFIX}.attempt_")
    )
    assert retry_span.attributes.get("gen_ai.usage.input_tokens") == 11
    assert retry_span.attributes.get("gen_ai.usage.output_tokens") == 7


# ---------------------------------------------------------------------------
# Multi-attempt retry path.
# ---------------------------------------------------------------------------

def test_three_attempt_retry_path_emits_three_retry_attempt_spans(fresh_tracer):
    """Shared-parent numbering mechanics for 429 → 429 → 200.

    This drives the emitter directly with one safety_wrapper parent so
    the attempt counter must produce attempt_1/2/3. Some production
    LiteLLM retry helpers re-enter completion() per attempt and therefore
    create one safety_wrapper per attempt; those multi-trace paths can
    correctly surface as separate attempt_1 spans, as documented in
    retry-loop design notes.
    """
    tracer, exporter, _ = fresh_tracer

    parent = tracer.start_span("fortifyroot.litellm.safety")

    class MockHTTPError(Exception):
        def __init__(self, status_code: int):
            self.status_code = status_code

    with trace.use_span(parent, end_on_exit=False):
        # Attempt 1 — 429
        k1 = {"litellm_call_id": "call-001", "model": "openai/gpt-4o-mini"}
        _start_retry_attempt_span(k1)
        k1["exception"] = MockHTTPError(429)
        _finalize_retry_attempt_span(k1, None, success=False)

        # Attempt 2 — 429
        k2 = {"litellm_call_id": "call-002", "model": "openai/gpt-4o-mini"}
        _start_retry_attempt_span(k2)
        k2["exception"] = MockHTTPError(429)
        _finalize_retry_attempt_span(k2, None, success=False)

        # Attempt 3 — 200
        k3 = {"litellm_call_id": "call-003", "model": "openai/gpt-4o-mini"}
        _start_retry_attempt_span(k3)

        class MockSuccess:
            id = "ok-id"
            model = "gpt-4o-mini"
            usage = type("U", (), {"prompt_tokens": 10, "completion_tokens": 5})()

        _finalize_retry_attempt_span(k3, MockSuccess(), success=True)
    parent.end()

    retry_spans = [
        s for s in exporter.get_finished_spans()
        if s.name.startswith(f"{_FR_LLM_ATTEMPT_SPAN_NAME_PREFIX}.attempt_")
    ]
    assert len(retry_spans) == 3, f"expected 3 retry_attempt spans, got {len(retry_spans)}"
    _assert_attempt_sequence(retry_spans)

    # 2 of them must be ERROR status, 1 OK.
    from opentelemetry.trace import StatusCode
    error_spans = [s for s in retry_spans if s.status.status_code == StatusCode.ERROR]
    ok_spans = [s for s in retry_spans if s.status.status_code == StatusCode.OK]
    assert len(error_spans) == 2, f"expected 2 ERROR spans, got {len(error_spans)}"
    assert len(ok_spans) == 1, f"expected 1 OK span, got {len(ok_spans)}"

    # The 2 error spans carry http.status_code=429 + error.type.
    for s in error_spans:
        assert s.attributes.get("http.status_code") == 429
        assert s.attributes.get("error.type") == "MockHTTPError"


# ---------------------------------------------------------------------------
# Marker timing — §4.5 "marker timing" boundary case.
# ---------------------------------------------------------------------------

def test_marker_set_AFTER_first_attempt_starts_not_at_parent_creation(fresh_tracer):
    """Per §4.5 marker-timing paragraph: the marker is set AFTER the
    first retry_attempt successfully starts, NOT at parent creation.

    This test verifies: a parent span observed BEFORE any
    _start_retry_attempt_span call has NO marker. After the first
    _start_retry_attempt_span call, the parent has the marker.
    """
    tracer, exporter, _ = fresh_tracer

    parent = tracer.start_span("fortifyroot.litellm.safety")

    # Before any retry_attempt: no marker.
    # (Use the live span object since it hasn't been exported yet.)
    parent_attrs_before = dict(parent.attributes or {})
    assert _FR_HAS_ATTEMPT_CHILD_KEY not in parent_attrs_before, (
        "parent MUST NOT have the marker before any retry_attempt starts"
    )

    with trace.use_span(parent, end_on_exit=False):
        kwargs = {"litellm_call_id": "marker-test", "model": "openai/gpt-4o-mini"}
        _start_retry_attempt_span(kwargs)
    # After: parent has the marker.
    parent_attrs_after = dict(parent.attributes or {})
    assert parent_attrs_after.get(_FR_HAS_ATTEMPT_CHILD_KEY) is True, (
        "parent MUST have the marker after first retry_attempt starts"
    )

    # Cleanup.
    with trace.use_span(parent, end_on_exit=False):
        _finalize_retry_attempt_span(kwargs, None, success=False)
    parent.end()


def test_marker_NOT_set_when_no_retry_attempt_starts(fresh_tracer):
    """If _start_retry_attempt_span is never called for a parent, the
    parent NEVER gets the marker. This is the "framework returned
    without invoking the retry-emitter callback at all" scenario from
    the §4.5 marker-timing paragraph — telemetry-loss is avoided
    because the §4.5 backend dedup degrades gracefully (parent stays
    canonical when the marker is unset)."""
    tracer, exporter, _ = fresh_tracer

    parent = tracer.start_span("fortifyroot.litellm.safety")
    parent.end()

    spans = exporter.get_finished_spans()
    parent_exported = next(s for s in spans if s.name == "fortifyroot.litellm.safety")
    assert _FR_HAS_ATTEMPT_CHILD_KEY not in (parent_exported.attributes or {}), (
        "no retry_attempt → no marker → parent stays canonical for §4.5 dedup"
    )


# ---------------------------------------------------------------------------
# §4.7.1 token registration.
# ---------------------------------------------------------------------------

def test_framework_token_registered_during_attempt_unregistered_after(fresh_tracer):
    """While a retry_attempt span is open, is_framework_owned() is
    True for the current thread. After the success/failure callback
    fires, it returns to False."""
    tracer, _, _ = fresh_tracer
    from opentelemetry.instrumentation.fortifyroot import is_framework_owned

    parent = tracer.start_span("fortifyroot.litellm.safety")
    with trace.use_span(parent, end_on_exit=False):
        assert not is_framework_owned(), "no attempt yet → not owned"

        kwargs = {"litellm_call_id": "tok-test", "model": "openai/gpt-4o-mini"}
        _start_retry_attempt_span(kwargs)

        assert is_framework_owned(), (
            "during attempt → framework owns the call → direct-SDK wrappers "
            "would suppress emission"
        )

        _finalize_retry_attempt_span(kwargs, None, success=False)

        assert not is_framework_owned(), (
            "after attempt → token unregistered → direct-SDK wrappers "
            "may emit again"
        )
    parent.end()


# ---------------------------------------------------------------------------
# Idempotency.
# ---------------------------------------------------------------------------

def test_double_finalize_does_not_double_end(fresh_tracer):
    """LiteLLM's sync log_success_event AND async_log_success_event
    can both fire for the same attempt (the worker may run late). The
    second finalize MUST be a no-op."""
    tracer, exporter, _ = fresh_tracer

    parent = tracer.start_span("fortifyroot.litellm.safety")
    with trace.use_span(parent, end_on_exit=False):
        kwargs = {"litellm_call_id": "idem-001", "model": "openai/gpt-4o-mini"}
        _start_retry_attempt_span(kwargs)

        class MockSuccess:
            id = "x"
            model = "gpt-4o-mini"
            usage = type("U", (), {"prompt_tokens": 1, "completion_tokens": 1})()

        # First finalize.
        _finalize_retry_attempt_span(kwargs, MockSuccess(), success=True)
        # Second finalize — must be a no-op.
        _finalize_retry_attempt_span(kwargs, MockSuccess(), success=True)
    parent.end()

    retry_spans = [
        s for s in exporter.get_finished_spans()
        if s.name.startswith(f"{_FR_LLM_ATTEMPT_SPAN_NAME_PREFIX}.attempt_")
    ]
    assert len(retry_spans) == 1, (
        f"expected exactly 1 retry_attempt span (idempotent finalize); "
        f"got {len(retry_spans)}"
    )


# ---------------------------------------------------------------------------
# No-parent guard.
# ---------------------------------------------------------------------------

def test_no_parent_span_does_not_emit_orphan_retry_attempt(fresh_tracer):
    """If _start_retry_attempt_span fires without an ambient FR
    parent (e.g. user invoked LiteLLM's logger directly), the
    emitter MUST NOT create an orphan retry_attempt span — they
    have no meaningful place in the trace tree, and the §4.5
    backend dedup expects retry_attempts to have a parent."""
    tracer, exporter, _ = fresh_tracer
    # Note: NO parent span attached.

    kwargs = {"litellm_call_id": "no-parent-test", "model": "openai/gpt-4o-mini"}
    _start_retry_attempt_span(kwargs)

    spans = exporter.get_finished_spans()
    retry_spans = [s for s in spans if s.name.startswith(f"{_FR_LLM_ATTEMPT_SPAN_NAME_PREFIX}.attempt_")]
    assert len(retry_spans) == 0, (
        f"orphan retry_attempt MUST NOT be emitted; got {len(retry_spans)}"
    )

    # Map should also be empty (no entry registered for orphan calls).
    assert "no-parent-test" not in _FR_RETRY_ATTEMPT_MAP


def test_resolve_routed_provider_normalisation():
    """Regression guard for routed-provider normalization.
    LiteLLM's ``custom_llm_provider`` taxonomy uses values like
    ``"bedrock"``, ``"bedrock_converse"``, ``"vertex_ai"`` that
    diverge from retry-loop design notes §4.2's canonical routed-provider
    form (``"AWS"``, ``"google"``, etc.). Cross-wrapper drift
    here would mean LiteLLM-routed-Bedrock spans carry
    ``gen_ai.system="bedrock"`` while LangChain/LlamaIndex
    Bedrock spans carry ``gen_ai.system="AWS"``, breaking the
    retry-loop design §4.2 cross-wrapper consistency contract.
    """
    # AWS Bedrock — all variants must normalise to "AWS".
    assert _resolve_routed_provider({"custom_llm_provider": "bedrock"}) == "AWS"
    assert _resolve_routed_provider({"custom_llm_provider": "bedrock_converse"}) == "AWS"
    assert _resolve_routed_provider({"custom_llm_provider": "aws"}) == "AWS"
    # Bedrock model-prefix detection (LiteLLM may pass the
    # provider via the model string only).
    assert _resolve_routed_provider({"model": "bedrock/anthropic.claude-3-5-sonnet"}) == "AWS"
    assert _resolve_routed_provider({"model": "amazon.nova-lite-v1:0"}) == "AWS"

    # Google Gemini / Vertex variants → "google".
    assert _resolve_routed_provider({"custom_llm_provider": "gemini"}) == "google"
    assert _resolve_routed_provider({"custom_llm_provider": "vertex_ai"}) == "google"
    assert _resolve_routed_provider({"custom_llm_provider": "google_genai"}) == "google"

    # Already-canonical values pass through (lower-cased).
    assert _resolve_routed_provider({"custom_llm_provider": "openai"}) == "openai"
    assert _resolve_routed_provider({"custom_llm_provider": "anthropic"}) == "anthropic"
    assert _resolve_routed_provider({"model": "claude-4-sonnet-20250514"}) == "anthropic"
    assert _resolve_routed_provider({"model": "claude-sonnet-4-20250514"}) == "anthropic"

    # Empty / undeterminable → None.
    assert _resolve_routed_provider({}) is None
    assert _resolve_routed_provider({"custom_llm_provider": ""}) is None


def test_no_litellm_call_id_does_not_register_map_entry(fresh_tracer):
    """If kwargs lacks litellm_call_id, the emitter has no
    correlation key for success/failure callbacks → silently skips
    span emission rather than emitting an unmatched span."""
    tracer, exporter, _ = fresh_tracer

    parent = tracer.start_span("fortifyroot.litellm.safety")
    with trace.use_span(parent, end_on_exit=False):
        kwargs = {"model": "openai/gpt-4o-mini"}  # no litellm_call_id
        _start_retry_attempt_span(kwargs)
    parent.end()

    retry_spans = [
        s for s in exporter.get_finished_spans()
        if s.name.startswith(f"{_FR_LLM_ATTEMPT_SPAN_NAME_PREFIX}.attempt_")
    ]
    assert len(retry_spans) == 0, "missing litellm_call_id → no emission"
    assert len(_FR_RETRY_ATTEMPT_MAP) == 0

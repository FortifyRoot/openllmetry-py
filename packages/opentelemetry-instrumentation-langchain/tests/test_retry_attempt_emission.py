"""Tests for LangChain retry-aware attempt emission.

Covers:
  - Instrumentor symmetry: _FortifyRootRetryHandler is wired up by
    LangchainInstrumentor and removed at uninstrument.
  - on_chat_model_start path (chat models — F1 finding from
    retry-loop design C2 POC).
  - on_llm_start path (legacy completion LLMs).
  - Single-attempt happy path: 1 llm_attempt span, parent has
    has_attempt_child=true, llm_attempt has gen_ai.system /
    gen_ai.request.model / role attributes.
  - Multiple callback invocations sharing a parent_run_id → 3
    llm_attempt SIBLINGS under one parent span. LangChain numbering is
    conservative: shared workflow parents can contain unrelated LLM calls,
    so each emitted span remains attempt_1 / is_retry=false.
  - Marker timing (§4.5): parent gets the marker only AFTER the
    first attempt's start callback fires.
  - §4.7.1 token registration symmetry.
  - No-parent guard: skip emission when no ambient parent exists.
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from langchain_core.messages import HumanMessage
from opentelemetry import trace
from opentelemetry.instrumentation.fortifyroot import (
    is_framework_owned,
    retry_registry,
)
from opentelemetry.instrumentation.langchain import LangchainInstrumentor
from opentelemetry.instrumentation.langchain.retry_handler import (
    _FortifyRootRetryHandler,
    _FR_HAS_ATTEMPT_CHILD_KEY,
    _FR_LLM_ATTEMPT_SPAN_NAME_PREFIX,
    _reset_state_for_test,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)


@pytest.fixture
def fresh_tracer():
    """A fresh TracerProvider + in-memory exporter installed as the
    GLOBAL tracer provider (so the retry handler's
    ``trace.get_tracer(...)`` lookups route through it). Same pattern
    as the LiteLLM retry-attempt tests."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    try:
        trace.set_tracer_provider(provider)
    except Exception:
        pass
    current_provider = trace.get_tracer_provider()
    if current_provider is not provider:
        # Already set by an earlier test — add OUR exporter to it.
        try:
            current_provider.add_span_processor(SimpleSpanProcessor(exporter))
        except Exception:
            pass
    yield trace.get_tracer("test"), exporter, current_provider


@pytest.fixture(autouse=True)
def reset_state():
    """Each test starts with empty handler state + clean §4.7.1 registry."""
    retry_registry._reset_for_test()
    _reset_state_for_test()
    yield
    retry_registry._reset_for_test()
    _reset_state_for_test()


def _assert_conservative_attempts(spans):
    for span in spans:
        assert span.name == f"{_FR_LLM_ATTEMPT_SPAN_NAME_PREFIX}.attempt_1"
        assert span.attributes.get("fortifyroot.attempt.number") == 1
        assert span.attributes.get("fortifyroot.attempt.is_retry") is False


# ---------------------------------------------------------------------------
# Instrumentor symmetry.
# ---------------------------------------------------------------------------

def test_instrumentor_wires_retry_handler_globally(instrument_legacy):
    """When LangchainInstrumentor instruments, every newly-created
    BaseCallbackManager has the _FortifyRootRetryHandler installed
    via the patched __init__."""
    from langchain_core.callbacks import BaseCallbackManager

    del instrument_legacy  # only requested for its side effect
    # Create a manager; the patched __init__ should register both
    # the existing TraceloopCallbackHandler AND our retry handler.
    mgr = BaseCallbackManager(handlers=[])
    retry_handlers = [
        h for h in mgr.inheritable_handlers
        if isinstance(h, _FortifyRootRetryHandler)
    ]
    assert len(retry_handlers) == 1, (
        f"expected exactly 1 retry handler on the manager, "
        f"got {len(retry_handlers)}; "
        f"handlers={[type(h).__name__ for h in mgr.inheritable_handlers]}"
    )


def test_traceloop_handler_registered_before_fr_retry_handler(instrument_legacy):
    """Regression guard for LangChain callback ordering.
    LangChain dispatches callbacks in registration order. The
    Traceloop handler MUST run BEFORE the FR retry handler, so
    Traceloop's context-attach/detach discipline (existing,
    prior LangChain-validated behavior) is preserved.

    History: an earlier fix attempted the OPPOSITE order (FR
    retry first) so the OTel ambient at retry_attempt creation
    would be the workflow span — required for retry-attempt
    sibling-grouping. That ordering caused a DIFFERENT bug:
    Traceloop's `on_chat_model_start` / `on_llm_end`
    context-attach/detach pair lost discipline when interleaved
    with the FR retry handler running first, leaving STALE
    OTel context attached past the end of LangChain tests.
    Subsequent LiteLLM tests in the same pytest session inherited
    that stale context, causing all LiteLLM spans to share the
    LangChain trace_id and break the per-test trace-isolation
    invariant. All 10 LiteLLM tests failed under this scenario.

    Fix: keep Traceloop first; resolve the FR retry-attempt
    parent via Traceloop's ``spans`` dict + ``parent_run_id``
    (see ``_resolve_parent_span`` Strategy A in
    ``retry_handler.py``). This preserves Traceloop's
    discipline AND gives the FR retry handler the right parent.
    """
    from langchain_core.callbacks import BaseCallbackManager
    from opentelemetry.instrumentation.langchain.callback_handler import (
        TraceloopCallbackHandler,
    )

    del instrument_legacy  # only requested for its side effect
    mgr = BaseCallbackManager(handlers=[])
    types = [type(h) for h in mgr.inheritable_handlers]
    try:
        retry_idx = next(
            i for i, h in enumerate(mgr.inheritable_handlers)
            if isinstance(h, _FortifyRootRetryHandler)
        )
    except StopIteration:
        retry_idx = None
    try:
        traceloop_idx = next(
            i for i, h in enumerate(mgr.inheritable_handlers)
            if isinstance(h, TraceloopCallbackHandler)
        )
    except StopIteration:
        traceloop_idx = None
    assert retry_idx is not None, f"FR retry handler missing; saw {types}"
    assert traceloop_idx is not None, f"Traceloop handler missing; saw {types}"
    assert traceloop_idx < retry_idx, (
        f"Traceloop handler must precede FR retry handler in "
        f"inheritable_handlers — traceloop={traceloop_idx}, "
        f"retry={retry_idx}, all={types}. Wrong order breaks "
        f"Traceloop's context-attach discipline and leaks OTel "
        f"context across pytest test boundaries."
    )


def test_no_leaked_ambient_context_after_simulated_workflow(instrument_legacy):
    """Regression guard for leaked ambient context after a workflow.

    The trace-id-leak bug that motivated this guard: a LangChain
    workflow that exercises the patched BaseCallbackManager.__init__
    path could leave OTel context attached past the end of its
    callback flow. Subsequent test code (e.g. a LiteLLM test in the
    same pytest session) then saw a STILL-VALID ambient span carrying
    the LangChain trace_id, causing the LiteLLM safety_wrapper to
    parent under it and inherit the leaking trace_id.

    This test runs the LangChain callback flow synchronously
    (on_chat_model_start → on_llm_end) under a workflow span, ends
    the workflow span explicitly, and asserts that NOTHING is left
    attached to the ambient OTel context. If anything stays attached,
    a subsequent LiteLLM-style ``trace.get_current_span()`` would
    return a leaked span — exactly the failure mode this guard
    prevents.

    Uses the conftest ``instrument_legacy`` session-scoped fixture
    (NOT a fresh instrument/uninstrument cycle) so the test plays
    nicely with the rest of the session's instrumentation — see the
    docstring on ``test_uninstrument_invokes_unwrap_for_callback_manager``
    for why ad-hoc instrument/uninstrument breaks subsequent tests
    when LangchainInstrumentor is a Singleton.
    """
    del instrument_legacy  # only requested for its side effect
    from langchain_core.callbacks import BaseCallbackManager

    mgr = BaseCallbackManager(handlers=[])

    # Sanity: at the start of this test we expect no leaked
    # ambient. If something earlier in the session leaked, this
    # test isn't the right place to surface it — bail clean.
    starting_ambient_valid = trace.get_current_span().get_span_context().is_valid
    if starting_ambient_valid:
        pytest.skip(
            "ambient OTel context already non-empty at test start; "
            "earlier test in this session may have leaked — this "
            "guard test wants a clean baseline."
        )

    # Drive a synchronous chat-model-start → end pair through
    # every handler the instrumentor registered (Traceloop +
    # FR retry handler).
    run_id = uuid4()
    parent_run_id = uuid4()
    serialized = {"id": ["langchain_openai", "chat_models", "base", "ChatOpenAI"]}

    for h in list(mgr.inheritable_handlers):
        try:
            h.on_chat_model_start(
                serialized=serialized,
                messages=[],
                run_id=run_id,
                parent_run_id=parent_run_id,
                invocation_params={"model": "gpt-4o-mini"},
            )
        except Exception:
            # Tolerate handler-internal errors — the guard is
            # about context discipline, not handler correctness.
            pass

    for h in list(mgr.inheritable_handlers):
        try:
            h.on_llm_end(
                _FakeLLMResult(model="gpt-4o-mini"),
                run_id=run_id,
                parent_run_id=parent_run_id,
            )
        except Exception:
            pass

    # After the synthetic workflow, the ambient must be clean.
    # If any handler attached a context and forgot to detach, we
    # would see a still-valid ambient here — the precise failure
    # mode that contaminates the subsequent test.
    remaining = trace.get_current_span().get_span_context()
    assert not remaining.is_valid, (
        f"OTel ambient context leaked after LangChain workflow — "
        f"a future test in the same pytest session would inherit "
        f"trace_id={remaining.trace_id:032x}, span_id="
        f"{remaining.span_id:016x}. The ambient context must be "
        f"fully detached after the simulated workflow."
    )


def test_uninstrument_invokes_unwrap_for_callback_manager():
    """LangchainInstrumentor._uninstrument MUST call
    ``unwrap("langchain_core.callbacks", "BaseCallbackManager.__init__")``
    so the retry-handler wrap (and the existing Traceloop wrap) are
    removed. We can't assert on the GLOBAL BaseCallbackManager state
    because LangchainInstrumentor is a Singleton and the session-scoped
    ``instrument_legacy`` fixture in conftest.py installs the
    instrumentor before our tests run — interleaving instrument/
    uninstrument cycles produces order-dependent state.

    Instead, we verify the unwrap CALL is made by patching
    ``opentelemetry.instrumentation.langchain.unwrap`` and asserting
    the BaseCallbackManager.__init__ target is in the call args.
    """
    from unittest.mock import patch

    with patch(
        "opentelemetry.instrumentation.langchain.unwrap"
    ) as mock_unwrap, patch(
        "opentelemetry.instrumentation.langchain.uninstrument_safety_wrappers"
    ):
        instrumentor = object.__new__(LangchainInstrumentor)
        instrumentor.disable_trace_context_propagation = True
        LangchainInstrumentor._uninstrument(instrumentor)
        # First positional arg of unwrap is the module path.
        unwrap_targets = [
            (call.args[0], call.args[1])
            for call in mock_unwrap.call_args_list
            if len(call.args) >= 2
        ]
        assert ("langchain_core.callbacks", "BaseCallbackManager.__init__") in unwrap_targets, (
            f"_uninstrument must unwrap BaseCallbackManager.__init__; "
            f"observed unwrap calls: {unwrap_targets}"
        )


# ---------------------------------------------------------------------------
# on_chat_model_start vs on_llm_start (F1 finding from POC).
# ---------------------------------------------------------------------------

def test_on_chat_model_start_emits_retry_attempt(fresh_tracer):
    """Chat models fire on_chat_model_start (NOT on_llm_start) — F1
    finding from retry-loop design C2 POC. The handler MUST handle it."""
    tracer, exporter, _ = fresh_tracer
    handler = _FortifyRootRetryHandler()

    parent = tracer.start_span("workflow")
    with trace.use_span(parent, end_on_exit=False):
        run_id = uuid4()
        parent_run_id = uuid4()
        handler.on_chat_model_start(
            serialized={"id": ["langchain_openai", "chat_models", "base", "ChatOpenAI"]},
            messages=[[HumanMessage(content="hello masked [EMAIL]")]],
            run_id=run_id,
            parent_run_id=parent_run_id,
            invocation_params={"model": "gpt-4o-mini"},
        )
        handler.on_llm_end(_FakeLLMResult(model="gpt-4o-mini"), run_id=run_id)
    parent.end()

    retry_spans = [
        s for s in exporter.get_finished_spans()
        if s.name.startswith(f"{_FR_LLM_ATTEMPT_SPAN_NAME_PREFIX}.attempt_")
    ]
    assert len(retry_spans) == 1, (
        f"on_chat_model_start MUST produce a retry_attempt span; got {len(retry_spans)}"
    )
    _assert_conservative_attempts(retry_spans)
    rs = retry_spans[0]
    assert rs.attributes.get("fortifyroot.span.role") == "llm_attempt"
    assert rs.attributes.get("gen_ai.system") == "openai", (
        "gen_ai.system must be the routed provider (openai), NOT 'langchain'"
    )
    assert rs.attributes.get("gen_ai.request.model") == "gpt-4o-mini"
    assert rs.attributes.get("gen_ai.prompt.0.role") == "user"
    assert rs.attributes.get("gen_ai.prompt.0.content") == "hello masked [EMAIL]"


def test_on_llm_start_emits_retry_attempt(fresh_tracer):
    """Legacy completion LLMs fire on_llm_start. Same handling."""
    tracer, exporter, _ = fresh_tracer
    handler = _FortifyRootRetryHandler()

    parent = tracer.start_span("workflow")
    with trace.use_span(parent, end_on_exit=False):
        run_id = uuid4()
        parent_run_id = uuid4()
        handler.on_llm_start(
            serialized={"id": ["langchain_anthropic", "llms", "Anthropic"]},
            prompts=["hi"],
            run_id=run_id,
            parent_run_id=parent_run_id,
            invocation_params={"model": "claude-haiku-4-5"},
        )
        handler.on_llm_end(_FakeLLMResult(model="claude-haiku-4-5"), run_id=run_id)
    parent.end()

    retry_spans = [
        s for s in exporter.get_finished_spans()
        if s.name.startswith(f"{_FR_LLM_ATTEMPT_SPAN_NAME_PREFIX}.attempt_")
    ]
    assert len(retry_spans) == 1
    _assert_conservative_attempts(retry_spans)
    assert retry_spans[0].attributes.get("gen_ai.system") == "anthropic"
    assert retry_spans[0].attributes.get("gen_ai.request.model") == "claude-haiku-4-5"
    assert retry_spans[0].attributes.get("gen_ai.prompt.0.role") == "user"
    assert retry_spans[0].attributes.get("gen_ai.prompt.0.content") == "hi"


# ---------------------------------------------------------------------------
# Single-attempt happy path.
# ---------------------------------------------------------------------------

def test_single_attempt_emits_one_sibling_with_marker(fresh_tracer):
    """A single attempt produces ONE retry_attempt span, parent under
    the workflow, and the workflow span carries
    has_attempt_child=true."""
    tracer, exporter, _ = fresh_tracer
    handler = _FortifyRootRetryHandler()

    parent = tracer.start_span("workflow")
    with trace.use_span(parent, end_on_exit=False):
        run_id = uuid4()
        parent_run_id = uuid4()
        handler.on_chat_model_start(
            serialized={"id": ["langchain_openai", "ChatOpenAI"]},
            messages=[],
            run_id=run_id,
            parent_run_id=parent_run_id,
            invocation_params={"model": "gpt-4o-mini"},
        )
        handler.on_llm_end(
            _FakeLLMResult(
                model="gpt-4o-mini",
                response_id="resp-abc",
                input_tokens=10,
                output_tokens=5,
            ),
            run_id=run_id,
        )
    parent.end()

    spans = exporter.get_finished_spans()
    parent_exported = next(s for s in spans if s.name == "workflow")
    retry_span = next(s for s in spans if s.name.startswith(f"{_FR_LLM_ATTEMPT_SPAN_NAME_PREFIX}.attempt_"))
    _assert_conservative_attempts([retry_span])

    assert retry_span.parent.span_id == parent_exported.context.span_id, (
        "retry_attempt MUST be a child of the workflow span (sibling-grouping requires this)"
    )
    assert parent_exported.attributes.get(_FR_HAS_ATTEMPT_CHILD_KEY) is True
    assert retry_span.attributes.get("gen_ai.response.id") == "resp-abc"
    assert retry_span.attributes.get("gen_ai.usage.input_tokens") == 10
    assert retry_span.attributes.get("gen_ai.usage.output_tokens") == 5


# ---------------------------------------------------------------------------
# Multi-attempt retry path — siblings under one parent.
# ---------------------------------------------------------------------------

def test_three_attempts_share_parent_under_workflow(fresh_tracer):
    """When 3 attempts share the SAME parent_run_id (the canonical
    LangChain Runnable.with_retry shape), all 3 retry_attempt spans
    are SIBLINGS under the same workflow parent — the structural
    invariant RetryDetectorProc needs for retry-loop detection."""
    tracer, exporter, _ = fresh_tracer
    handler = _FortifyRootRetryHandler()

    parent = tracer.start_span("workflow")
    parent_run_id = uuid4()  # shared across 3 attempts
    serialized = {"id": ["langchain_openai", "ChatOpenAI"]}

    class MockHTTPError(Exception):
        def __init__(self, status_code: int):
            self.status_code = status_code
            super().__init__(f"http {status_code}")

    with trace.use_span(parent, end_on_exit=False):
        # Attempt 1 — 429
        rid1 = uuid4()
        handler.on_chat_model_start(
            serialized=serialized, messages=[], run_id=rid1,
            parent_run_id=parent_run_id,
            invocation_params={"model": "gpt-4o-mini"},
        )
        handler.on_llm_error(MockHTTPError(429), run_id=rid1)

        # Attempt 2 — 429
        rid2 = uuid4()
        handler.on_chat_model_start(
            serialized=serialized, messages=[], run_id=rid2,
            parent_run_id=parent_run_id,
            invocation_params={"model": "gpt-4o-mini"},
        )
        handler.on_llm_error(MockHTTPError(429), run_id=rid2)

        # Attempt 3 — 200
        rid3 = uuid4()
        handler.on_chat_model_start(
            serialized=serialized, messages=[], run_id=rid3,
            parent_run_id=parent_run_id,
            invocation_params={"model": "gpt-4o-mini"},
        )
        handler.on_llm_end(_FakeLLMResult(model="gpt-4o-mini"), run_id=rid3)
    parent.end()

    spans = exporter.get_finished_spans()
    parent_exported = next(s for s in spans if s.name == "workflow")
    retry_spans = [s for s in spans if s.name.startswith(f"{_FR_LLM_ATTEMPT_SPAN_NAME_PREFIX}.attempt_")]

    assert len(retry_spans) == 3, f"expected 3 retry_attempt spans, got {len(retry_spans)}"
    _assert_conservative_attempts(retry_spans)

    # All 3 must share the workflow as parent — this is the sibling
    # invariant RetryDetectorProc relies on.
    parent_span_ids = {s.parent.span_id for s in retry_spans}
    assert parent_span_ids == {parent_exported.context.span_id}, (
        f"all 3 retry_attempts MUST be siblings under one parent; "
        f"saw distinct parents: {parent_span_ids}"
    )

    # 2 ERROR + 1 OK.
    from opentelemetry.trace import StatusCode
    error_count = sum(1 for s in retry_spans if s.status.status_code == StatusCode.ERROR)
    ok_count = sum(1 for s in retry_spans if s.status.status_code == StatusCode.OK)
    assert error_count == 2, f"expected 2 ERROR, got {error_count}"
    assert ok_count == 1, f"expected 1 OK, got {ok_count}"

    # The 2 error spans must carry http.status_code=429 + error.type.
    for s in retry_spans:
        if s.status.status_code == StatusCode.ERROR:
            assert s.attributes.get("http.status_code") == 429
            assert s.attributes.get("error.type") == "MockHTTPError"


# ---------------------------------------------------------------------------
# Marker timing (§4.5).
# ---------------------------------------------------------------------------

def test_marker_set_AFTER_first_attempt_not_at_parent_creation(fresh_tracer):
    """Per §4.5 marker-timing: parent gets has_attempt_child=true
    only AFTER the first attempt's start callback fires, NOT at parent
    creation."""
    tracer, exporter, _ = fresh_tracer
    handler = _FortifyRootRetryHandler()

    parent = tracer.start_span("workflow")
    # Before any retry_attempt: no marker.
    assert _FR_HAS_ATTEMPT_CHILD_KEY not in dict(parent.attributes or {})

    with trace.use_span(parent, end_on_exit=False):
        run_id = uuid4()
        parent_run_id = uuid4()
        handler.on_chat_model_start(
            serialized={"id": ["langchain_openai", "ChatOpenAI"]},
            messages=[],
            run_id=run_id,
            parent_run_id=parent_run_id,
            invocation_params={"model": "gpt-4o-mini"},
        )
    # After first attempt start: parent has the marker.
    assert dict(parent.attributes or {}).get(_FR_HAS_ATTEMPT_CHILD_KEY) is True

    handler.on_llm_end(_FakeLLMResult(model="gpt-4o-mini"), run_id=run_id)
    parent.end()


def test_marker_NOT_set_when_no_attempts_fire(fresh_tracer):
    """If no on_chat_model_start / on_llm_start ever fires for a
    parent, the parent never gets the marker — graceful degradation
    per §4.5."""
    tracer, exporter, _ = fresh_tracer

    parent = tracer.start_span("workflow")
    parent.end()

    parent_exported = next(
        s for s in exporter.get_finished_spans() if s.name == "workflow"
    )
    assert _FR_HAS_ATTEMPT_CHILD_KEY not in (parent_exported.attributes or {})


# ---------------------------------------------------------------------------
# §4.7.1 token registration.
# ---------------------------------------------------------------------------

def test_framework_token_registered_during_attempt_unregistered_after(fresh_tracer):
    """While a retry_attempt is in flight, is_framework_owned() is
    True for the current thread; after the end/error callback fires,
    it returns to False."""
    tracer, _, _ = fresh_tracer
    handler = _FortifyRootRetryHandler()

    parent = tracer.start_span("workflow")
    with trace.use_span(parent, end_on_exit=False):
        assert not is_framework_owned()
        run_id = uuid4()
        handler.on_chat_model_start(
            serialized={"id": ["langchain_openai", "ChatOpenAI"]},
            messages=[],
            run_id=run_id,
            parent_run_id=uuid4(),
            invocation_params={"model": "gpt-4o-mini"},
        )
        assert is_framework_owned(), "during attempt → framework owns the call"
        handler.on_llm_end(_FakeLLMResult(model="gpt-4o-mini"), run_id=run_id)
        assert not is_framework_owned(), "after attempt → released"
    parent.end()


# ---------------------------------------------------------------------------
# No-parent guard.
# ---------------------------------------------------------------------------

def test_no_parent_does_not_emit_orphan_retry_attempt(fresh_tracer):
    """If the handler fires with no ambient OTel parent and no
    parent_run_id memo, it must NOT emit an orphan retry_attempt
    (no place in the trace tree)."""
    tracer, exporter, _ = fresh_tracer
    handler = _FortifyRootRetryHandler()

    # No parent span attached at all.
    handler.on_chat_model_start(
        serialized={"id": ["langchain_openai", "ChatOpenAI"]},
        messages=[],
        run_id=uuid4(),
        parent_run_id=uuid4(),
        invocation_params={"model": "gpt-4o-mini"},
    )

    retry_spans = [
        s for s in exporter.get_finished_spans()
        if s.name.startswith(f"{_FR_LLM_ATTEMPT_SPAN_NAME_PREFIX}.attempt_")
    ]
    assert len(retry_spans) == 0


# ---------------------------------------------------------------------------
# Idempotency.
# ---------------------------------------------------------------------------

def test_double_finalize_does_not_double_end(fresh_tracer):
    """Calling on_llm_end twice for the same run_id is a no-op the
    second time."""
    tracer, exporter, _ = fresh_tracer
    handler = _FortifyRootRetryHandler()

    parent = tracer.start_span("workflow")
    with trace.use_span(parent, end_on_exit=False):
        run_id = uuid4()
        handler.on_chat_model_start(
            serialized={"id": ["langchain_openai", "ChatOpenAI"]},
            messages=[],
            run_id=run_id,
            parent_run_id=uuid4(),
            invocation_params={"model": "gpt-4o-mini"},
        )
        handler.on_llm_end(_FakeLLMResult(model="gpt-4o-mini"), run_id=run_id)
        handler.on_llm_end(_FakeLLMResult(model="gpt-4o-mini"), run_id=run_id)  # no-op
    parent.end()

    retry_spans = [
        s for s in exporter.get_finished_spans()
        if s.name.startswith(f"{_FR_LLM_ATTEMPT_SPAN_NAME_PREFIX}.attempt_")
    ]
    assert len(retry_spans) == 1, (
        f"expected exactly 1 retry_attempt span (idempotent finalize); got {len(retry_spans)}"
    )


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------

class _FakeGeneration:
    def __init__(self, response_id=None):
        self.generation_info = {"response_id": response_id} if response_id else {}


class _FakeLLMResult:
    def __init__(self, model=None, response_id=None, input_tokens=None, output_tokens=None):
        self.generations = [[_FakeGeneration(response_id=response_id)]]
        self.llm_output = {}
        if model:
            self.llm_output["model_name"] = model
        if input_tokens is not None or output_tokens is not None:
            usage = {}
            if input_tokens is not None:
                usage["prompt_tokens"] = input_tokens
            if output_tokens is not None:
                usage["completion_tokens"] = output_tokens
            self.llm_output["token_usage"] = usage

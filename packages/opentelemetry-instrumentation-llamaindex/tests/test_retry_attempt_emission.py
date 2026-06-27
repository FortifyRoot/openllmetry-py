"""Tests for LlamaIndex retry-aware attempt emission.

Covers:
  - F4 de-dup contract: outer chat() AND inner _chat() both fire
    dispatcher spans, but only ONE retry_attempt span is emitted
    (the outer's) — the load-bearing test.
  - Method-name whitelist: non-LLM dispatcher spans don't trigger
    retry_attempt; outer-only filter correctly rejects "_chat".
  - BaseLLM instance check: dispatcher spans on non-LLM classes
    are ignored.
  - Single-attempt happy path under tenacity-style wrapping.
  - Multiple outer calls sharing a parent → 3 llm_attempt SIBLINGS
    under one OTel parent. LlamaIndex numbering is conservative:
    shared workflow parents can contain unrelated LLM calls, so each
    emitted span remains attempt_1 / is_retry=false.
  - Marker timing (§4.5).
  - §4.7.1 token registration symmetry.
  - No-parent guard.
  - Idempotent finalize.
"""

from __future__ import annotations

import inspect

import pytest
from llama_index.core.base.llms.base import BaseLLM
from opentelemetry import trace
from opentelemetry.instrumentation.fortifyroot import (
    is_framework_owned,
    retry_registry,
)
from opentelemetry.instrumentation.llamaindex.retry_handler import (
    _FortifyRootRetryHandler,
    _FR_HAS_ATTEMPT_CHILD_KEY,
    _FR_RETRY_ATTEMPT_MAP,
    _FR_LLM_ATTEMPT_SPAN_NAME_PREFIX,
    _is_outer_llm_method,
    _OUTER_LLM_METHODS,
    _reset_state_for_test,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)


@pytest.fixture
def fresh_tracer():
    """Fresh TracerProvider + in-memory exporter installed as global,
    matching the LiteLLM retry-attempt / LangChain retry-attempt fixture pattern."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    try:
        trace.set_tracer_provider(provider)
    except Exception:
        pass
    current_provider = trace.get_tracer_provider()
    if current_provider is not provider:
        try:
            current_provider.add_span_processor(SimpleSpanProcessor(exporter))
        except Exception:
            pass
    yield trace.get_tracer("test"), exporter, current_provider


@pytest.fixture(autouse=True)
def reset_state():
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
# Helpers
# ---------------------------------------------------------------------------

class _FakeLLM(BaseLLM):
    """A minimal BaseLLM-shaped object for testing. Just satisfies the
    isinstance(instance, BaseLLM) check; doesn't implement any
    abstract methods."""

    model: str = "gpt-4o-mini"

    @classmethod
    def class_name(cls) -> str:
        return "FakeLLM"

    @property
    def metadata(self):
        from llama_index.core.base.llms.types import LLMMetadata
        return LLMMetadata()

    def chat(self, messages, **kwargs):  # pragma: no cover
        raise NotImplementedError

    async def achat(self, messages, **kwargs):  # pragma: no cover
        raise NotImplementedError

    def stream_chat(self, messages, **kwargs):  # pragma: no cover
        raise NotImplementedError

    async def astream_chat(self, messages, **kwargs):  # pragma: no cover
        raise NotImplementedError

    def complete(self, prompt, formatted=False, **kwargs):  # pragma: no cover
        raise NotImplementedError

    async def acomplete(self, prompt, formatted=False, **kwargs):  # pragma: no cover
        raise NotImplementedError

    def stream_complete(self, prompt, formatted=False, **kwargs):  # pragma: no cover
        raise NotImplementedError

    async def astream_complete(self, prompt, formatted=False, **kwargs):  # pragma: no cover
        raise NotImplementedError


def _empty_bound_args():
    """Return an empty inspect.BoundArguments for tests that don't
    care about the arg payload."""
    sig = inspect.signature(lambda: None)
    return sig.bind()


def _prompt_bound_args(prompt: str):
    sig = inspect.signature(lambda prompt: None)
    return sig.bind(prompt)


def _make_id(class_name: str, method: str) -> str:
    """Mimic LlamaIndex's dispatcher span id format
    ``ClassName.method-uuid``."""
    return f"{class_name}.{method}-test-uuid-1234"


class _FakeChatResponse:
    def __init__(self, response_id=None, model=None, prompt_tokens=None, completion_tokens=None):
        class _Raw:
            pass
        self.raw = _Raw()
        if response_id:
            self.raw.id = response_id
        if model:
            self.raw.model = model
        if prompt_tokens is not None or completion_tokens is not None:
            class _Usage:
                pass
            self.raw.usage = _Usage()
            if prompt_tokens is not None:
                self.raw.usage.prompt_tokens = prompt_tokens
            if completion_tokens is not None:
                self.raw.usage.completion_tokens = completion_tokens


# ---------------------------------------------------------------------------
# F4 de-dup contract — the LOAD-BEARING test.
# ---------------------------------------------------------------------------

def test_outer_method_emits_inner_method_does_not(fresh_tracer):
    """LlamaIndex dispatcher fires spans on BOTH the public ``chat()``
    AND the inner ``_chat()`` (F4 finding from POC). LlamaIndex retry-attempt hooks
    SpanHandler.new_span and filters via
    ``_is_outer_llm_method`` → only the outer method emits a
    retry_attempt span.

    Net: ONE HTTP attempt = ONE retry_attempt span (NOT TWO),
    even though the dispatcher emits two spans.
    """
    tracer, exporter, _ = fresh_tracer
    handler = _FortifyRootRetryHandler()
    instance = _FakeLLM()

    parent = tracer.start_span("workflow")
    with trace.use_span(parent, end_on_exit=False):
        # Outer chat() span fires → emit retry_attempt.
        outer_id = _make_id("FakeLLM", "chat")
        handler.new_span(id_=outer_id, bound_args=_empty_bound_args(), instance=instance)

        # Inner _chat() span fires → DO NOT emit retry_attempt.
        inner_id = _make_id("FakeLLM", "_chat")
        handler.new_span(id_=inner_id, bound_args=_empty_bound_args(), instance=instance)

        # Inner _chat() exits → no-op (no entry to finalize).
        handler.prepare_to_exit_span(
            id_=inner_id, bound_args=_empty_bound_args(),
            instance=instance, result=_FakeChatResponse(model="gpt-4o-mini"),
        )
        # Outer chat() exits → finalize the retry_attempt.
        handler.prepare_to_exit_span(
            id_=outer_id, bound_args=_empty_bound_args(),
            instance=instance, result=_FakeChatResponse(model="gpt-4o-mini"),
        )
    parent.end()

    retry_spans = [
        s for s in exporter.get_finished_spans()
        if s.name.startswith(f"{_FR_LLM_ATTEMPT_SPAN_NAME_PREFIX}.attempt_")
    ]
    assert len(retry_spans) == 1, (
        f"F4 de-dup invariant: outer chat() emits ONE retry_attempt; "
        f"inner _chat() must NOT emit. Got {len(retry_spans)} spans."
    )
    _assert_conservative_attempts(retry_spans)


def test_is_outer_llm_method_filter():
    """Direct unit test of the filter function. Outer public methods
    pass, inner private methods fail, non-LLM instances fail."""
    instance = _FakeLLM()
    not_instance = "not-an-llm"

    # Outer public methods — all pass.
    assert _is_outer_llm_method(_make_id("FakeLLM", "chat"), instance)
    assert _is_outer_llm_method(_make_id("FakeLLM", "achat"), instance)
    assert _is_outer_llm_method(_make_id("FakeLLM", "complete"), instance)
    assert _is_outer_llm_method(_make_id("FakeLLM", "stream_chat"), instance)
    assert _is_outer_llm_method(_make_id("FakeLLM", "astream_complete"), instance)

    # Inner private methods — all fail.
    assert not _is_outer_llm_method(_make_id("FakeLLM", "_chat"), instance)
    assert not _is_outer_llm_method(_make_id("FakeLLM", "_complete"), instance)
    assert not _is_outer_llm_method(_make_id("FakeLLM", "_predict"), instance)

    # Not a BaseLLM instance — fail.
    assert not _is_outer_llm_method(_make_id("FakeLLM", "chat"), not_instance)
    assert not _is_outer_llm_method(_make_id("FakeLLM", "chat"), None)

    # Malformed dispatcher id — fail.
    assert not _is_outer_llm_method("malformed", instance)


def test_outer_method_whitelist_is_complete():
    """Sanity-check the _OUTER_LLM_METHODS whitelist matches the
    BaseLLM public surface. If LlamaIndex adds new public methods
    in a future version, the whitelist may need updating — this
    test alerts on that condition."""
    expected_at_minimum = {"chat", "achat", "complete", "acomplete"}
    assert expected_at_minimum.issubset(_OUTER_LLM_METHODS)


# ---------------------------------------------------------------------------
# Single-attempt happy path.
# ---------------------------------------------------------------------------

def test_single_attempt_emits_one_retry_attempt(fresh_tracer):
    tracer, exporter, _ = fresh_tracer
    handler = _FortifyRootRetryHandler()
    instance = _FakeLLM()

    parent = tracer.start_span("workflow")
    with trace.use_span(parent, end_on_exit=False):
        id_ = _make_id("FakeLLM", "complete")
        handler.new_span(
            id_=id_,
            bound_args=_prompt_bound_args("hello masked [EMAIL]"),
            instance=instance,
        )
        handler.prepare_to_exit_span(
            id_=id_, bound_args=_empty_bound_args(),
            instance=instance,
            result=_FakeChatResponse(
                response_id="resp-001", model="gpt-4o-mini",
                prompt_tokens=10, completion_tokens=5,
            ),
        )
    parent.end()

    spans = exporter.get_finished_spans()
    parent_exported = next(s for s in spans if s.name == "workflow")
    retry_span = next(s for s in spans if s.name.startswith(f"{_FR_LLM_ATTEMPT_SPAN_NAME_PREFIX}.attempt_"))
    _assert_conservative_attempts([retry_span])

    assert retry_span.parent.span_id == parent_exported.context.span_id
    assert parent_exported.attributes.get(_FR_HAS_ATTEMPT_CHILD_KEY) is True
    assert retry_span.attributes.get("fortifyroot.span.role") == "llm_attempt"
    assert retry_span.attributes.get("gen_ai.request.model") == "gpt-4o-mini"
    assert retry_span.attributes.get("gen_ai.response.id") == "resp-001"
    assert retry_span.attributes.get("gen_ai.usage.input_tokens") == 10
    assert retry_span.attributes.get("gen_ai.usage.output_tokens") == 5
    assert retry_span.attributes.get("gen_ai.prompt.0.role") == "user"
    assert retry_span.attributes.get("gen_ai.prompt.0.content") == "hello masked [EMAIL]"


# ---------------------------------------------------------------------------
# Multi-attempt retry path — siblings under one parent.
# ---------------------------------------------------------------------------

def test_three_attempts_share_one_otel_parent(fresh_tracer):
    """Under tenacity wrapping at the application layer, each retry
    creates a fresh dispatcher span tree (different id_ per attempt),
    BUT they all run under the same enclosing OTel span as ambient.
    The retry_handler uses ambient OTel context as parent → all 3
    retry_attempts become SIBLINGS under one OTel parent."""
    tracer, exporter, _ = fresh_tracer
    handler = _FortifyRootRetryHandler()
    instance = _FakeLLM()

    class MockHTTPError(Exception):
        def __init__(self, status_code):
            self.status_code = status_code
            super().__init__(f"http {status_code}")

    parent = tracer.start_span("workflow")
    with trace.use_span(parent, end_on_exit=False):
        # Attempt 1 — 429
        id1 = _make_id("FakeLLM", "chat") + "-attempt1"
        handler.new_span(id_=id1, bound_args=_empty_bound_args(), instance=instance)
        # Inner _chat span also fires — but is filtered.
        inner1 = _make_id("FakeLLM", "_chat") + "-attempt1"
        handler.new_span(
            id_=inner1, bound_args=_empty_bound_args(), instance=instance,
        )
        handler.prepare_to_drop_span(
            id_=inner1, bound_args=_empty_bound_args(),
            instance=instance, err=MockHTTPError(429),
        )
        handler.prepare_to_drop_span(
            id_=id1, bound_args=_empty_bound_args(),
            instance=instance, err=MockHTTPError(429),
        )

        # Attempt 2 — 429
        id2 = _make_id("FakeLLM", "chat") + "-attempt2"
        handler.new_span(id_=id2, bound_args=_empty_bound_args(), instance=instance)
        handler.prepare_to_drop_span(id_=id2, bound_args=_empty_bound_args(), instance=instance, err=MockHTTPError(429))

        # Attempt 3 — 200
        id3 = _make_id("FakeLLM", "chat") + "-attempt3"
        handler.new_span(id_=id3, bound_args=_empty_bound_args(), instance=instance)
        handler.prepare_to_exit_span(
            id_=id3, bound_args=_empty_bound_args(), instance=instance,
            result=_FakeChatResponse(model="gpt-4o-mini"),
        )
    parent.end()

    spans = exporter.get_finished_spans()
    parent_exported = next(s for s in spans if s.name == "workflow")
    retry_spans = [s for s in spans if s.name.startswith(f"{_FR_LLM_ATTEMPT_SPAN_NAME_PREFIX}.attempt_")]

    assert len(retry_spans) == 3, (
        f"3 outer-method spans should produce 3 retry_attempts (inner _chat filtered out); "
        f"got {len(retry_spans)}"
    )
    _assert_conservative_attempts(retry_spans)
    parent_ids = {s.parent.span_id for s in retry_spans}
    assert parent_ids == {parent_exported.context.span_id}, (
        f"all 3 retry_attempts must be SIBLINGS under one OTel parent; "
        f"saw distinct parents: {parent_ids}"
    )

    from opentelemetry.trace import StatusCode
    error_count = sum(1 for s in retry_spans if s.status.status_code == StatusCode.ERROR)
    ok_count = sum(1 for s in retry_spans if s.status.status_code == StatusCode.OK)
    assert error_count == 2 and ok_count == 1
    for s in retry_spans:
        if s.status.status_code == StatusCode.ERROR:
            assert s.attributes.get("http.status_code") == 429
            assert s.attributes.get("error.type") == "MockHTTPError"


# ---------------------------------------------------------------------------
# Marker timing (§4.5).
# ---------------------------------------------------------------------------

def test_marker_set_AFTER_first_attempt(fresh_tracer):
    tracer, _, _ = fresh_tracer
    handler = _FortifyRootRetryHandler()
    instance = _FakeLLM()

    parent = tracer.start_span("workflow")
    assert _FR_HAS_ATTEMPT_CHILD_KEY not in dict(parent.attributes or {})

    with trace.use_span(parent, end_on_exit=False):
        handler.new_span(
            id_=_make_id("FakeLLM", "chat"), bound_args=_empty_bound_args(),
            instance=instance,
        )
    assert dict(parent.attributes or {}).get(_FR_HAS_ATTEMPT_CHILD_KEY) is True

    handler.prepare_to_exit_span(
        id_=_make_id("FakeLLM", "chat"), bound_args=_empty_bound_args(),
        instance=instance, result=_FakeChatResponse(model="gpt-4o-mini"),
    )
    parent.end()


def test_marker_NOT_set_when_inner_only_fires(fresh_tracer):
    """If only inner methods fire (no outer chat ever), no
    retry_attempt is emitted and the parent stays unmarked."""
    tracer, exporter, _ = fresh_tracer
    handler = _FortifyRootRetryHandler()
    instance = _FakeLLM()

    parent = tracer.start_span("workflow")
    with trace.use_span(parent, end_on_exit=False):
        # Only inner method fires.
        handler.new_span(
            id_=_make_id("FakeLLM", "_chat"), bound_args=_empty_bound_args(),
            instance=instance,
        )
        handler.prepare_to_exit_span(
            id_=_make_id("FakeLLM", "_chat"), bound_args=_empty_bound_args(),
            instance=instance, result=_FakeChatResponse(model="gpt-4o-mini"),
        )
    parent.end()

    parent_exported = next(
        s for s in exporter.get_finished_spans() if s.name == "workflow"
    )
    assert _FR_HAS_ATTEMPT_CHILD_KEY not in (parent_exported.attributes or {})


# ---------------------------------------------------------------------------
# §4.7.1 token registration.
# ---------------------------------------------------------------------------

def test_framework_token_lifecycle(fresh_tracer):
    tracer, _, _ = fresh_tracer
    handler = _FortifyRootRetryHandler()
    instance = _FakeLLM()

    parent = tracer.start_span("workflow")
    with trace.use_span(parent, end_on_exit=False):
        assert not is_framework_owned()
        id_ = _make_id("FakeLLM", "chat")
        handler.new_span(id_=id_, bound_args=_empty_bound_args(), instance=instance)
        assert is_framework_owned(), "during attempt → framework owns the call"
        handler.prepare_to_exit_span(
            id_=id_, bound_args=_empty_bound_args(),
            instance=instance, result=_FakeChatResponse(model="gpt-4o-mini"),
        )
        assert not is_framework_owned(), "after attempt → released"
    parent.end()


# ---------------------------------------------------------------------------
# No-parent guard.
# ---------------------------------------------------------------------------

def test_no_ambient_parent_no_emission(fresh_tracer):
    tracer, exporter, _ = fresh_tracer
    handler = _FortifyRootRetryHandler()
    instance = _FakeLLM()

    # No parent span attached.
    handler.new_span(
        id_=_make_id("FakeLLM", "chat"), bound_args=_empty_bound_args(),
        instance=instance,
    )

    retry_spans = [
        s for s in exporter.get_finished_spans()
        if s.name.startswith(f"{_FR_LLM_ATTEMPT_SPAN_NAME_PREFIX}.attempt_")
    ]
    assert len(retry_spans) == 0
    assert len(_FR_RETRY_ATTEMPT_MAP) == 0


# ---------------------------------------------------------------------------
# Idempotency.
# ---------------------------------------------------------------------------

def test_double_finalize_is_idempotent(fresh_tracer):
    tracer, exporter, _ = fresh_tracer
    handler = _FortifyRootRetryHandler()
    instance = _FakeLLM()

    parent = tracer.start_span("workflow")
    with trace.use_span(parent, end_on_exit=False):
        id_ = _make_id("FakeLLM", "chat")
        handler.new_span(id_=id_, bound_args=_empty_bound_args(), instance=instance)
        handler.prepare_to_exit_span(
            id_=id_, bound_args=_empty_bound_args(),
            instance=instance, result=_FakeChatResponse(model="gpt-4o-mini"),
        )
        # Second finalize — no-op.
        handler.prepare_to_exit_span(
            id_=id_, bound_args=_empty_bound_args(),
            instance=instance, result=_FakeChatResponse(model="gpt-4o-mini"),
        )
    parent.end()

    retry_spans = [
        s for s in exporter.get_finished_spans()
        if s.name.startswith(f"{_FR_LLM_ATTEMPT_SPAN_NAME_PREFIX}.attempt_")
    ]
    assert len(retry_spans) == 1

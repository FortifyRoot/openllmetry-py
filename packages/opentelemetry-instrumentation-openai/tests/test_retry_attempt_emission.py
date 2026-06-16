"""Tests for ST-10.4 OpenAI direct-SDK retry-attempt emission.

Covers (per RETRY_LOOP.md §4.4 OpenAI row + §4.7 suppression):
  - Instrumentor symmetry: install/uninstall flips state cleanly and is
    idempotent.
  - Single-attempt happy path: ONE llm_attempt span under the active
    parent; parent gets has_attempt_child=true; span carries
    role / gen_ai.system=openai / gen_ai.request.model / http.status_code.
  - Multi-attempt retry path: N siblings under one parent (the
    structural shape RetryDetectorProc relies on).
  - Error-attempt: status=ERROR, http.status_code set, error.type set.
  - No-parent guard: skip when no ambient OTel parent.
  - §4.7 suppression — BOTH paths:
      * OTel context SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY active → skip
      * framework-attempt registry says owned (LiteLLM/LangChain/LlamaIndex
        wrapping in flight) → skip
  - §4.4.1 endpoint allow-list: non-LLM SDK traffic (e.g. /v1/models)
    does NOT emit a retry_attempt span.
  - Private-symbol missing guard: import-time wrappability check —
    if the private symbol is absent, log warning + skip, do NOT crash.

These are UNIT tests — we drive ``_sync_send_wrapper`` /
``_async_send_wrapper`` directly with fake httpx Request / Response
objects so the test doesn't depend on real openai SDK behaviour or
network calls.
"""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace
from typing import Any, Optional
from unittest.mock import patch

import pytest
from opentelemetry import context as context_api
from opentelemetry import trace
from opentelemetry.instrumentation.fortifyroot import (
    clear_attempt_counters_for_test,
    is_framework_owned,
    register_framework_attempt,
    retry_registry,
    unregister_framework_attempt,
)
from opentelemetry.instrumentation.openai.retry_handler import (
    _FR_HAS_ATTEMPT_CHILD_KEY,
    _FR_LLM_ATTEMPT_SPAN_NAME_PREFIX,
    OPENAI_DIRECT_RETRY_PARENT_ACTIVE_KEY,
    _async_send_wrapper,
    _has_wrappable_symbol,
    _is_installed_for_test,
    _sync_send_wrapper,
    instrument_retry_emitter,
    uninstrument_retry_emitter,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.semconv_ai import SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY


# ---------------------------------------------------------------------------
# Test infrastructure.
# ---------------------------------------------------------------------------

@pytest.fixture
def fresh_tracer():
    """Fresh TracerProvider + in-memory exporter as the global provider,
    so the retry handler's ``trace.get_tracer(...)`` lookups route here.
    """
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    try:
        trace.set_tracer_provider(provider)
    except Exception:
        pass
    current = trace.get_tracer_provider()
    if current is not provider:
        try:
            current.add_span_processor(SimpleSpanProcessor(exporter))
        except Exception:
            pass
    yield trace.get_tracer("test"), exporter, current


@pytest.fixture(autouse=True)
def reset_registry_and_state():
    """Each test starts with empty registry + emitter uninstalled."""
    retry_registry._reset_for_test()
    clear_attempt_counters_for_test()
    # Defensive uninstall in case a prior test left it installed.
    try:
        uninstrument_retry_emitter()
    except Exception:
        pass
    yield
    retry_registry._reset_for_test()
    clear_attempt_counters_for_test()
    try:
        uninstrument_retry_emitter()
    except Exception:
        pass


def _make_request(path: str = "/v1/chat/completions", host: str = "api.openai.com",
                  port: Optional[int] = None, scheme: str = "https",
                  model: Optional[str] = "gpt-4o-mini") -> SimpleNamespace:
    """Build a fake ``httpx.Request``-shaped object for the wrapper.

    The wrapper reads .url.path / .url.host / .url.port / .url.scheme
    plus .content (JSON-encoded body) — all we need.
    """
    import json
    body = {"model": model, "messages": [{"role": "user", "content": "hi"}]} if model else {}
    content = json.dumps(body).encode("utf-8")
    url = SimpleNamespace(path=path, host=host, port=port, scheme=scheme)
    return SimpleNamespace(url=url, content=content)


def _make_response(status_code: int = 200, request_id: str = "req-abc",
                   body: Optional[dict] = None) -> SimpleNamespace:
    """Build a fake ``httpx.Response``-shaped object.

    The retry handler's non-streaming body-parse path calls
    ``response.json()`` directly, so we expose ``json()`` as a method
    returning the supplied dict. The default body matches the OpenAI
    chat-completion schema with usage so the success path's usage
    extraction has something to read.
    """
    if body is None:
        body = {
            "id": f"chatcmpl-{request_id}",
            "model": "gpt-4o-mini-2024-07-18",
            "usage": {
                "prompt_tokens": 7,
                "completion_tokens": 3,
                "total_tokens": 10,
                "prompt_tokens_details": {
                    "cached_tokens": 5,
                    "audio_tokens": 0,
                },
            },
        }
    headers = {"x-request-id": request_id, "openai-request-id": request_id}
    return SimpleNamespace(
        status_code=status_code,
        headers=headers,
        json=lambda: body,
    )


def _retry_spans(exporter: InMemorySpanExporter):
    return [s for s in exporter.get_finished_spans()
            if s.name.startswith(f"{_FR_LLM_ATTEMPT_SPAN_NAME_PREFIX}.attempt_")]


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
# Instrumentor symmetry.
# ---------------------------------------------------------------------------

def test_install_then_uninstall_is_idempotent_and_symmetric():
    """install → installed; uninstall → not installed; double-install
    and double-uninstall are no-ops."""
    assert not _is_installed_for_test()

    instrument_retry_emitter()
    assert _is_installed_for_test()

    # Idempotent install: state stays installed; no exception.
    instrument_retry_emitter()
    assert _is_installed_for_test()

    uninstrument_retry_emitter()
    assert not _is_installed_for_test()

    # Idempotent uninstall.
    uninstrument_retry_emitter()
    assert not _is_installed_for_test()


def test_install_actually_wraps_the_openai_private_send():
    """REGRESSION GUARD: install must put a wrapt wrapper on
    ``openai._base_client.SyncHttpxClientWrapper.send``. Without this
    check the install path could silently no-op and leave retry_attempt
    emission disabled in production.
    """
    pytest.importorskip("openai")
    from openai import _base_client as base

    if not hasattr(base, "SyncHttpxClientWrapper") or not hasattr(
        base.SyncHttpxClientWrapper, "send"
    ):
        pytest.skip("openai._base_client.SyncHttpxClientWrapper.send not present")

    instrument_retry_emitter()
    try:
        send = base.SyncHttpxClientWrapper.send
        # wrapt sets __wrapped__ on the wrapped descriptor.
        assert hasattr(send, "__wrapped__"), (
            "after instrument, SyncHttpxClientWrapper.send must be a wrapt wrapper"
        )
    finally:
        uninstrument_retry_emitter()

    # After uninstall, the wrap must be gone.
    send_after = base.SyncHttpxClientWrapper.send
    assert not hasattr(send_after, "__wrapped__"), (
        "after uninstall, SyncHttpxClientWrapper.send must be the original (no __wrapped__)"
    )


# ---------------------------------------------------------------------------
# Private-symbol-missing guard.
# ---------------------------------------------------------------------------

def test_missing_private_symbol_logs_warning_and_does_not_crash(caplog):
    """If ``openai._base_client.SyncHttpxClientWrapper.send`` is missing,
    install must log a warning and skip emission for that variant —
    NOT raise. Normal openai instrumentation continues unaffected.
    """
    import opentelemetry.instrumentation.openai.retry_handler as rh

    with patch.object(rh, "_has_wrappable_symbol", return_value=False):
        with caplog.at_level("WARNING"):
            # Must not raise.
            instrument_retry_emitter()
    # State still flips to installed (a no-op install is still an
    # install — uninstall is symmetric).
    assert _is_installed_for_test()
    # Warning emitted for at least one of the two variants.
    msgs = " ".join(r.getMessage() for r in caplog.records)
    assert "missing/incompatible" in msgs
    uninstrument_retry_emitter()


def test_has_wrappable_symbol_returns_true_for_real_openai():
    """Sanity: the import-time wrappability check works against the
    real installed openai SDK. Catches the case where the private API
    is renamed between SDK versions (the regression this guard
    protects against)."""
    pytest.importorskip("openai")
    assert _has_wrappable_symbol("openai._base_client", "SyncHttpxClientWrapper", "send")
    assert _has_wrappable_symbol("openai._base_client", "AsyncHttpxClientWrapper", "send")


def test_has_wrappable_symbol_returns_false_for_unknown():
    """A bogus class name returns False (does not raise)."""
    assert not _has_wrappable_symbol("openai._base_client", "NoSuchClass", "send")
    assert not _has_wrappable_symbol("openai._base_client", "SyncHttpxClientWrapper", "no_such_method")


# ---------------------------------------------------------------------------
# Single-attempt happy path.
# ---------------------------------------------------------------------------

def test_single_attempt_emits_one_span_with_marker(fresh_tracer):
    tracer, exporter, _ = fresh_tracer

    request = _make_request(model="gpt-4o-mini", path="/v1/chat/completions")
    response = _make_response(status_code=200, request_id="req-1")

    parent = tracer.start_span("openai.chat")  # mimics outer span
    with trace.use_span(parent, end_on_exit=False):
        # _sync_send_wrapper(wrapped, instance, args, kwargs)
        result = _sync_send_wrapper(lambda *a, **kw: response, None, (request,), {})
    parent.end()

    assert result is response

    spans = exporter.get_finished_spans()
    parent_exported = next(s for s in spans if s.name == "openai.chat")
    retry_spans = _retry_spans(exporter)
    assert len(retry_spans) == 1
    rs = retry_spans[0]
    _assert_attempt_sequence(retry_spans)
    assert rs.parent.span_id == parent_exported.context.span_id, (
        "retry_attempt must be a child of the active parent"
    )
    assert rs.attributes.get("fortifyroot.span.role") == "llm_attempt"
    assert rs.attributes.get("gen_ai.system") == "openai"
    assert rs.attributes.get("gen_ai.request.model") == "gpt-4o-mini"
    assert rs.attributes.get("gen_ai.operation.name") == "chat"
    assert rs.attributes.get("http.status_code") == 200
    # Response-body-derived attrs (overrides the header-only response id
    # with the body's chatcmpl-* id; adds response.model and usage tokens).
    assert rs.attributes.get("gen_ai.response.id") == "chatcmpl-req-1"
    assert rs.attributes.get("gen_ai.response.model") == "gpt-4o-mini-2024-07-18"
    assert rs.attributes.get("gen_ai.usage.input_tokens") == 7
    assert rs.attributes.get("gen_ai.usage.output_tokens") == 3
    assert rs.attributes.get("llm.usage.total_tokens") == 10
    assert rs.attributes.get("gen_ai.usage.cache_read_input_tokens") == 5
    assert rs.attributes.get("server.address") == "api.openai.com"
    assert rs.attributes.get("server.port") == 443
    assert parent_exported.attributes.get(_FR_HAS_ATTEMPT_CHILD_KEY) is True


# ---------------------------------------------------------------------------
# Multi-attempt retry path — siblings under one parent.
# ---------------------------------------------------------------------------

def test_three_attempts_share_parent(fresh_tracer):
    """429 → 429 → 200 chain produces 3 sibling retry_attempt spans
    under one parent. RetryDetectorProc relies on this shape."""
    tracer, exporter, _ = fresh_tracer

    request = _make_request(model="gpt-4o-mini", path="/v1/chat/completions")
    resp_429 = _make_response(status_code=429, request_id="req-429a")
    resp_200 = _make_response(status_code=200, request_id="req-200")

    parent = tracer.start_span("openai.chat")
    with trace.use_span(parent, end_on_exit=False):
        # attempt 1 — 429
        _sync_send_wrapper(lambda *a, **kw: resp_429, None, (request,), {})
        # attempt 2 — 429
        _sync_send_wrapper(lambda *a, **kw: resp_429, None, (request,), {})
        # attempt 3 — 200
        _sync_send_wrapper(lambda *a, **kw: resp_200, None, (request,), {})
    parent.end()

    spans = exporter.get_finished_spans()
    parent_exported = next(s for s in spans if s.name == "openai.chat")
    retry_spans = _retry_spans(exporter)
    assert len(retry_spans) == 3, f"expected 3 retry_attempts, got {len(retry_spans)}"
    _assert_attempt_sequence(retry_spans)

    parent_ids = {s.parent.span_id for s in retry_spans}
    assert parent_ids == {parent_exported.context.span_id}, (
        f"all 3 retry_attempts must be siblings under one parent; got {parent_ids}"
    )

    from opentelemetry.trace import StatusCode
    error_count = sum(1 for s in retry_spans if s.status.status_code == StatusCode.ERROR)
    ok_count = sum(1 for s in retry_spans if s.status.status_code == StatusCode.OK)
    assert error_count == 2 and ok_count == 1, (
        f"expected 2 ERROR + 1 OK, got error={error_count}, ok={ok_count}"
    )

    for s in retry_spans:
        if s.status.status_code == StatusCode.ERROR:
            assert s.attributes.get("http.status_code") == 429
            assert s.attributes.get("error.type") == "openai.RateLimitError"


# ---------------------------------------------------------------------------
# Error attempt: exception path.
# ---------------------------------------------------------------------------

def test_exception_path_records_error_and_reraises(fresh_tracer):
    """If wrapped(send) raises, the wrapper finalises the span as
    ERROR with error.type set, then re-raises the original exception."""
    tracer, exporter, _ = fresh_tracer

    request = _make_request(model="gpt-4o-mini")

    class TimeoutLikeError(Exception):
        pass

    err = TimeoutLikeError("connect timeout")

    def raising(*a, **kw):
        raise err

    parent = tracer.start_span("openai.chat")
    with trace.use_span(parent, end_on_exit=False):
        with pytest.raises(TimeoutLikeError):
            _sync_send_wrapper(raising, None, (request,), {})
    parent.end()

    from opentelemetry.trace import StatusCode
    retry_spans = _retry_spans(exporter)
    assert len(retry_spans) == 1
    rs = retry_spans[0]
    assert rs.status.status_code == StatusCode.ERROR
    # error.type carries the namespaced exception class.
    assert "TimeoutLikeError" in (rs.attributes.get("error.type") or "")


def test_http_status_code_on_response_exception(fresh_tracer):
    """If the SDK raises something that carries .status_code (e.g.
    APIStatusError-shaped), the wrapper extracts and records it."""
    tracer, exporter, _ = fresh_tracer

    request = _make_request(model="gpt-4o-mini")

    class FakeStatusError(Exception):
        def __init__(self, status_code):
            self.status_code = status_code
            super().__init__(f"http {status_code}")

    err = FakeStatusError(503)

    parent = tracer.start_span("openai.chat")
    with trace.use_span(parent, end_on_exit=False):
        with pytest.raises(FakeStatusError):
            _sync_send_wrapper(lambda *a, **kw: (_ for _ in ()).throw(err), None, (request,), {})
    parent.end()

    retry_spans = _retry_spans(exporter)
    assert len(retry_spans) == 1
    rs = retry_spans[0]
    assert rs.attributes.get("http.status_code") == 503


# ---------------------------------------------------------------------------
# Marker timing (§4.5).
# ---------------------------------------------------------------------------

def test_marker_set_AFTER_first_attempt_not_at_parent_creation(fresh_tracer):
    tracer, exporter, _ = fresh_tracer
    request = _make_request()

    parent = tracer.start_span("openai.chat")
    # Before any retry_attempt: no marker.
    assert _FR_HAS_ATTEMPT_CHILD_KEY not in dict(parent.attributes or {})

    with trace.use_span(parent, end_on_exit=False):
        _sync_send_wrapper(lambda *a, **kw: _make_response(200), None, (request,), {})

    # After first attempt: marker present on the still-open parent.
    assert dict(parent.attributes or {}).get(_FR_HAS_ATTEMPT_CHILD_KEY) is True
    parent.end()


def test_marker_NOT_set_when_no_attempts_fire(fresh_tracer):
    tracer, exporter, _ = fresh_tracer
    parent = tracer.start_span("openai.chat")
    parent.end()
    parent_exported = next(
        s for s in exporter.get_finished_spans() if s.name == "openai.chat"
    )
    assert _FR_HAS_ATTEMPT_CHILD_KEY not in (parent_exported.attributes or {})


# ---------------------------------------------------------------------------
# No-parent guard.
# ---------------------------------------------------------------------------

def test_no_parent_does_not_emit(fresh_tracer):
    tracer, exporter, _ = fresh_tracer
    request = _make_request()
    # No active span at all.
    result = _sync_send_wrapper(lambda *a, **kw: _make_response(200), None, (request,), {})
    assert result.status_code == 200
    assert len(_retry_spans(exporter)) == 0


# ---------------------------------------------------------------------------
# §4.7 suppression — context API.
# ---------------------------------------------------------------------------

def test_context_suppression_skips_emission(fresh_tracer):
    """If SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY is True in the
    OTel context, no retry_attempt span emits — wrapped(send) still
    runs normally."""
    tracer, exporter, _ = fresh_tracer
    request = _make_request()

    parent = tracer.start_span("openai.chat")
    suppress_token = context_api.attach(
        context_api.set_value(SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY, True)
    )
    try:
        with trace.use_span(parent, end_on_exit=False):
            result = _sync_send_wrapper(lambda *a, **kw: _make_response(200), None, (request,), {})
    finally:
        context_api.detach(suppress_token)
    parent.end()

    assert result.status_code == 200
    assert len(_retry_spans(exporter)) == 0, (
        "SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY active → no retry_attempt"
    )


# ---------------------------------------------------------------------------
# §4.7 suppression — framework-attempt registry.
# ---------------------------------------------------------------------------

def test_framework_registry_suppression_skips_emission(fresh_tracer):
    """If is_framework_owned() is True (a framework wrapper has an
    in-flight attempt on this thread), the direct-SDK wrapper must
    SKIP emission. This is the §4.7 invariant that prevents double-
    emission when both framework + direct-SDK wrappers are active."""
    tracer, exporter, _ = fresh_tracer
    request = _make_request()

    parent = tracer.start_span("openai.chat")
    token = register_framework_attempt()
    assert is_framework_owned()
    try:
        with trace.use_span(parent, end_on_exit=False):
            result = _sync_send_wrapper(lambda *a, **kw: _make_response(200), None, (request,), {})
    finally:
        unregister_framework_attempt(token)
    parent.end()

    assert result.status_code == 200
    assert len(_retry_spans(exporter)) == 0, (
        "is_framework_owned() active → no retry_attempt"
    )


# ---------------------------------------------------------------------------
# Framework registry token: registered DURING our attempt, released AFTER.
# ---------------------------------------------------------------------------

def test_direct_sdk_wrapper_does_not_self_register_in_framework_registry(fresh_tracer):
    """REGRESSION GUARD (review-driven fix 2026-05-13): direct-SDK
    wrappers MUST NOT register tokens in the §4.7.1 framework registry.
    The registry's contract reserves registration for FRAMEWORK wrappers
    (LiteLLM / LangChain / LlamaIndex); direct-SDK wrappers only CONSULT
    via ``is_framework_owned()``.

    If this contract is violated, two concurrent asyncio tasks on the
    same OS thread will see each other's token via the thread-keyed
    registry, and the second task's emission gets suppressed —
    silently dropping a retry_attempt span. See
    ``test_two_concurrent_async_sends_each_emit_a_retry_attempt`` for
    the end-to-end repro."""
    tracer, _, _ = fresh_tracer
    request = _make_request()

    seen_during: list[bool] = []

    def wrapped(*a, **kw):
        seen_during.append(is_framework_owned())
        return _make_response(200)

    parent = tracer.start_span("openai.chat")
    with trace.use_span(parent, end_on_exit=False):
        assert not is_framework_owned()
        _sync_send_wrapper(wrapped, None, (request,), {})
        assert not is_framework_owned(), "after attempt → still not registered"
    parent.end()

    assert seen_during == [False], (
        "the direct-SDK wrapper MUST NOT self-register a framework "
        "token; is_framework_owned() must remain False during the wrapped "
        "send. Got True → wrapper is incorrectly registering."
    )


def test_two_concurrent_async_sends_each_emit_a_retry_attempt(fresh_tracer):
    """REGRESSION GUARD (review-driven fix 2026-05-13): two asyncio
    tasks sharing one OS thread MUST each emit their own retry_attempt
    span. Pre-fix, the first task's self-registered framework token
    would suppress the second task via the thread-keyed registry —
    silently dropping one of the spans."""
    tracer, exporter, _ = fresh_tracer

    request = _make_request(path="/v1/chat/completions", model="gpt-4o-mini")
    parent = tracer.start_span("openai.chat")

    async def one_send():
        # Yield to the event loop after entering the wrap, so the two
        # tasks' wrap bodies interleave (otherwise they could run
        # sequentially and the bug would be hidden).
        async def wrapped(*a, **kw):
            await asyncio.sleep(0)
            return _make_response(200)
        with trace.use_span(parent, end_on_exit=False):
            return await _async_send_wrapper(wrapped, None, (request,), {})

    async def run():
        return await asyncio.gather(one_send(), one_send())

    loop = asyncio.new_event_loop()
    try:
        results = loop.run_until_complete(run())
    finally:
        loop.close()
    parent.end()

    assert all(r.status_code == 200 for r in results)
    retry_spans = _retry_spans(exporter)
    assert len(retry_spans) == 2, (
        f"expected 2 retry_attempt spans (one per concurrent task); got {len(retry_spans)}. "
        f"This is the §4.7 self-registration regression — the second task is being "
        f"suppressed by the first task's lingering framework token."
    )
    _assert_attempt_sequence(retry_spans)


# ---------------------------------------------------------------------------
# §4.4.1 endpoint allow-list.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", [
    "/v1/models",
    "/v1/files",
    "/v1/files/file-123",
    "/v1/fine_tuning/jobs",
    "/v1/threads/thread-abc",
    "/",
    "",
])
def test_non_llm_endpoints_do_not_emit(fresh_tracer, path):
    """Non-LLM SDK traffic (model listing, file ops, auth refresh, etc.)
    MUST NOT emit retry_attempt spans — per §4.4.1 allow-list."""
    tracer, exporter, _ = fresh_tracer
    request = _make_request(path=path, model=None)
    parent = tracer.start_span("openai.chat")
    with trace.use_span(parent, end_on_exit=False):
        _sync_send_wrapper(lambda *a, **kw: _make_response(200), None, (request,), {})
    parent.end()
    assert len(_retry_spans(exporter)) == 0, (
        f"non-LLM endpoint {path!r} must not emit retry_attempt"
    )


@pytest.mark.parametrize("path,expected_op", [
    ("/v1/chat/completions", "chat"),
    ("/v1/completions", "text_completion"),
    ("/v1/embeddings", "embeddings"),
    ("/v1/responses", "chat"),
    ("/v1/messages", "chat"),  # Azure-OpenAI messages parity layer
])
def test_llm_endpoints_emit_with_correct_operation(fresh_tracer, path, expected_op):
    tracer, exporter, _ = fresh_tracer
    request = _make_request(path=path, model="gpt-4o-mini")
    parent = tracer.start_span("openai.chat")
    with trace.use_span(parent, end_on_exit=False):
        _sync_send_wrapper(lambda *a, **kw: _make_response(200), None, (request,), {})
    parent.end()
    retry_spans = _retry_spans(exporter)
    assert len(retry_spans) == 1
    assert retry_spans[0].attributes.get("gen_ai.operation.name") == expected_op


# ---------------------------------------------------------------------------
# §4.5-driven usage-extraction policy (review-driven fix 2026-05-13).
# ---------------------------------------------------------------------------

def test_non_streaming_response_body_populates_usage_id_and_model(fresh_tracer):
    """REGRESSION GUARD: backend dedup makes the retry_attempt span the
    canonical LLMUsageEvent even for single-attempt calls. If the wrap
    omits usage tokens, the canonical event has zero cost. For
    non-streaming responses we MUST parse the body and copy usage onto
    the retry_attempt span. See review finding C2 in the 2026-05-13
    addendum.
    """
    tracer, exporter, _ = fresh_tracer
    request = _make_request(model="gpt-4o-mini")
    response = _make_response(
        status_code=200,
        request_id="r-1",
        body={
            "id": "chatcmpl-XYZ",
            "model": "gpt-4o-mini-2024-07-18",
            "usage": {"prompt_tokens": 42, "completion_tokens": 11, "total_tokens": 53},
        },
    )
    parent = tracer.start_span("openai.chat")
    with trace.use_span(parent, end_on_exit=False):
        _sync_send_wrapper(lambda *a, **kw: response, None, (request,), {})
    parent.end()

    retry_spans = _retry_spans(exporter)
    assert len(retry_spans) == 1
    rs = retry_spans[0]
    assert rs.attributes.get("gen_ai.usage.input_tokens") == 42
    assert rs.attributes.get("gen_ai.usage.output_tokens") == 11
    assert rs.attributes.get("gen_ai.response.id") == "chatcmpl-XYZ"
    assert rs.attributes.get("gen_ai.response.model") == "gpt-4o-mini-2024-07-18"


def test_streaming_request_skips_emission_entirely(fresh_tracer):
    """ST-10.4 (review-driven 2026-05-17): when ``stream=True`` is
    passed to send, the wrap SKIPS retry_attempt emission entirely —
    no span is created, and ``response.json()`` is never called.

    Rationale: streaming retry_attempts cannot carry token usage at
    attempt-end (SSE stream consumption would break the SDK), but
    backend §4.5 dedup would promote them to canonical
    LLMUsageEvents → zero-token events → fr-system-tests
    ``prompt_tokens > 0`` assertions fail. The parent
    ``openai.chat`` span (which gets full usage from ``ChatStream``'s
    stream-completion callback) stays canonical. Streaming
    retry-loop detection is the deferred follow-up
    ``ST-10.4-FOLLOWUP-streaming-usage``.
    """
    tracer, exporter, _ = fresh_tracer
    request = _make_request(model="gpt-4o-mini")

    json_called = {"n": 0}

    def explode():
        json_called["n"] += 1
        raise RuntimeError("body must not be read on streaming response")

    headers = {"x-request-id": "req-stream-1"}
    response = SimpleNamespace(status_code=200, headers=headers, json=explode)

    parent = tracer.start_span("openai.chat")
    with trace.use_span(parent, end_on_exit=False):
        result = _sync_send_wrapper(
            lambda *a, **kw: response,
            None,
            (request,),
            {"stream": True},
        )
    parent.end()

    # wrap returns the response unchanged.
    assert result is response
    # response.json() never invoked.
    assert json_called["n"] == 0
    # No retry_attempt span emitted at all.
    retry_spans = _retry_spans(exporter)
    assert len(retry_spans) == 0, (
        f"streaming attempts MUST skip retry_attempt emission entirely; "
        f"got {len(retry_spans)} span(s). The parent span stays the "
        f"canonical LLM event."
    )
    # Parent MUST NOT receive the has_attempt_child marker
    # either (no child was emitted).
    parent_exported = next(
        s for s in exporter.get_finished_spans() if s.name == "openai.chat"
    )
    assert _FR_HAS_ATTEMPT_CHILD_KEY not in (parent_exported.attributes or {})


def test_non_2xx_response_with_usage_in_body_populates_usage_tokens(fresh_tracer):
    """REGRESSION GUARD (review-driven follow-up 2026-05-13): per
    RETRY_LOOP.md §4.4 token-usage rule (around line 164), wrappers
    MUST extract usage from the response body whenever it's present,
    REGARDLESS of whether the attempt succeeded. Some failures (e.g.
    context-length-exceeded errors) consume tokens and the provider
    returns usage in the error body. The retry_attempt span MUST carry
    those tokens so backend §4.5 dedup doesn't promote a zero-cost
    canonical event.
    """
    tracer, exporter, _ = fresh_tracer
    request = _make_request(model="gpt-4o-mini")
    # Simulated context-length-exceeded shape: 400 + usage in body.
    response = _make_response(
        status_code=400,
        request_id="r-ctx-exceeded",
        body={
            "error": {
                "type": "invalid_request_error",
                "code": "context_length_exceeded",
                "message": "This model's max context...",
            },
            "id": "chatcmpl-ERR",
            "model": "gpt-4o-mini-2024-07-18",
            "usage": {"prompt_tokens": 128000, "completion_tokens": 0, "total_tokens": 128000},
        },
    )
    parent = tracer.start_span("openai.chat")
    with trace.use_span(parent, end_on_exit=False):
        _sync_send_wrapper(lambda *a, **kw: response, None, (request,), {})
    parent.end()

    from opentelemetry.trace import StatusCode
    retry_spans = _retry_spans(exporter)
    assert len(retry_spans) == 1
    rs = retry_spans[0]
    assert rs.status.status_code == StatusCode.ERROR
    assert rs.attributes.get("http.status_code") == 400
    assert rs.attributes.get("error.type") == "openai.HTTPStatusError"
    # Usage was in the error body → must be on the span (the load-bearing assertion).
    assert rs.attributes.get("gen_ai.usage.input_tokens") == 128000
    assert rs.attributes.get("gen_ai.usage.output_tokens") == 0
    assert rs.attributes.get("gen_ai.response.id") == "chatcmpl-ERR"


def test_non_2xx_response_without_usage_in_body_omits_usage_tokens(fresh_tracer):
    """Counterpart to the above: when the error body has no ``usage``
    field, the wrap must NOT set usage attrs (per §4.4 rule: omit when
    unknown; setting 0 risks under-counting cost on other paths)."""
    tracer, exporter, _ = fresh_tracer
    request = _make_request(model="gpt-4o-mini")
    response = _make_response(
        status_code=500,
        request_id="r-server-err",
        body={"error": {"type": "internal_server_error", "message": "boom"}},
    )
    parent = tracer.start_span("openai.chat")
    with trace.use_span(parent, end_on_exit=False):
        _sync_send_wrapper(lambda *a, **kw: response, None, (request,), {})
    parent.end()

    from opentelemetry.trace import StatusCode
    retry_spans = _retry_spans(exporter)
    assert len(retry_spans) == 1
    rs = retry_spans[0]
    assert rs.status.status_code == StatusCode.ERROR
    assert rs.attributes.get("http.status_code") == 500
    assert rs.attributes.get("error.type") == "openai.InternalServerError"
    assert rs.attributes.get("gen_ai.usage.input_tokens") is None
    assert rs.attributes.get("gen_ai.usage.output_tokens") is None


def test_non_streaming_response_with_malformed_body_does_not_crash(fresh_tracer):
    """If ``response.json()`` raises (non-JSON body, e.g. error HTML),
    the wrap must record the basic status attrs without raising and
    without leaving the span in an unended state."""
    tracer, exporter, _ = fresh_tracer
    request = _make_request(model="gpt-4o-mini")

    def explode():
        raise ValueError("malformed JSON")

    headers = {"x-request-id": "req-bad-body"}
    response = SimpleNamespace(status_code=200, headers=headers, json=explode)

    parent = tracer.start_span("openai.chat")
    with trace.use_span(parent, end_on_exit=False):
        # MUST NOT raise.
        _sync_send_wrapper(lambda *a, **kw: response, None, (request,), {})
    parent.end()

    retry_spans = _retry_spans(exporter)
    assert len(retry_spans) == 1
    from opentelemetry.trace import StatusCode
    assert retry_spans[0].status.status_code == StatusCode.OK
    assert retry_spans[0].attributes.get("http.status_code") == 200
    # Header-based response id survives even when body parsing fails.
    assert retry_spans[0].attributes.get("gen_ai.response.id") == "req-bad-body"


# ---------------------------------------------------------------------------
# Async path mirrors sync.
# ---------------------------------------------------------------------------

def test_external_suppression_with_no_override_skips_emission(fresh_tracer):
    """REGRESSION GUARD (review-driven 2026-05-16, Issue 3B counter-proof):
    when SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY is set in the OTel
    context WITHOUT the OpenAI override key, the retry handler MUST
    suppress emission. This is the user-explicit "disable LLM
    instrumentation for this scope" path, distinct from the
    openai-wrapper-internal use of the same key."""
    tracer, exporter, _ = fresh_tracer
    request = _make_request()
    parent = tracer.start_span("some.workflow")
    suppress_token = context_api.attach(
        context_api.set_value(SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY, True)
    )
    try:
        with trace.use_span(parent, end_on_exit=False):
            _sync_send_wrapper(lambda *a, **kw: _make_response(200), None, (request,), {})
    finally:
        context_api.detach(suppress_token)
    parent.end()

    assert len(_retry_spans(exporter)) == 0, (
        "external SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY without the "
        "openai override key MUST skip emission"
    )


def test_openai_wrapper_override_key_unblocks_emission_under_suppression(fresh_tracer):
    """REGRESSION GUARD (review-driven 2026-05-16, Issue 3B):
    the openai chat_wrapper sets SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY
    around its wrapped SDK call (to protect against OTHER LLM
    instrumentors double-counting). To preserve retry_attempt emission
    under that scope, the chat_wrapper ALSO sets
    OPENAI_DIRECT_RETRY_PARENT_ACTIVE_KEY in the same context. This
    test mimics that contract: when BOTH keys are set, emission MUST
    proceed (the suppression is openai-wrapper-internal, not external)."""
    tracer, exporter, _ = fresh_tracer
    request = _make_request()
    parent = tracer.start_span("openai.chat")
    ctx = context_api.set_value(SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY, True)
    ctx = context_api.set_value(OPENAI_DIRECT_RETRY_PARENT_ACTIVE_KEY, True, ctx)
    token = context_api.attach(ctx)
    try:
        with trace.use_span(parent, end_on_exit=False):
            _sync_send_wrapper(lambda *a, **kw: _make_response(200), None, (request,), {})
    finally:
        context_api.detach(token)
    parent.end()

    retry_spans = _retry_spans(exporter)
    assert len(retry_spans) == 1, (
        "retry_attempt MUST emit when the openai override key is set "
        "alongside SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY (this is the "
        "chat_wrapper's own internal HTTP-send scope)"
    )


def test_tracer_provider_plumbed_through_instrument_retry_emitter():
    """REGRESSION GUARD (review-driven 2026-05-16, Issue 3A):
    ``instrument_retry_emitter(tracer_provider=provider)`` MUST cause
    the wrapper to emit retry_attempt spans through that explicit
    provider, not the global default. Without this plumbing, a
    consumer who passes a tracer_provider to
    ``OpenAIInstrumentor.instrument(...)`` would get the parent openai
    span on their provider but the retry_attempt sent into the global
    (no-op when global is unset)."""
    from opentelemetry.instrumentation.fortifyroot import retry_registry as _rr
    _rr._reset_for_test()
    try:
        uninstrument_retry_emitter()
    except Exception:
        pass

    # Build a NON-global provider with its own exporter.
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    instrument_retry_emitter(tracer_provider=provider)
    try:
        request = _make_request(path="/v1/chat/completions", model="gpt-4o-mini")
        # Create the parent via the SAME local provider so both the
        # parent and the retry_attempt share one trace tree on `provider`.
        parent_tracer = provider.get_tracer("test")
        parent = parent_tracer.start_span("openai.chat")
        with trace.use_span(parent, end_on_exit=False):
            _sync_send_wrapper(lambda *a, **kw: _make_response(200), None, (request,), {})
        parent.end()
    finally:
        uninstrument_retry_emitter()

    spans = exporter.get_finished_spans()
    retry_spans = [s for s in spans if s.name.startswith(f"{_FR_LLM_ATTEMPT_SPAN_NAME_PREFIX}.attempt_")]
    assert len(retry_spans) == 1, (
        f"retry_attempt MUST be exported via the explicit tracer_provider "
        f"passed to instrument_retry_emitter; saw {len(retry_spans)} in this exporter "
        f"(all span names: {[s.name for s in spans]})"
    )


def test_async_send_wrapper_emits(fresh_tracer):
    tracer, exporter, _ = fresh_tracer
    request = _make_request()

    async def wrapped(*a, **kw):
        return _make_response(200)

    async def run():
        parent = tracer.start_span("openai.chat")
        try:
            with trace.use_span(parent, end_on_exit=False):
                return await _async_send_wrapper(wrapped, None, (request,), {})
        finally:
            parent.end()

    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(run())
    finally:
        loop.close()
    assert result.status_code == 200
    assert len(_retry_spans(exporter)) == 1
    assert _retry_spans(exporter)[0].attributes.get("gen_ai.system") == "openai"

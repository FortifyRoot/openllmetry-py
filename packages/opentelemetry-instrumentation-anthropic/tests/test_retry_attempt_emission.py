"""Tests for ST-10.4 Anthropic direct-SDK retry-attempt emission.

Covers (per RETRY_LOOP.md §4.4 Anthropic row + §4.7 suppression):
  - Instrumentor symmetry: install/uninstall flips state cleanly and is
    idempotent.
  - Single-attempt happy path: ONE retry_attempt span under the active
    parent; parent gets has_retry_attempt_child=true; span carries
    role / gen_ai.system=anthropic / gen_ai.request.model / http.status_code.
  - Multi-attempt retry path: N siblings under one parent.
  - Error-attempt: status=ERROR, http.status_code, error.type.
  - No-parent guard.
  - §4.7 suppression — context-API and framework-registry both skip.
  - §4.4.1 endpoint allow-list: non-LLM SDK traffic does NOT emit.
  - Private-symbol missing guard: warning + skip, no crash.

UNIT tests — drive ``_sync_send_wrapper`` / ``_async_send_wrapper``
directly with fake httpx Request / Response objects.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Optional
from unittest.mock import patch

import pytest
from opentelemetry import context as context_api
from opentelemetry import trace
from opentelemetry.instrumentation.anthropic.retry_handler import (
    _FR_HAS_RETRY_ATTEMPT_CHILD_KEY,
    _FR_RETRY_ATTEMPT_SPAN_NAME,
    _async_send_wrapper,
    _has_wrappable_symbol,
    _is_installed_for_test,
    _sync_send_wrapper,
    instrument_retry_emitter,
    uninstrument_retry_emitter,
)
from opentelemetry.instrumentation.fortifyroot import (
    is_framework_owned,
    register_framework_attempt,
    retry_registry,
    unregister_framework_attempt,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.semconv_ai import SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY


@pytest.fixture
def fresh_tracer():
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
def reset_state():
    retry_registry._reset_for_test()
    try:
        uninstrument_retry_emitter()
    except Exception:
        pass
    yield
    retry_registry._reset_for_test()
    try:
        uninstrument_retry_emitter()
    except Exception:
        pass


def _make_request(path: str = "/v1/messages", host: str = "api.anthropic.com",
                  port: Optional[int] = None, scheme: str = "https",
                  model: Optional[str] = "claude-haiku-4-5") -> SimpleNamespace:
    import json
    body = {"model": model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 10} if model else {}
    content = json.dumps(body).encode("utf-8")
    url = SimpleNamespace(path=path, host=host, port=port, scheme=scheme)
    return SimpleNamespace(url=url, content=content)


def _make_response(status_code: int = 200, request_id: str = "req-abc",
                   body: Optional[dict] = None) -> SimpleNamespace:
    """Fake ``httpx.Response``. Non-streaming retry_attempt path now
    parses the body for usage / response.id / response.model — see the
    2026-05-13 review-driven C2 fix.
    """
    if body is None:
        body = {
            "id": f"msg_{request_id}",
            "model": "claude-haiku-4-5-20251001",
            "usage": {"input_tokens": 9, "output_tokens": 4},
        }
    headers = {"request-id": request_id, "x-request-id": request_id}
    return SimpleNamespace(
        status_code=status_code,
        headers=headers,
        json=lambda: body,
    )


def _retry_spans(exporter: InMemorySpanExporter):
    return [s for s in exporter.get_finished_spans()
            if s.name == _FR_RETRY_ATTEMPT_SPAN_NAME]


# ---------------------------------------------------------------------------
# Instrumentor symmetry.
# ---------------------------------------------------------------------------

def test_install_then_uninstall_is_idempotent_and_symmetric():
    assert not _is_installed_for_test()

    instrument_retry_emitter()
    assert _is_installed_for_test()

    instrument_retry_emitter()  # idempotent
    assert _is_installed_for_test()

    uninstrument_retry_emitter()
    assert not _is_installed_for_test()

    uninstrument_retry_emitter()  # idempotent
    assert not _is_installed_for_test()


def test_install_actually_wraps_the_anthropic_private_send():
    """REGRESSION GUARD: install must wrap
    ``anthropic._base_client.SyncHttpxClientWrapper.send``. Without this
    check the install path could silently no-op."""
    pytest.importorskip("anthropic")
    from anthropic import _base_client as base

    if not hasattr(base, "SyncHttpxClientWrapper") or not hasattr(
        base.SyncHttpxClientWrapper, "send"
    ):
        pytest.skip("anthropic._base_client.SyncHttpxClientWrapper.send not present")

    instrument_retry_emitter()
    try:
        send = base.SyncHttpxClientWrapper.send
        assert hasattr(send, "__wrapped__"), (
            "after instrument, SyncHttpxClientWrapper.send must be a wrapt wrapper"
        )
    finally:
        uninstrument_retry_emitter()

    send_after = base.SyncHttpxClientWrapper.send
    assert not hasattr(send_after, "__wrapped__")


# ---------------------------------------------------------------------------
# Private-symbol guard.
# ---------------------------------------------------------------------------

def test_missing_private_symbol_logs_warning_and_does_not_crash(caplog):
    import opentelemetry.instrumentation.anthropic.retry_handler as rh

    with patch.object(rh, "_has_wrappable_symbol", return_value=False):
        with caplog.at_level("WARNING"):
            instrument_retry_emitter()  # must not raise
    assert _is_installed_for_test()
    msgs = " ".join(r.getMessage() for r in caplog.records)
    assert "missing/incompatible" in msgs
    uninstrument_retry_emitter()


def test_has_wrappable_symbol_returns_true_for_real_anthropic():
    pytest.importorskip("anthropic")
    assert _has_wrappable_symbol("anthropic._base_client", "SyncHttpxClientWrapper", "send")
    assert _has_wrappable_symbol("anthropic._base_client", "AsyncHttpxClientWrapper", "send")


def test_has_wrappable_symbol_returns_false_for_unknown():
    assert not _has_wrappable_symbol("anthropic._base_client", "NoSuchClass", "send")
    assert not _has_wrappable_symbol("anthropic._base_client", "SyncHttpxClientWrapper", "no_such_method")


# ---------------------------------------------------------------------------
# Single-attempt happy path.
# ---------------------------------------------------------------------------

def test_single_attempt_emits_one_span_with_marker(fresh_tracer):
    tracer, exporter, _ = fresh_tracer
    request = _make_request(model="claude-haiku-4-5", path="/v1/messages")
    response = _make_response(status_code=200, request_id="req-1")

    parent = tracer.start_span("anthropic.chat")
    with trace.use_span(parent, end_on_exit=False):
        result = _sync_send_wrapper(lambda *a, **kw: response, None, (request,), {})
    parent.end()

    assert result is response

    spans = exporter.get_finished_spans()
    parent_exported = next(s for s in spans if s.name == "anthropic.chat")
    retry_spans = _retry_spans(exporter)
    assert len(retry_spans) == 1
    rs = retry_spans[0]
    assert rs.parent.span_id == parent_exported.context.span_id
    assert rs.attributes.get("fortifyroot.span.role") == "retry_attempt"
    assert rs.attributes.get("gen_ai.system") == "Anthropic"
    assert rs.attributes.get("gen_ai.request.model") == "claude-haiku-4-5"
    assert rs.attributes.get("gen_ai.operation.name") == "chat"
    assert rs.attributes.get("http.status_code") == 200
    # Response-body-derived attrs (replaces header-only id with the body's msg_*).
    assert rs.attributes.get("gen_ai.response.id") == "msg_req-1"
    assert rs.attributes.get("gen_ai.response.model") == "claude-haiku-4-5-20251001"
    assert rs.attributes.get("gen_ai.usage.input_tokens") == 9
    assert rs.attributes.get("gen_ai.usage.output_tokens") == 4
    assert rs.attributes.get("server.address") == "api.anthropic.com"
    assert rs.attributes.get("server.port") == 443
    assert parent_exported.attributes.get(_FR_HAS_RETRY_ATTEMPT_CHILD_KEY) is True


# ---------------------------------------------------------------------------
# Multi-attempt retry path.
# ---------------------------------------------------------------------------

def test_three_attempts_share_parent(fresh_tracer):
    tracer, exporter, _ = fresh_tracer
    request = _make_request(model="claude-haiku-4-5")
    resp_429 = _make_response(status_code=429)
    resp_200 = _make_response(status_code=200, request_id="r-ok")

    parent = tracer.start_span("anthropic.chat")
    with trace.use_span(parent, end_on_exit=False):
        _sync_send_wrapper(lambda *a, **kw: resp_429, None, (request,), {})
        _sync_send_wrapper(lambda *a, **kw: resp_429, None, (request,), {})
        _sync_send_wrapper(lambda *a, **kw: resp_200, None, (request,), {})
    parent.end()

    spans = exporter.get_finished_spans()
    parent_exported = next(s for s in spans if s.name == "anthropic.chat")
    retry_spans = _retry_spans(exporter)
    assert len(retry_spans) == 3

    parent_ids = {s.parent.span_id for s in retry_spans}
    assert parent_ids == {parent_exported.context.span_id}

    from opentelemetry.trace import StatusCode
    error_count = sum(1 for s in retry_spans if s.status.status_code == StatusCode.ERROR)
    ok_count = sum(1 for s in retry_spans if s.status.status_code == StatusCode.OK)
    assert error_count == 2 and ok_count == 1

    for s in retry_spans:
        if s.status.status_code == StatusCode.ERROR:
            assert s.attributes.get("http.status_code") == 429
            assert s.attributes.get("error.type") == "anthropic.RateLimitError"


# ---------------------------------------------------------------------------
# Error / exception path.
# ---------------------------------------------------------------------------

def test_exception_path_records_error_and_reraises(fresh_tracer):
    tracer, exporter, _ = fresh_tracer
    request = _make_request(model="claude-haiku-4-5")

    class ConnectError(Exception):
        pass

    err = ConnectError("connect failed")

    parent = tracer.start_span("anthropic.chat")
    with trace.use_span(parent, end_on_exit=False):
        with pytest.raises(ConnectError):
            _sync_send_wrapper(lambda *a, **kw: (_ for _ in ()).throw(err), None, (request,), {})
    parent.end()

    from opentelemetry.trace import StatusCode
    retry_spans = _retry_spans(exporter)
    assert len(retry_spans) == 1
    rs = retry_spans[0]
    assert rs.status.status_code == StatusCode.ERROR
    assert "ConnectError" in (rs.attributes.get("error.type") or "")


# ---------------------------------------------------------------------------
# Marker timing (§4.5).
# ---------------------------------------------------------------------------

def test_marker_set_AFTER_first_attempt_not_at_parent_creation(fresh_tracer):
    tracer, exporter, _ = fresh_tracer
    request = _make_request()
    parent = tracer.start_span("anthropic.chat")
    assert _FR_HAS_RETRY_ATTEMPT_CHILD_KEY not in dict(parent.attributes or {})
    with trace.use_span(parent, end_on_exit=False):
        _sync_send_wrapper(lambda *a, **kw: _make_response(200), None, (request,), {})
    assert dict(parent.attributes or {}).get(_FR_HAS_RETRY_ATTEMPT_CHILD_KEY) is True
    parent.end()


def test_marker_NOT_set_when_no_attempts_fire(fresh_tracer):
    tracer, exporter, _ = fresh_tracer
    parent = tracer.start_span("anthropic.chat")
    parent.end()
    parent_exported = next(
        s for s in exporter.get_finished_spans() if s.name == "anthropic.chat"
    )
    assert _FR_HAS_RETRY_ATTEMPT_CHILD_KEY not in (parent_exported.attributes or {})


# ---------------------------------------------------------------------------
# No-parent guard.
# ---------------------------------------------------------------------------

def test_no_parent_does_not_emit(fresh_tracer):
    tracer, exporter, _ = fresh_tracer
    request = _make_request()
    result = _sync_send_wrapper(lambda *a, **kw: _make_response(200), None, (request,), {})
    assert result.status_code == 200
    assert len(_retry_spans(exporter)) == 0


# ---------------------------------------------------------------------------
# §4.7 suppression.
# ---------------------------------------------------------------------------

def test_context_suppression_skips_emission(fresh_tracer):
    tracer, exporter, _ = fresh_tracer
    request = _make_request()
    parent = tracer.start_span("anthropic.chat")
    token = context_api.attach(
        context_api.set_value(SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY, True)
    )
    try:
        with trace.use_span(parent, end_on_exit=False):
            result = _sync_send_wrapper(lambda *a, **kw: _make_response(200), None, (request,), {})
    finally:
        context_api.detach(token)
    parent.end()
    assert result.status_code == 200
    assert len(_retry_spans(exporter)) == 0


def test_framework_registry_suppression_skips_emission(fresh_tracer):
    tracer, exporter, _ = fresh_tracer
    request = _make_request()
    parent = tracer.start_span("anthropic.chat")
    tok = register_framework_attempt()
    assert is_framework_owned()
    try:
        with trace.use_span(parent, end_on_exit=False):
            result = _sync_send_wrapper(lambda *a, **kw: _make_response(200), None, (request,), {})
    finally:
        unregister_framework_attempt(tok)
    parent.end()
    assert result.status_code == 200
    assert len(_retry_spans(exporter)) == 0


def test_direct_sdk_wrapper_does_not_self_register_in_framework_registry(fresh_tracer):
    """REGRESSION GUARD (review-driven fix 2026-05-13): direct-SDK
    wrappers MUST NOT register tokens in the §4.7.1 framework registry.
    Self-registration causes false suppression of concurrent direct-SDK
    calls sharing the same thread (asyncio repro in
    ``test_two_concurrent_async_sends_each_emit_a_retry_attempt``)."""
    tracer, _, _ = fresh_tracer
    request = _make_request()

    seen_during: list[bool] = []

    def wrapped(*a, **kw):
        seen_during.append(is_framework_owned())
        return _make_response(200)

    parent = tracer.start_span("anthropic.chat")
    with trace.use_span(parent, end_on_exit=False):
        assert not is_framework_owned()
        _sync_send_wrapper(wrapped, None, (request,), {})
        assert not is_framework_owned()
    parent.end()
    assert seen_during == [False], (
        "direct-SDK wrapper must NOT self-register; got is_framework_owned() == True "
        "during the wrapped send, which would falsely suppress concurrent calls"
    )


def test_two_concurrent_async_sends_each_emit_a_retry_attempt(fresh_tracer):
    """REGRESSION GUARD (review-driven fix 2026-05-13): two asyncio
    tasks sharing one OS thread MUST each emit their own retry_attempt."""
    tracer, exporter, _ = fresh_tracer

    request = _make_request(path="/v1/messages", model="claude-haiku-4-5")
    parent = tracer.start_span("anthropic.chat")

    async def one_send():
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
        f"expected 2 retry_attempt spans, got {len(retry_spans)} — async self-suppression regression"
    )


# ---------------------------------------------------------------------------
# §4.4.1 endpoint allow-list.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", [
    "/v1/models",
    "/v1/organizations/usage",
    "/",
    "",
])
def test_non_llm_endpoints_do_not_emit(fresh_tracer, path):
    tracer, exporter, _ = fresh_tracer
    request = _make_request(path=path, model=None)
    parent = tracer.start_span("anthropic.chat")
    with trace.use_span(parent, end_on_exit=False):
        _sync_send_wrapper(lambda *a, **kw: _make_response(200), None, (request,), {})
    parent.end()
    assert len(_retry_spans(exporter)) == 0, f"non-LLM endpoint {path!r} must not emit"


@pytest.mark.parametrize("path,expected_op", [
    ("/v1/messages", "chat"),
    ("/v1/complete", "text_completion"),
])
def test_llm_endpoints_emit_with_correct_operation(fresh_tracer, path, expected_op):
    tracer, exporter, _ = fresh_tracer
    request = _make_request(path=path, model="claude-haiku-4-5")
    parent = tracer.start_span("anthropic.chat")
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
    """Backend dedup makes the retry_attempt span the canonical
    LLMUsageEvent (see ``proc_llm_extractor.go``). Anthropic non-streaming
    responses MUST carry usage on the retry_attempt span."""
    tracer, exporter, _ = fresh_tracer
    request = _make_request(model="claude-haiku-4-5")
    response = _make_response(
        status_code=200,
        request_id="r-1",
        body={
            "id": "msg_01ABC",
            "model": "claude-haiku-4-5-20251001",
            "usage": {"input_tokens": 41, "output_tokens": 17},
        },
    )
    parent = tracer.start_span("anthropic.chat")
    with trace.use_span(parent, end_on_exit=False):
        _sync_send_wrapper(lambda *a, **kw: response, None, (request,), {})
    parent.end()

    retry_spans = _retry_spans(exporter)
    assert len(retry_spans) == 1
    rs = retry_spans[0]
    assert rs.attributes.get("gen_ai.usage.input_tokens") == 41
    assert rs.attributes.get("gen_ai.usage.output_tokens") == 17
    assert rs.attributes.get("gen_ai.response.id") == "msg_01ABC"
    assert rs.attributes.get("gen_ai.response.model") == "claude-haiku-4-5-20251001"


def test_streaming_request_skips_emission_entirely(fresh_tracer):
    """ST-10.4 (review-driven 2026-05-17): when ``stream=True`` is
    passed to send, the wrap SKIPS retry_attempt emission entirely.
    See the openai analog for full rationale. The parent
    ``anthropic.chat`` span stays the canonical LLMUsageEvent.
    """
    tracer, exporter, _ = fresh_tracer
    request = _make_request(model="claude-haiku-4-5")

    json_called = {"n": 0}

    def explode():
        json_called["n"] += 1
        raise RuntimeError("body must not be read on streaming response")

    headers = {"request-id": "req-stream-1", "x-request-id": "req-stream-1"}
    response = SimpleNamespace(status_code=200, headers=headers, json=explode)

    parent = tracer.start_span("anthropic.chat")
    with trace.use_span(parent, end_on_exit=False):
        result = _sync_send_wrapper(
            lambda *a, **kw: response,
            None,
            (request,),
            {"stream": True},
        )
    parent.end()

    assert result is response
    assert json_called["n"] == 0
    retry_spans = _retry_spans(exporter)
    assert len(retry_spans) == 0, (
        f"streaming attempts MUST skip retry_attempt emission entirely; "
        f"got {len(retry_spans)} span(s)."
    )
    parent_exported = next(
        s for s in exporter.get_finished_spans() if s.name == "anthropic.chat"
    )
    assert _FR_HAS_RETRY_ATTEMPT_CHILD_KEY not in (parent_exported.attributes or {})


def test_non_2xx_response_with_usage_in_body_populates_usage_tokens(fresh_tracer):
    """REGRESSION GUARD (review-driven follow-up 2026-05-13): per
    RETRY_LOOP.md §4.4 token-usage rule, wrappers MUST extract usage
    from the response body whenever present, regardless of success.
    Some failures consume tokens and the provider returns usage in
    the error body."""
    tracer, exporter, _ = fresh_tracer
    request = _make_request(model="claude-haiku-4-5")
    # Anthropic-shaped 400 with usage (hypothetical context-length scenario).
    response = _make_response(
        status_code=400,
        request_id="r-ctx-exceeded",
        body={
            "type": "error",
            "error": {"type": "invalid_request_error", "message": "context too long"},
            "id": "msg_ERR",
            "model": "claude-haiku-4-5-20251001",
            "usage": {"input_tokens": 200000, "output_tokens": 0},
        },
    )
    parent = tracer.start_span("anthropic.chat")
    with trace.use_span(parent, end_on_exit=False):
        _sync_send_wrapper(lambda *a, **kw: response, None, (request,), {})
    parent.end()

    from opentelemetry.trace import StatusCode
    retry_spans = _retry_spans(exporter)
    assert len(retry_spans) == 1
    rs = retry_spans[0]
    assert rs.status.status_code == StatusCode.ERROR
    assert rs.attributes.get("http.status_code") == 400
    assert rs.attributes.get("error.type") == "anthropic.APIStatusError"
    assert rs.attributes.get("gen_ai.usage.input_tokens") == 200000
    assert rs.attributes.get("gen_ai.usage.output_tokens") == 0
    assert rs.attributes.get("gen_ai.response.id") == "msg_ERR"


def test_non_2xx_response_without_usage_in_body_omits_usage_tokens(fresh_tracer):
    """Counterpart: error body without ``usage`` must leave the span's
    usage attrs unset (per §4.4: omit when unknown)."""
    tracer, exporter, _ = fresh_tracer
    request = _make_request(model="claude-haiku-4-5")
    response = _make_response(
        status_code=500,
        request_id="r-server-err",
        body={"type": "error", "error": {"type": "api_error", "message": "boom"}},
    )
    parent = tracer.start_span("anthropic.chat")
    with trace.use_span(parent, end_on_exit=False):
        _sync_send_wrapper(lambda *a, **kw: response, None, (request,), {})
    parent.end()

    from opentelemetry.trace import StatusCode
    retry_spans = _retry_spans(exporter)
    assert len(retry_spans) == 1
    rs = retry_spans[0]
    assert rs.status.status_code == StatusCode.ERROR
    assert rs.attributes.get("http.status_code") == 500
    assert rs.attributes.get("error.type") == "anthropic.InternalServerError"
    assert rs.attributes.get("gen_ai.usage.input_tokens") is None
    assert rs.attributes.get("gen_ai.usage.output_tokens") is None


def test_non_streaming_response_with_malformed_body_does_not_crash(fresh_tracer):
    """Body parse failure (non-JSON / malformed) must NOT raise."""
    tracer, exporter, _ = fresh_tracer
    request = _make_request(model="claude-haiku-4-5")

    def explode():
        raise ValueError("malformed JSON")

    headers = {"request-id": "req-bad-body"}
    response = SimpleNamespace(status_code=200, headers=headers, json=explode)

    parent = tracer.start_span("anthropic.chat")
    with trace.use_span(parent, end_on_exit=False):
        _sync_send_wrapper(lambda *a, **kw: response, None, (request,), {})
    parent.end()

    retry_spans = _retry_spans(exporter)
    assert len(retry_spans) == 1
    from opentelemetry.trace import StatusCode
    assert retry_spans[0].status.status_code == StatusCode.OK
    assert retry_spans[0].attributes.get("http.status_code") == 200
    assert retry_spans[0].attributes.get("gen_ai.response.id") == "req-bad-body"


# ---------------------------------------------------------------------------
# Async path.
# ---------------------------------------------------------------------------

def test_async_send_wrapper_emits(fresh_tracer):
    tracer, exporter, _ = fresh_tracer
    request = _make_request()

    async def wrapped(*a, **kw):
        return _make_response(200)

    async def run():
        parent = tracer.start_span("anthropic.chat")
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
    retry_spans = _retry_spans(exporter)
    assert len(retry_spans) == 1
    assert retry_spans[0].attributes.get("gen_ai.system") == "Anthropic"

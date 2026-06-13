"""Tests for ST-10.4 Bedrock direct-SDK retry-attempt emission.

Covers (per RETRY_LOOP.md §4.4 Bedrock row + §4.7 suppression):
  - install_event_hooks_on_client registers both event-name patterns
    on a bedrock-runtime client's botocore event system.
  - Single-attempt happy path: before-send → response-received pair
    emits ONE retry_attempt span; parent marker set; attrs include
    model + http.status_code + gen_ai.usage.* from Converse response.
  - Multi-attempt retry path: N siblings under one parent.
  - Error attempt: response-received with ``exception`` sets ERROR
    status + error.type.
  - No-parent guard.
  - §4.7 suppression — context-API and framework-registry both skip.
  - Defensive: before-send firing twice without an intervening
    response-received closes the orphaned prior span as ERROR.

UNIT tests — drive ``_before_send_hook`` / ``_response_received_hook``
directly with fake AWSPreparedRequest-shaped objects, so the test does
not depend on a real boto3 client / real AWS credentials / network.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from opentelemetry import context as context_api
from opentelemetry import trace
from opentelemetry.instrumentation.bedrock.retry_handler import (
    _CTX_SPAN_KEY,
    _CTX_TOKEN_KEY,
    _FR_HAS_ATTEMPT_CHILD_KEY,
    _FR_LLM_ATTEMPT_SPAN_NAME_PREFIX,
    _BEFORE_SEND_PATTERN,
    _RESPONSE_RECEIVED_PATTERN,
    _before_send_hook,
    _response_received_hook,
    install_event_hooks_on_client,
    uninstall_event_hooks_on_client,
)
from opentelemetry.instrumentation.fortifyroot import (
    clear_attempt_counters_for_test,
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
def reset_registry():
    retry_registry._reset_for_test()
    clear_attempt_counters_for_test()
    yield
    retry_registry._reset_for_test()
    clear_attempt_counters_for_test()


def _make_request(model: str = "anthropic.claude-haiku-4-5",
                  region: str = "us-east-1",
                  operation: str = "invoke") -> SimpleNamespace:
    """Build a fake AWSPreparedRequest-shaped object.

    The retry hook reads:
      - request.url           (string with /model/{modelId}/{op})
      - request.context       (mutable dict — propagated to response-received)
    """
    url = f"https://bedrock-runtime.{region}.amazonaws.com/model/{model}/{operation}"
    return SimpleNamespace(url=url, context={})


def _make_http_response(status_code: int = 200, request_id: str = "amzn-req-1") -> SimpleNamespace:
    headers = {"x-amzn-RequestId": request_id}
    return SimpleNamespace(status_code=status_code, headers=headers)


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
# Event-hook registration.
# ---------------------------------------------------------------------------

def test_install_event_hooks_registers_both_patterns():
    """install_event_hooks_on_client(client) MUST register the
    before-send AND response-received patterns on the client's
    botocore event system."""
    client = MagicMock()
    install_event_hooks_on_client(client)

    register_calls = client.meta.events.register.call_args_list
    patterns_registered = [call.args[0] for call in register_calls]
    assert _BEFORE_SEND_PATTERN in patterns_registered, (
        f"before-send pattern must be registered; got {patterns_registered}"
    )
    assert _RESPONSE_RECEIVED_PATTERN in patterns_registered, (
        f"response-received pattern must be registered; got {patterns_registered}"
    )

    # Each registration uses a stable unique_id so botocore dedups.
    for call in register_calls:
        assert "unique_id" in call.kwargs, "unique_id must be supplied for dedup"
        assert call.kwargs["unique_id"].startswith("fortifyroot.bedrock.llm_attempt.")


def test_uninstall_event_hooks_unregisters_both_patterns():
    client = MagicMock()
    install_event_hooks_on_client(client)
    uninstall_event_hooks_on_client(client)

    unregister_calls = client.meta.events.unregister.call_args_list
    patterns_unregistered = [call.args[0] for call in unregister_calls]
    assert _BEFORE_SEND_PATTERN in patterns_unregistered
    assert _RESPONSE_RECEIVED_PATTERN in patterns_unregistered


def test_install_on_client_without_meta_events_is_a_noop_no_crash():
    """If the client doesn't have .meta.events, install logs debug
    and returns gracefully — no exception."""
    client = SimpleNamespace()
    install_event_hooks_on_client(client)  # must not raise
    uninstall_event_hooks_on_client(client)  # must not raise


# ---------------------------------------------------------------------------
# Single-attempt happy path.
# ---------------------------------------------------------------------------

def test_single_attempt_emits_one_span_with_marker(fresh_tracer):
    tracer, exporter, _ = fresh_tracer
    request = _make_request(model="anthropic.claude-haiku-4-5", operation="invoke")

    parent = tracer.start_span("bedrock.completion")
    with trace.use_span(parent, end_on_exit=False):
        # before-send fires first.
        _before_send_hook(
            event_name="before-send.bedrock-runtime.InvokeModel",
            request=request,
        )
        # Span stowed on request.context.
        assert _CTX_SPAN_KEY in request.context

        # response-received fires with the same context dict.
        _response_received_hook(
            http_response=_make_http_response(200, "amzn-req-1"),
            parsed={"usage": {"inputTokens": 12, "outputTokens": 7}},
            context=request.context,
            exception=None,
        )
        # Context cleaned up.
        assert _CTX_SPAN_KEY not in request.context
        assert _CTX_TOKEN_KEY not in request.context

    parent.end()

    spans = exporter.get_finished_spans()
    parent_exported = next(s for s in spans if s.name == "bedrock.completion")
    retry_spans = _retry_spans(exporter)
    assert len(retry_spans) == 1
    rs = retry_spans[0]
    _assert_attempt_sequence(retry_spans)
    assert rs.parent.span_id == parent_exported.context.span_id
    assert rs.attributes.get("fortifyroot.span.role") == "llm_attempt"
    assert rs.attributes.get("gen_ai.system") == "AWS"
    assert rs.attributes.get("gen_ai.request.model") == "anthropic.claude-haiku-4-5"
    assert rs.attributes.get("gen_ai.operation.name") == "chat"
    assert rs.attributes.get("http.status_code") == 200
    assert rs.attributes.get("gen_ai.response.id") == "amzn-req-1"
    assert rs.attributes.get("gen_ai.usage.input_tokens") == 12
    assert rs.attributes.get("gen_ai.usage.output_tokens") == 7
    assert rs.attributes.get("server.address") == "bedrock-runtime.us-east-1.amazonaws.com"
    assert rs.attributes.get("server.port") == 443
    assert parent_exported.attributes.get(_FR_HAS_ATTEMPT_CHILD_KEY) is True


# ---------------------------------------------------------------------------
# Multi-attempt retry path.
# ---------------------------------------------------------------------------

def test_three_attempts_share_parent(fresh_tracer):
    """429 → 429 → 200 chain produces 3 sibling retry_attempt spans
    under one parent. Each attempt is a fresh AWSPreparedRequest with
    its own context dict (matching botocore's per-attempt model)."""
    tracer, exporter, _ = fresh_tracer

    parent = tracer.start_span("bedrock.converse")
    with trace.use_span(parent, end_on_exit=False):
        for status_code, _ in [(429, "a"), (429, "b"), (200, "c")]:
            req = _make_request(operation="converse")
            _before_send_hook(
                event_name="before-send.bedrock-runtime.Converse",
                request=req,
            )
            _response_received_hook(
                http_response=_make_http_response(status_code),
                parsed={},
                context=req.context,
                exception=None,
            )
    parent.end()

    spans = exporter.get_finished_spans()
    parent_exported = next(s for s in spans if s.name == "bedrock.converse")
    retry_spans = _retry_spans(exporter)
    assert len(retry_spans) == 3
    _assert_attempt_sequence(retry_spans)

    parent_ids = {s.parent.span_id for s in retry_spans}
    assert parent_ids == {parent_exported.context.span_id}

    from opentelemetry.trace import StatusCode
    err = sum(1 for s in retry_spans if s.status.status_code == StatusCode.ERROR)
    ok = sum(1 for s in retry_spans if s.status.status_code == StatusCode.OK)
    assert err == 2 and ok == 1

    for s in retry_spans:
        if s.status.status_code == StatusCode.ERROR:
            assert s.attributes.get("http.status_code") == 429
            assert s.attributes.get("error.type") == "botocore.ThrottlingException"


# ---------------------------------------------------------------------------
# Error / exception path.
# ---------------------------------------------------------------------------

def test_exception_path_records_error(fresh_tracer):
    """When response-received fires with ``exception`` set, the span
    finalises as ERROR with error.type."""
    tracer, exporter, _ = fresh_tracer

    class ConnectError(Exception):
        pass

    err = ConnectError("connect timeout")

    request = _make_request()
    parent = tracer.start_span("bedrock.completion")
    with trace.use_span(parent, end_on_exit=False):
        _before_send_hook(
            event_name="before-send.bedrock-runtime.InvokeModel",
            request=request,
        )
        _response_received_hook(
            http_response=None,
            parsed=None,
            context=request.context,
            exception=err,
        )
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

def test_marker_NOT_set_when_no_attempts_fire(fresh_tracer):
    tracer, exporter, _ = fresh_tracer
    parent = tracer.start_span("bedrock.completion")
    parent.end()
    parent_exported = next(
        s for s in exporter.get_finished_spans() if s.name == "bedrock.completion"
    )
    assert _FR_HAS_ATTEMPT_CHILD_KEY not in (parent_exported.attributes or {})


# ---------------------------------------------------------------------------
# No-parent guard.
# ---------------------------------------------------------------------------

def test_no_parent_does_not_emit(fresh_tracer):
    tracer, exporter, _ = fresh_tracer
    request = _make_request()
    _before_send_hook(
        event_name="before-send.bedrock-runtime.InvokeModel",
        request=request,
    )
    # No span stowed on context because no parent.
    assert _CTX_SPAN_KEY not in request.context
    _response_received_hook(
        http_response=_make_http_response(200),
        parsed={},
        context=request.context,
        exception=None,
    )
    assert len(_retry_spans(exporter)) == 0


# ---------------------------------------------------------------------------
# §4.7 suppression.
# ---------------------------------------------------------------------------

def test_context_suppression_skips_emission(fresh_tracer):
    tracer, exporter, _ = fresh_tracer
    request = _make_request()
    parent = tracer.start_span("bedrock.completion")
    token = context_api.attach(
        context_api.set_value(SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY, True)
    )
    try:
        with trace.use_span(parent, end_on_exit=False):
            _before_send_hook(
                event_name="before-send.bedrock-runtime.InvokeModel",
                request=request,
            )
            _response_received_hook(
                http_response=_make_http_response(200),
                parsed={},
                context=request.context,
                exception=None,
            )
    finally:
        context_api.detach(token)
    parent.end()
    assert len(_retry_spans(exporter)) == 0


def test_framework_registry_suppression_skips_emission(fresh_tracer):
    tracer, exporter, _ = fresh_tracer
    request = _make_request()
    parent = tracer.start_span("bedrock.completion")
    tok = register_framework_attempt()
    assert is_framework_owned()
    try:
        with trace.use_span(parent, end_on_exit=False):
            _before_send_hook(
                event_name="before-send.bedrock-runtime.InvokeModel",
                request=request,
            )
            _response_received_hook(
                http_response=_make_http_response(200),
                parsed={},
                context=request.context,
                exception=None,
            )
    finally:
        unregister_framework_attempt(tok)
    parent.end()
    assert len(_retry_spans(exporter)) == 0


def test_direct_sdk_wrapper_does_not_self_register_in_framework_registry(fresh_tracer):
    """REGRESSION GUARD (review-driven fix 2026-05-13): the bedrock
    retry hooks MUST NOT register tokens in the §4.7.1 framework
    registry. Direct-SDK wrappers only CONSULT via
    ``is_framework_owned()``."""
    tracer, _, _ = fresh_tracer
    request = _make_request()
    parent = tracer.start_span("bedrock.completion")
    with trace.use_span(parent, end_on_exit=False):
        assert not is_framework_owned()
        _before_send_hook(
            event_name="before-send.bedrock-runtime.InvokeModel",
            request=request,
        )
        assert not is_framework_owned(), (
            "before-send hook must NOT self-register a framework token"
        )
        _response_received_hook(
            http_response=_make_http_response(200),
            parsed={},
            context=request.context,
            exception=None,
        )
        assert not is_framework_owned()
    parent.end()


# ---------------------------------------------------------------------------
# Defensive: before-send fires twice (botocore pairs guaranteed but
# tolerate weird sequences).
# ---------------------------------------------------------------------------

def test_double_before_send_closes_orphan_as_error(fresh_tracer):
    tracer, exporter, _ = fresh_tracer
    request = _make_request()
    parent = tracer.start_span("bedrock.completion")
    with trace.use_span(parent, end_on_exit=False):
        _before_send_hook(
            event_name="before-send.bedrock-runtime.InvokeModel",
            request=request,
        )
        first_span = request.context[_CTX_SPAN_KEY]
        # A second before-send (without the paired response-received)
        # supersedes the first — the orphan is closed as ERROR so it
        # doesn't leak.
        _before_send_hook(
            event_name="before-send.bedrock-runtime.InvokeModel",
            request=request,
        )
        _response_received_hook(
            http_response=_make_http_response(200),
            parsed={},
            context=request.context,
            exception=None,
        )
    parent.end()

    retry_spans = _retry_spans(exporter)
    assert len(retry_spans) == 2, f"expected 2 spans (orphan + completed); got {len(retry_spans)}"
    # The orphan span (first one started) MUST be ERROR.
    from opentelemetry.trace import StatusCode
    errors = [s for s in retry_spans if s.status.status_code == StatusCode.ERROR]
    oks = [s for s in retry_spans if s.status.status_code == StatusCode.OK]
    assert len(errors) == 1 and len(oks) == 1
    # The orphan and first_span share the same span context.
    assert first_span.get_span_context().span_id == errors[0].context.span_id


# ---------------------------------------------------------------------------
# Response-received with no prior before-send is a no-op.
# ---------------------------------------------------------------------------

def test_response_received_with_no_prior_before_send_is_noop(fresh_tracer):
    """Stray response-received (no paired before-send started a span)
    must NOT crash. Defensive contract per the hook docstring."""
    _, exporter, _ = fresh_tracer
    _response_received_hook(
        http_response=_make_http_response(200),
        parsed={},
        context={},  # empty context
        exception=None,
    )
    assert len(_retry_spans(exporter)) == 0


# ---------------------------------------------------------------------------
# Defensive: request with no .context dict.
# ---------------------------------------------------------------------------

def test_before_send_with_request_missing_context_is_noop(fresh_tracer):
    """If a botocore request somehow arrives without a usable
    ``.context`` dict (defensive — every AWSPreparedRequest in current
    botocore has one), the hook must NOT crash and must NOT emit a
    span (no place to stash state for the paired response-received)."""
    tracer, exporter, _ = fresh_tracer
    # SimpleNamespace with no `context` attribute.
    request_no_context = SimpleNamespace(
        url="https://bedrock-runtime.us-east-1.amazonaws.com/model/x/invoke"
    )
    parent = tracer.start_span("bedrock.completion")
    with trace.use_span(parent, end_on_exit=False):
        # Must not raise.
        _before_send_hook(
            event_name="before-send.bedrock-runtime.InvokeModel",
            request=request_no_context,
        )
    parent.end()
    assert len(_retry_spans(exporter)) == 0


# ---------------------------------------------------------------------------
# M1: streaming-span ambient-context fix (review-driven 2026-05-13).
# ---------------------------------------------------------------------------

def test_streaming_event_skips_emission_entirely(fresh_tracer):
    """ST-10.4 (review-driven 2026-05-17): when the botocore operation
    is a streaming one (event name ends with ``Stream``:
    ``InvokeModelWithResponseStream`` / ``ConverseStream``), the
    ``_before_send_hook`` SKIPS retry_attempt emission entirely.

    Rationale matches openai/anthropic streaming-skip: streaming
    retry_attempts cannot carry usage at attempt-end (usage arrives
    via the stream-completion callback in the Bedrock streaming
    wrapper, AFTER our hook has finalised) but §4.5 dedup would
    still promote them to canonical → zero-token LLMUsageEvents
    breaking fr-system-tests. The parent
    ``bedrock.completion`` / ``bedrock.converse`` span (which DOES
    get full usage via ``stream_done``) stays canonical. Streaming
    retry-loop detection is the deferred follow-up
    ``ST-10.4-FOLLOWUP-streaming-usage``.

    This test supersedes the earlier
    ``test_streaming_span_via_use_span_parents_retry_attempt`` from
    the M1 fix round — the M1 ``trace.use_span`` wrapper in the
    Bedrock instrumentor is still needed for non-streaming
    operations and for future work that wants to plug streaming
    retry-loop detection back in.
    """
    tracer, exporter, _ = fresh_tracer
    streaming_span = tracer.start_span("bedrock.completion")
    request = _make_request(operation="invoke-with-response-stream")

    with trace.use_span(streaming_span, end_on_exit=False):
        _before_send_hook(
            event_name="before-send.bedrock-runtime.InvokeModelWithResponseStream",
            request=request,
        )
        # ``request.context`` should be UNTOUCHED — no span stashed.
        assert _CTX_SPAN_KEY not in request.context
        _response_received_hook(
            http_response=_make_http_response(200),
            parsed={},
            context=request.context,
            exception=None,
        )
    streaming_span.end()

    retry_spans = _retry_spans(exporter)
    assert len(retry_spans) == 0, (
        f"streaming events MUST skip retry_attempt emission entirely; "
        f"got {len(retry_spans)} span(s)."
    )

    # ConverseStream variant — same behaviour.
    request2 = _make_request(operation="converse-stream")
    with trace.use_span(streaming_span, end_on_exit=False):
        _before_send_hook(
            event_name="before-send.bedrock-runtime.ConverseStream",
            request=request2,
        )
        assert _CTX_SPAN_KEY not in request2.context


def test_streaming_span_without_use_span_skips_emission(fresh_tracer):
    """Counterpoint to the M1 fix: WITHOUT ``trace.use_span``, the
    streaming Bedrock span is NOT ambient → ``_resolve_parent_span``
    returns None (no valid ambient) → retry_attempt skipped. Documents
    the pre-fix failure mode so a future regression on the instrumentor
    side gets caught."""
    tracer, exporter, _ = fresh_tracer
    streaming_span = tracer.start_span("bedrock.completion")  # NOT made current
    request = _make_request(operation="invoke-with-response-stream")

    # No use_span wrap — mimicking the pre-M1 instrumentor.
    _before_send_hook(
        event_name="before-send.bedrock-runtime.InvokeModelWithResponseStream",
        request=request,
    )
    _response_received_hook(
        http_response=_make_http_response(200),
        parsed={},
        context=request.context,
        exception=None,
    )
    streaming_span.end()

    assert len(_retry_spans(exporter)) == 0, (
        "without use_span(streaming_span), retry_attempt must be skipped — "
        "no valid ambient parent"
    )

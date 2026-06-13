"""ST-10.4 retry-aware emission for the Bedrock (botocore) direct SDK.

Per RETRY_LOOP.md §4.4 Bedrock row + §4.7 suppression discipline +
ST-10.0 hook-table addendum (in phase_st10_retryloop.txt):

  - Use botocore's PUBLIC event-hook API on bedrock-runtime clients:
      * ``before-send.bedrock-runtime.*`` fires once per HTTP attempt
        (verified in botocore 1.35.x ``endpoint.py:_do_get_response``)
        with the prepared ``request`` as a keyword arg.
      * ``response-received.bedrock-runtime.*`` fires when the attempt
        completes (success OR error) with ``http_response``, ``parsed``,
        ``exception``, and ``context`` kwargs.
  - We DO NOT monkey-patch botocore internals. Event hooks are
    documented public API, low version-fragility. The earlier draft
    referenced ``after-call.*`` — that event does NOT exist in current
    botocore for per-attempt callbacks and has been removed.
  - Per-attempt correlation: store the started span on
    ``request.context`` under ``_CTX_SPAN_KEY`` (a per-request mutable
    dict botocore propagates end-to-end). ``before-send`` writes;
    ``response-received`` reads via the ``context`` kwarg, which
    references the SAME dict. This avoids thread-local state and
    survives async/aiobotocore variations cleanly. NB: we deliberately
    do NOT register a framework-attempt token here — the §4.7.1
    registry's documented contract reserves registration for FRAMEWORK
    wrappers (LiteLLM / LangChain / LlamaIndex). Direct-SDK wrappers
    only CONSULT via ``is_framework_owned()``. (See the 2026-05-13
    review-driven C1 fix; an earlier draft of this module incorrectly
    registered a token.)
  - Endpoint allow-list: implicit via the event-name pattern
    ``bedrock-runtime.*``. Only bedrock-runtime operations (Invoke /
    Converse and their streaming variants) fire these events; non-LLM
    AWS service traffic (S3, STS, etc.) is unaffected.
  - Suppression discipline (§4.7): before emitting, check BOTH the
    OTel context ``SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY`` AND
    the shared ``is_framework_owned()`` registry (consult-only — see
    above).
  - Parent resolution: use the current OTel ambient span (the existing
    Bedrock outer span ``bedrock.completion`` / ``bedrock.converse``
    when the user calls through the FR-wrapped Bedrock client). If no
    valid ambient parent exists, skip emission.

This module exports ``install_event_hooks_on_client(client)`` which the
Bedrock instrumentor calls inside its existing ``_wrap`` path when a
bedrock-runtime client is created. ``uninstall_event_hooks_on_client(client)``
is provided for symmetry / tests, though in practice clients are
short-lived and don't need explicit unhook.

Tests: see ``tests/test_retry_attempt_emission.py``.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Optional

from opentelemetry import context as context_api
from opentelemetry import trace
from opentelemetry.instrumentation.bedrock.version import __version__
from opentelemetry.instrumentation.fortifyroot import (
    FR_HAS_ATTEMPT_CHILD_KEY,
    is_framework_owned,
    llm_attempt_attributes,
    next_llm_attempt,
)
from opentelemetry.semconv_ai import SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY
from opentelemetry.trace import SpanKind, Status, StatusCode

logger = logging.getLogger(__name__)


# ST-10 §4.4 / §4.5 constants.
_FR_SPAN_ROLE_KEY = "fortifyroot.span.role"
_FR_LLM_ATTEMPT_SPAN_NAME_PREFIX = "fortifyroot.bedrock"
_FR_SPAN_ROLE_LLM_ATTEMPT = "llm_attempt"
_FR_HAS_ATTEMPT_CHILD_KEY = FR_HAS_ATTEMPT_CHILD_KEY

# Event-name patterns. ``*`` matches any operation under bedrock-runtime.
_BEFORE_SEND_PATTERN = "before-send.bedrock-runtime.*"
_RESPONSE_RECEIVED_PATTERN = "response-received.bedrock-runtime.*"

# Stable correlation key used inside the per-request context dict.
_CTX_SPAN_KEY = "_fr_retry_attempt_span"
_CTX_TOKEN_KEY = "_fr_retry_attempt_framework_token"

# Tracer provider the hooks should emit spans through. ``None`` → fall
# back to the global TracerProvider. Set via the optional
# ``tracer_provider`` arg to ``install_event_hooks_on_client`` so retry
# spans go to the same provider as the bedrock logical span (e.g.
# ``bedrock.completion``) emitted by the FR Bedrock instrumentor.
# Without this, a consumer passing an explicit provider to
# ``BedrockInstrumentor().instrument(tracer_provider=...)`` would get
# the parent span on their provider but the retry_attempt span sent
# into a no-op tracer. Module-level (not per-client) because all
# bedrock-runtime clients created under one instrumentor share the
# same provider config. (review-driven 2026-05-16 fix.)
_tracer_provider = None


def _is_suppressed() -> bool:
    try:
        if context_api.get_value(SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY):
            return True
    except Exception:
        pass
    try:
        if is_framework_owned():
            return True
    except Exception:
        pass
    return False


def _resolve_parent_span() -> Optional["trace.Span"]:
    current = trace.get_current_span()
    if current is None:
        return None
    ctx = current.get_span_context()
    if ctx is None or not ctx.is_valid:
        return None
    return current


def _set_parent_marker(parent_span: "trace.Span") -> None:
    try:
        parent_span.set_attribute(_FR_HAS_ATTEMPT_CHILD_KEY, True)
    except Exception:
        logger.debug("failed to set has_attempt_child on parent", exc_info=True)


def _operation_from_event_name(event_name: str) -> str:
    """Derive ``gen_ai.operation.name`` from a botocore event name like
    ``before-send.bedrock-runtime.InvokeModel`` →  ``chat`` /
    ``text_completion`` mapping.

    Bedrock operations seen at the bedrock-runtime endpoint:
      - InvokeModel / InvokeModelWithResponseStream → chat/completion
        (depends on the model — we generalise to "chat")
      - Converse / ConverseStream → chat
    """
    if not isinstance(event_name, str):
        return "chat"
    name = event_name.split(".")[-1].lower()
    if "invoke" in name or "converse" in name:
        return "chat"
    return "chat"


def _model_from_request(request: Any) -> Optional[str]:
    """Best-effort: extract ``modelId`` from the request URL.

    Bedrock URL shape: ``/model/{model-id}/invoke`` or
    ``/model/{model-id}/converse``. Some model IDs themselves contain
    slashes after the vendor prefix, so we extract everything between
    ``/model/`` and the trailing ``/invoke``/``/converse``.
    """
    try:
        url = getattr(request, "url", None)
        if not isinstance(url, str):
            return None
        # Path-only form for endpoint requests: starts with /model/...
        idx = url.find("/model/")
        if idx < 0:
            return None
        rest = url[idx + len("/model/"):]
        for suffix in ("/invoke-with-response-stream", "/invoke", "/converse-stream", "/converse"):
            if rest.endswith(suffix):
                return rest[: -len(suffix)]
        # Fallback: trim any trailing query / slash.
        slash = rest.find("/")
        if slash > 0:
            return rest[:slash]
        return rest or None
    except Exception:
        return None


def _server_attrs_from_request(request: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    try:
        url = getattr(request, "url", None)
        if not isinstance(url, str):
            return out
        # AWSPreparedRequest.url is a full URL string, not a parsed object.
        from urllib.parse import urlparse
        parsed = urlparse(url)
        if parsed.hostname:
            out["server.address"] = parsed.hostname
        if parsed.port is not None:
            out["server.port"] = parsed.port
        elif parsed.scheme == "https":
            out["server.port"] = 443
        elif parsed.scheme == "http":
            out["server.port"] = 80
    except Exception:
        pass
    return out


def _start_attempt_span(
    request: Any,
    event_name: str,
    parent_span: "trace.Span",
) -> "trace.Span":
    span_name, attempt_number, is_retry = next_llm_attempt(
        parent_span,
        _FR_LLM_ATTEMPT_SPAN_NAME_PREFIX,
    )
    attrs: dict[str, Any] = {
        "gen_ai.system": "AWS",  # matches existing Bedrock instrumentor
        "gen_ai.operation.name": _operation_from_event_name(event_name),
    }
    attrs.update(llm_attempt_attributes(attempt_number, is_retry))
    model = _model_from_request(request)
    if model:
        attrs["gen_ai.request.model"] = model
    attrs.update(_server_attrs_from_request(request))

    tracer = trace.get_tracer(__name__, __version__, _tracer_provider)
    parent_ctx = trace.set_span_in_context(parent_span)
    span = tracer.start_span(
        span_name,
        kind=SpanKind.CLIENT,
        attributes=attrs,
        context=parent_ctx,
    )
    return span


def _request_context(request: Any) -> Optional[dict]:
    """Return the AWSPreparedRequest's per-request context dict, or None
    if unavailable. botocore exposes this as ``request.context`` —
    a mutable dict that propagates from request creation through to the
    ``response-received`` event.
    """
    ctx = getattr(request, "context", None)
    if isinstance(ctx, dict):
        return ctx
    return None


def _finalize_success(span: "trace.Span", http_response: Any, parsed: Any) -> None:
    try:
        status_code = getattr(http_response, "status_code", None)
        if isinstance(status_code, int):
            span.set_attribute("http.status_code", status_code)
        if isinstance(status_code, int) and 200 <= status_code < 300:
            # Best-effort: response id from x-amzn-RequestId header.
            try:
                headers = getattr(http_response, "headers", None) or {}
                if hasattr(headers, "get"):
                    rid = headers.get("x-amzn-RequestId") or headers.get("x-amzn-requestid")
                    if isinstance(rid, str) and rid:
                        span.set_attribute("gen_ai.response.id", rid)
            except Exception:
                pass
            # Best-effort: token usage from parsed Converse response.
            try:
                if isinstance(parsed, dict):
                    usage = parsed.get("usage") or {}
                    if isinstance(usage, dict):
                        if isinstance(usage.get("inputTokens"), int):
                            span.set_attribute("gen_ai.usage.input_tokens", usage["inputTokens"])
                        if isinstance(usage.get("outputTokens"), int):
                            span.set_attribute("gen_ai.usage.output_tokens", usage["outputTokens"])
            except Exception:
                pass
            span.set_status(Status(StatusCode.OK))
        else:
            err_type = "botocore.HTTPStatusError"
            if isinstance(status_code, int):
                if status_code == 429:
                    err_type = "botocore.ThrottlingException"
                elif 500 <= status_code < 600:
                    err_type = "botocore.InternalServerError"
                elif status_code in (401, 403):
                    err_type = "botocore.AccessDeniedException"
            span.set_attribute("error.type", err_type)
            span.set_status(Status(StatusCode.ERROR, f"http {status_code}"))
    except Exception:
        logger.debug("failed to set success attrs on bedrock llm_attempt", exc_info=True)


def _finalize_error(span: "trace.Span", exception: BaseException) -> None:
    try:
        error_type = type(exception).__name__
        mod = type(exception).__module__
        if isinstance(mod, str) and mod and mod != "builtins":
            error_type = f"{mod.split('.')[0]}.{error_type}"
        span.set_attribute("error.type", error_type)
        status_code = getattr(exception, "status_code", None) or getattr(
            getattr(exception, "response", None), "status_code", None
        )
        if isinstance(status_code, int):
            span.set_attribute("http.status_code", status_code)
        try:
            span.record_exception(exception)
        except Exception:
            pass
        span.set_status(Status(StatusCode.ERROR, str(exception)))
    except Exception:
        logger.debug("failed to set error attrs on bedrock llm_attempt", exc_info=True)


def _before_send_hook(event_name: Optional[str] = None, request: Any = None, **_kwargs) -> None:
    """botocore ``before-send.bedrock-runtime.*`` event handler.

    Starts a retry_attempt span (when guards permit) and stows it on
    ``request.context`` for retrieval by the paired ``response-received``
    hook.

    No self-registration in the §4.7.1 framework registry: the registry's
    contract reserves registration for FRAMEWORK wrappers (LiteLLM /
    LangChain / LlamaIndex). Direct-SDK wrappers only CONSULT via
    ``is_framework_owned()``. Defensive cleanup of any prior context-stored
    span on this request still runs, but it no longer touches the
    framework registry.

    Streaming skip (ST-10.4 review-driven 2026-05-17): when the
    botocore operation is a streaming one (event name ends with
    ``Stream``: ``InvokeModelWithResponseStream`` /
    ``ConverseStream``), this hook skips retry_attempt emission.
    Rationale matches openai/anthropic: streaming retry_attempts
    cannot carry token usage at attempt-end (usage arrives via the
    stream-completion callback installed by the Bedrock streaming
    wrapper, AFTER our hook has already finalised the span), but
    §4.5 dedup would still promote them to canonical → zero-token
    LLMUsageEvent. Leaving the parent ``bedrock.completion`` /
    ``bedrock.converse`` span (which DOES get full usage from
    ``stream_done``) as the canonical event. Streaming retry-loop
    detection is the deferred follow-up
    ``ST-10.4-FOLLOWUP-streaming-usage``.
    """
    try:
        if request is None:
            return
        if isinstance(event_name, str) and event_name.endswith("Stream"):
            return
        ctx = _request_context(request)
        if ctx is None:
            return

        # Defensive: if a prior attempt's span wasn't cleared (e.g. the
        # paired ``response-received`` didn't fire — botocore normally
        # guarantees pairing but we tolerate the weird sequence), end
        # it as ERROR so the span doesn't leak.
        prev_span = ctx.pop(_CTX_SPAN_KEY, None)
        # Old payload may have a token field (pre-fix state) — drop it
        # silently if present; we no longer register tokens.
        ctx.pop(_CTX_TOKEN_KEY, None)
        if prev_span is not None:
            try:
                prev_span.set_status(
                    Status(StatusCode.ERROR, "bedrock retry_attempt superseded (paired response-received missing)")
                )
                prev_span.end()
            except Exception:
                pass

        if _is_suppressed():
            return

        parent = _resolve_parent_span()
        if parent is None:
            return
        span = _start_attempt_span(request, event_name or "", parent)
        _set_parent_marker(parent)
        ctx[_CTX_SPAN_KEY] = span
    except Exception:
        logger.debug("ST-10.4: bedrock before-send hook failed", exc_info=True)


class _ResponseDictAdapter:
    """Minimal adapter so ``_finalize_success`` can read ``status_code``
    + ``headers`` uniformly from either an httpx-style response object
    (``.status_code``, ``.headers``) OR botocore's ``response_dict``
    (a plain dict with ``'status_code'`` + ``'headers'`` keys).

    botocore 1.42.x (and likely earlier point-releases) emits the
    ``response-received`` event with ``response_dict=`` + ``parsed_response=``
    kwargs — NOT ``http_response=`` + ``parsed=``. The earlier hook
    signature missed both rename pairs, leaving the retry_attempt span
    with no status code, no OK/ERROR status, and ``IsError=False`` for
    every attempt. Backend's RetryDetector then skipped the whole
    sibling group on its ``failedCount == 0`` check.
    """

    __slots__ = ("status_code", "headers")

    def __init__(self, response_dict: dict) -> None:
        sc = response_dict.get("status_code")
        self.status_code = sc if isinstance(sc, int) else None
        h = response_dict.get("headers")
        self.headers = h if isinstance(h, dict) else {}


def _response_received_hook(
    http_response: Any = None,
    parsed: Any = None,
    response_dict: Any = None,
    parsed_response: Any = None,
    context: Any = None,
    exception: Any = None,
    **_kwargs,
) -> None:
    """botocore ``response-received.bedrock-runtime.*`` event handler.

    Finalises the retry_attempt span started by the paired
    ``before-send`` hook. ``context`` is the per-request context dict
    (same dict referenced by ``request.context`` at before-send time).

    Accepts BOTH botocore parameter-name conventions:
      * ``http_response`` + ``parsed`` (older botocore signature the
        fork-side unit tests drive directly).
      * ``response_dict`` + ``parsed_response`` (botocore 1.42.x — the
        names the real event emitter actually uses; verified via
        ST-10.6 fr-system-tests probe).

    Per-attempt finalisation reads ``status_code`` and ``headers`` from
    whichever shape is present; the rest of the body just falls through
    to ``_finalize_success`` / ``_finalize_error``.
    """
    try:
        if not isinstance(context, dict):
            return
        span = context.pop(_CTX_SPAN_KEY, None)
        # Old payload key (pre-fix state) — drop silently if present.
        context.pop(_CTX_TOKEN_KEY, None)
        if span is None:
            return

        # Reconcile old/new botocore param names.  If both are present
        # (unlikely, but defensive), prefer the explicit older names so
        # the existing unit tests' direct invocation path is not
        # disturbed.
        if http_response is None and isinstance(response_dict, dict):
            http_response = _ResponseDictAdapter(response_dict)
        if parsed is None and parsed_response is not None:
            parsed = parsed_response

        try:
            if exception is not None:
                _finalize_error(span, exception)
            else:
                _finalize_success(span, http_response, parsed)
        finally:
            try:
                span.end()
            except Exception:
                pass
    except Exception:
        logger.debug("ST-10.4: bedrock response-received hook failed", exc_info=True)


def install_event_hooks_on_client(client: Any, tracer_provider=None) -> None:
    """Register retry-attempt event hooks on a freshly-created
    bedrock-runtime client. Idempotent per-client — botocore's event
    system de-dups identical (event-pattern, callable) registrations
    via the unique-id mechanism.

    ``tracer_provider`` is stashed in module-level state and used by
    the event hooks when creating retry_attempt spans. The same
    provider is used for ALL clients in this process — all bedrock
    clients created under one ``BedrockInstrumentor.instrument(...)``
    call share the same provider config. If ``None``, falls back to
    the global TracerProvider.

    Called from ``BedrockInstrumentor._wrap`` after the bedrock-runtime
    client is created and the FR per-method wraps are applied.
    """
    global _tracer_provider
    if tracer_provider is not None:
        _tracer_provider = tracer_provider
    try:
        events = getattr(client, "meta", None)
        events = getattr(events, "events", None) if events is not None else None
        if events is None:
            logger.debug(
                "ST-10.4: bedrock client missing .meta.events; "
                "skipping retry_attempt event-hook registration"
            )
            return
        events.register(
            _BEFORE_SEND_PATTERN,
            _before_send_hook,
            unique_id="fortifyroot.bedrock.llm_attempt.before-send",
        )
        events.register(
            _RESPONSE_RECEIVED_PATTERN,
            _response_received_hook,
            unique_id="fortifyroot.bedrock.llm_attempt.response-received",
        )
    except Exception:
        logger.warning(
            "ST-10.4: failed to register bedrock retry_attempt event hooks; "
            "retry_attempt emission disabled for this client",
            exc_info=True,
        )


def uninstall_event_hooks_on_client(client: Any) -> None:
    """Remove the retry-attempt event hooks from a bedrock-runtime client.
    Idempotent. Provided for tests and symmetry; production clients are
    usually short-lived and don't need explicit unhook.

    Note: the module-level ``_tracer_provider`` is NOT cleared here —
    it is shared across all bedrock clients created under one
    instrumentor and remains relevant until the instrumentor itself is
    uninstrumented. ``BedrockInstrumentor._uninstrument`` clears it
    (see ``bedrock/__init__.py``).
    """
    try:
        events = getattr(client, "meta", None)
        events = getattr(events, "events", None) if events is not None else None
        if events is None:
            return
        try:
            events.unregister(
                _BEFORE_SEND_PATTERN,
                unique_id="fortifyroot.bedrock.llm_attempt.before-send",
            )
        except Exception:
            pass
        try:
            events.unregister(
                _RESPONSE_RECEIVED_PATTERN,
                unique_id="fortifyroot.bedrock.llm_attempt.response-received",
            )
        except Exception:
            pass
    except Exception:
        logger.debug("ST-10.4: bedrock event-hook unregister failed", exc_info=True)


def _reset_tracer_provider_for_test() -> None:
    """Test-only helper: clear the module-level tracer provider. Not part
    of the public contract."""
    global _tracer_provider
    _tracer_provider = None


__all__ = [
    "install_event_hooks_on_client",
    "uninstall_event_hooks_on_client",
    "_FR_LLM_ATTEMPT_SPAN_NAME_PREFIX",
    "_FR_HAS_ATTEMPT_CHILD_KEY",
    "_before_send_hook",
    "_response_received_hook",
]

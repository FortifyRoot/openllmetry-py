"""ST-10.4 retry-aware emission for the Anthropic direct SDK.

Per RETRY_LOOP.md §4.4 Anthropic row + §4.7 suppression discipline +
ST-10.0 hook-table addendum (in phase_st10_retryloop.txt):

  - Hook ``anthropic._base_client.SyncHttpxClientWrapper.send`` and
    ``AsyncHttpxClientWrapper.send`` — fires once per HTTP attempt
    inside the SDK's internal retry loop. Avoids global httpx
    monkey-patching (which would affect every httpx caller in the
    process — including non-LLM ones).
  - Private-symbol guard: ``anthropic._base_client`` is private and
    may shift across SDK versions. We wrap defensively; if the
    private symbol is missing or incompatibly typed, log a warning
    and skip retry-attempt emission for that variant. Normal anthropic
    instrumentation continues unaffected.
  - Endpoint allow-list: only LLM endpoints emit retry_attempt spans
    (``/v1/messages`` for the modern Messages API,
    ``/v1/complete`` for legacy completions). Anything else (token
    refresh, model listing, etc.) does NOT emit retry_attempt — per
    §4.4.1 allow-listing requirement.
  - Suppression discipline (§4.7): check BOTH the OTel context
    ``SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY`` AND the shared
    ``is_framework_owned()`` registry. If either says "suppress",
    skip emission. Prevents framework retries (LiteLLM / LangChain /
    LlamaIndex) from DOUBLE-emitting when both framework wrappers
    and direct-SDK wrappers are active.
  - Parent resolution: use the current OTel ambient span. If invalid
    or absent, skip gracefully (no orphan retry_attempt span). The
    §4.5 backend dedup degrades gracefully.

Tests: see ``tests/test_retry_attempt_emission.py``.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Optional

from opentelemetry import context as context_api
from opentelemetry import trace
from opentelemetry.instrumentation.anthropic.version import __version__
from opentelemetry.instrumentation.fortifyroot import (
    is_framework_owned,
)
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.semconv_ai import SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY
from opentelemetry.trace import SpanKind, Status, StatusCode
from wrapt import wrap_function_wrapper

logger = logging.getLogger(__name__)


# ST-10 §4.4 / §4.5 constants.
_FR_RETRY_ATTEMPT_SPAN_NAME = "fortifyroot.anthropic.retry_attempt"
_FR_SPAN_ROLE_KEY = "fortifyroot.span.role"
_FR_SPAN_ROLE_RETRY_ATTEMPT = "retry_attempt"
_FR_HAS_RETRY_ATTEMPT_CHILD_KEY = "fortifyroot.span.has_retry_attempt_child"

_ANTHROPIC_BASE_CLIENT_MODULE = "anthropic._base_client"
_SYNC_WRAPPER_CLASS = "SyncHttpxClientWrapper"
_ASYNC_WRAPPER_CLASS = "AsyncHttpxClientWrapper"
_WRAPPED_METHOD = "send"

# §4.4.1 endpoint allow-list — match against the URL path suffix.
# The Anthropic SDK targets ``api.anthropic.com`` for the public
# endpoints and ``bedrock-runtime.*.amazonaws.com`` when routed through
# the AWS SDK; in both cases the path identifies the operation.
_LLM_PATH_SUFFIXES = (
    "/v1/messages",   # modern Messages API
    "/v1/complete",   # legacy completions
    "/messages",      # generic suffix (covers /v1/messages and Azure-style paths)
    "/complete",
)


_state_lock = threading.Lock()
_installed = False

# Tracer provider the wrap was installed with. ``None`` → fall back to
# the global TracerProvider. Set via ``instrument_retry_emitter(tracer_provider=...)``
# so retry spans go to the same provider as the anthropic logical span
# (e.g. ``anthropic.chat``) emitted by ``_wrap``. Without this, a
# consumer passing an explicit provider to
# ``AnthropicInstrumentor().instrument(tracer_provider=...)`` gets the
# parent span on their provider but the retry_attempt span sent into
# a no-op tracer.
_tracer_provider = None


def _request_path(request: Any) -> str:
    try:
        url = getattr(request, "url", None)
        if url is None:
            return ""
        path = getattr(url, "path", None)
        if isinstance(path, str):
            return path
        return str(url)
    except Exception:
        return ""


def _is_llm_endpoint(path: str) -> bool:
    if not path:
        return False
    for suffix in _LLM_PATH_SUFFIXES:
        if path.endswith(suffix):
            return True
    return False


def _operation_for_path(path: str) -> str:
    if path.endswith("/complete"):
        return "text_completion"
    return "chat"  # /v1/messages and variants


def _resolve_model_from_request(request: Any) -> Optional[str]:
    """Best-effort: pull ``model`` from the JSON request body."""
    try:
        content = getattr(request, "content", None)
        if content is None or not isinstance(content, (bytes, bytearray)):
            return None
        import json
        body = json.loads(content.decode("utf-8"))
        if isinstance(body, dict):
            model = body.get("model")
            if isinstance(model, str) and model:
                return model
    except Exception:
        return None
    return None


def _server_attrs_from_request(request: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    try:
        url = getattr(request, "url", None)
        if url is None:
            return out
        host = getattr(url, "host", None)
        if isinstance(host, str) and host:
            out["server.address"] = host
        port = getattr(url, "port", None)
        if port is None:
            scheme = getattr(url, "scheme", None)
            port = 443 if scheme == "https" else 80 if scheme == "http" else None
        if isinstance(port, int):
            out["server.port"] = port
    except Exception:
        pass
    return out


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
        parent_span.set_attribute(_FR_HAS_RETRY_ATTEMPT_CHILD_KEY, True)
    except Exception:
        logger.debug("failed to set has_retry_attempt_child on parent", exc_info=True)


def _start_attempt_span(request: Any, parent_span: "trace.Span") -> "trace.Span":
    path = _request_path(request)
    attrs: dict[str, Any] = {
        _FR_SPAN_ROLE_KEY: _FR_SPAN_ROLE_RETRY_ATTEMPT,
        # Match the existing Anthropic instrumentor's gen_ai.system
        # value ("Anthropic" title-case — see
        # ``anthropic/__init__.py`` ``_wrap``). The earlier draft of
        # this handler used lower-case "anthropic" for cross-handler
        # consistency, but that mismatched the upstream Anthropic
        # instrumentor and the fr-system-tests ProviderModel
        # ``gen_ai_system="Anthropic"`` assertion — ST-10.4
        # review-driven fix 2026-05-17. Backend provider grouping is
        # case-sensitive in practice; canonicalisation happens via
        # ``event_provider`` (which is always lower-case in tests).
        "gen_ai.system": "Anthropic",
        "gen_ai.operation.name": _operation_for_path(path),
    }
    model = _resolve_model_from_request(request)
    if model:
        attrs["gen_ai.request.model"] = model
    attrs.update(_server_attrs_from_request(request))

    tracer = trace.get_tracer(__name__, __version__, _tracer_provider)
    parent_ctx = trace.set_span_in_context(parent_span)
    span = tracer.start_span(
        _FR_RETRY_ATTEMPT_SPAN_NAME,
        kind=SpanKind.CLIENT,
        attributes=attrs,
        context=parent_ctx,
    )
    return span


def _finalize_success(span: "trace.Span", response: Any, *, is_streaming: bool = False) -> None:
    """Apply response-side attributes + status.

    For non-streaming responses (regardless of 2xx vs non-2xx), parse
    the JSON body and extract usage tokens + response id + response
    model. Per RETRY_LOOP.md §4.4 token-usage rule (around line 164):
    wrappers MUST extract usage from the response body whenever it's
    present, regardless of whether the attempt succeeded — some
    failures DO consume tokens and the provider returns usage in the
    error body. §4.5 backend dedup makes the qualifying retry_attempt
    canonical (even single-attempt), so usage must live on this span.

    Streaming responses (``stream=True`` passed to ``send()``) skip
    body reading to avoid consuming the SSE stream before the SDK can
    iterate it. Streaming usage capture is a deferred follow-up.
    """
    try:
        status_code = getattr(response, "status_code", None)
        if isinstance(status_code, int):
            span.set_attribute("http.status_code", status_code)

        # Header-based response id (works for streaming too, both success and error).
        try:
            headers = getattr(response, "headers", None) or {}
            rid = None
            if hasattr(headers, "get"):
                rid = headers.get("request-id") or headers.get("x-request-id")
            if isinstance(rid, str) and rid:
                span.set_attribute("gen_ai.response.id", rid)
        except Exception:
            pass

        # Non-streaming body parse — applies to BOTH 2xx and non-2xx per the
        # §4.4 token-usage rule. ``_extract_usage_from_body`` is fully
        # defensive (missing fields / parse failure / no ``usage`` block all
        # degrade silently).
        if not is_streaming:
            _extract_usage_from_body(span, response)

        if isinstance(status_code, int) and 200 <= status_code < 300:
            span.set_status(Status(StatusCode.OK))
        else:
            err_type = "anthropic.APIStatusError"
            if isinstance(status_code, int):
                if status_code == 429:
                    err_type = "anthropic.RateLimitError"
                elif 500 <= status_code < 600:
                    err_type = "anthropic.InternalServerError"
                elif status_code == 401:
                    err_type = "anthropic.AuthenticationError"
                elif status_code == 403:
                    err_type = "anthropic.PermissionDeniedError"
            span.set_attribute("error.type", err_type)
            span.set_status(Status(StatusCode.ERROR, f"http {status_code}"))
    except Exception:
        logger.debug("failed to set response attrs on anthropic retry_attempt", exc_info=True)


def _extract_usage_from_body(span: "trace.Span", response: Any) -> None:
    """Parse a non-streaming Anthropic response body and copy usage /
    response id / response model attrs to the retry_attempt span.

    Anthropic Messages response: ``{"id": "...", "model": "...",
    "usage": {"input_tokens": N, "output_tokens": M, ...}}``. Legacy
    /v1/complete may not carry usage; tolerate absence.
    """
    try:
        body = response.json()
    except Exception:
        return
    if not isinstance(body, dict):
        return
    try:
        rid = body.get("id")
        if isinstance(rid, str) and rid:
            span.set_attribute("gen_ai.response.id", rid)
        rmodel = body.get("model")
        if isinstance(rmodel, str) and rmodel:
            span.set_attribute("gen_ai.response.model", rmodel)
        usage = body.get("usage")
        if isinstance(usage, dict):
            pt = usage.get("input_tokens")
            ct = usage.get("output_tokens")
            if isinstance(pt, int):
                span.set_attribute("gen_ai.usage.input_tokens", pt)
            if isinstance(ct, int):
                span.set_attribute("gen_ai.usage.output_tokens", ct)
    except Exception:
        logger.debug("failed to extract usage from anthropic response body", exc_info=True)


def _finalize_error(span: "trace.Span", error: BaseException) -> None:
    try:
        error_type = type(error).__name__
        mod = type(error).__module__
        if isinstance(mod, str) and mod and mod != "builtins":
            error_type = f"{mod.split('.')[0]}.{error_type}"
        span.set_attribute("error.type", error_type)
        status_code = getattr(error, "status_code", None) or getattr(
            getattr(error, "response", None), "status_code", None
        )
        if isinstance(status_code, int):
            span.set_attribute("http.status_code", status_code)
        try:
            span.record_exception(error)
        except Exception:
            pass
        span.set_status(Status(StatusCode.ERROR, str(error)))
    except Exception:
        logger.debug("failed to set error attrs on anthropic retry_attempt", exc_info=True)


def _should_emit_for(request: Any) -> bool:
    if _is_suppressed():
        return False
    path = _request_path(request)
    if not _is_llm_endpoint(path):
        return False
    return True


def _sync_send_wrapper(wrapped, instance, args, kwargs):
    """Wraps ``anthropic._base_client.SyncHttpxClientWrapper.send``.

    Direct-SDK wrappers do NOT register tokens in the §4.7.1 framework
    registry (the registry's contract reserves registration for FRAMEWORK
    wrappers — LiteLLM / LangChain / LlamaIndex). They only consult via
    ``is_framework_owned()``. Self-registering would falsely suppress
    concurrent direct-SDK calls on the same OS thread, which manifests
    most visibly under asyncio (multiple tasks sharing one thread).

    Streaming skip (ST-10.4 review-driven 2026-05-17): when
    ``stream=True`` is passed to send, this wrapper SKIPS retry_attempt
    emission. Streaming retry_attempts cannot carry usage (SSE stream
    can't be peeked) but §4.5 dedup would still promote them to the
    canonical LLMUsageEvent, producing zero-token events. Leaving the
    parent ``anthropic.chat`` span as the canonical (it gets full
    usage from the Anthropic streaming wrapper). Streaming retry-loop
    detection is the deferred follow-up
    ``ST-10.4-FOLLOWUP-streaming-usage``.
    """
    request = args[0] if args else kwargs.get("request")
    if request is None or not _should_emit_for(request):
        return wrapped(*args, **kwargs)

    if bool(kwargs.get("stream", False)):
        return wrapped(*args, **kwargs)

    parent = _resolve_parent_span()
    if parent is None:
        return wrapped(*args, **kwargs)

    is_streaming = False
    span = _start_attempt_span(request, parent)
    _set_parent_marker(parent)

    try:
        response = wrapped(*args, **kwargs)
    except BaseException as exc:
        try:
            _finalize_error(span, exc)
        finally:
            try:
                span.end()
            except Exception:
                pass
        raise
    try:
        _finalize_success(span, response, is_streaming=is_streaming)
    finally:
        try:
            span.end()
        except Exception:
            pass
    return response


async def _async_send_wrapper(wrapped, instance, args, kwargs):
    """Wraps ``anthropic._base_client.AsyncHttpxClientWrapper.send``.

    Same no-self-registration contract and streaming-skip as
    ``_sync_send_wrapper``.
    """
    request = args[0] if args else kwargs.get("request")
    if request is None or not _should_emit_for(request):
        return await wrapped(*args, **kwargs)

    if bool(kwargs.get("stream", False)):
        return await wrapped(*args, **kwargs)

    parent = _resolve_parent_span()
    if parent is None:
        return await wrapped(*args, **kwargs)

    is_streaming = False
    span = _start_attempt_span(request, parent)
    _set_parent_marker(parent)

    try:
        response = await wrapped(*args, **kwargs)
    except BaseException as exc:
        try:
            _finalize_error(span, exc)
        finally:
            try:
                span.end()
            except Exception:
                pass
        raise
    try:
        _finalize_success(span, response, is_streaming=is_streaming)
    finally:
        try:
            span.end()
        except Exception:
            pass
    return response


def _has_wrappable_symbol(module_name: str, class_name: str, method_name: str) -> bool:
    try:
        import importlib
        module = importlib.import_module(module_name)
    except Exception:
        return False
    cls = getattr(module, class_name, None)
    if cls is None or not isinstance(cls, type):
        return False
    method = getattr(cls, method_name, None)
    if method is None:
        return False
    if not callable(method):
        return False
    return True


def instrument_retry_emitter(tracer_provider=None) -> None:
    """Install the Anthropic retry_attempt emitter. Idempotent.

    Wraps:
      - anthropic._base_client.SyncHttpxClientWrapper.send
      - anthropic._base_client.AsyncHttpxClientWrapper.send

    ``tracer_provider`` is the TracerProvider retry spans should be
    emitted through. Caller should pass the same provider the parent
    ``AnthropicInstrumentor`` was configured with so retry_attempt
    spans land in the same exporter chain as the anthropic logical
    span. If ``None``, falls back to the global TracerProvider.

    If a private symbol is missing or incompatible, logs a warning and
    skips wrapping that variant. Normal anthropic instrumentation is NOT
    affected.
    """
    global _installed, _tracer_provider
    with _state_lock:
        if _installed:
            return
        _tracer_provider = tracer_provider
        sync_ok = _has_wrappable_symbol(
            _ANTHROPIC_BASE_CLIENT_MODULE, _SYNC_WRAPPER_CLASS, _WRAPPED_METHOD,
        )
        async_ok = _has_wrappable_symbol(
            _ANTHROPIC_BASE_CLIENT_MODULE, _ASYNC_WRAPPER_CLASS, _WRAPPED_METHOD,
        )
        if not sync_ok:
            logger.warning(
                "ST-10.4: anthropic._base_client.%s.%s missing/incompatible; "
                "skipping sync retry_attempt emission. Normal anthropic "
                "instrumentation is unaffected.",
                _SYNC_WRAPPER_CLASS, _WRAPPED_METHOD,
            )
        else:
            try:
                wrap_function_wrapper(
                    _ANTHROPIC_BASE_CLIENT_MODULE,
                    f"{_SYNC_WRAPPER_CLASS}.{_WRAPPED_METHOD}",
                    _sync_send_wrapper,
                )
            except Exception as e:
                logger.warning(
                    "ST-10.4: failed to wrap anthropic sync httpx send (%s); "
                    "retry_attempt emission disabled for sync path",
                    e,
                )
        if not async_ok:
            logger.warning(
                "ST-10.4: anthropic._base_client.%s.%s missing/incompatible; "
                "skipping async retry_attempt emission. Normal anthropic "
                "instrumentation is unaffected.",
                _ASYNC_WRAPPER_CLASS, _WRAPPED_METHOD,
            )
        else:
            try:
                wrap_function_wrapper(
                    _ANTHROPIC_BASE_CLIENT_MODULE,
                    f"{_ASYNC_WRAPPER_CLASS}.{_WRAPPED_METHOD}",
                    _async_send_wrapper,
                )
            except Exception as e:
                logger.warning(
                    "ST-10.4: failed to wrap anthropic async httpx send (%s); "
                    "retry_attempt emission disabled for async path",
                    e,
                )
        _installed = True


def uninstrument_retry_emitter() -> None:
    """Remove the Anthropic retry_attempt wraps. Idempotent."""
    global _installed, _tracer_provider
    with _state_lock:
        if not _installed:
            return
        _tracer_provider = None
        for cls in (_SYNC_WRAPPER_CLASS, _ASYNC_WRAPPER_CLASS):
            try:
                unwrap(f"{_ANTHROPIC_BASE_CLIENT_MODULE}.{cls}", _WRAPPED_METHOD)
            except Exception:
                logger.debug(
                    "ST-10.4: anthropic unwrap of %s.%s failed (likely "
                    "wrap was never installed for this variant)",
                    cls, _WRAPPED_METHOD,
                    exc_info=True,
                )
        _installed = False


def _is_installed_for_test() -> bool:
    with _state_lock:
        return _installed


__all__ = [
    "instrument_retry_emitter",
    "uninstrument_retry_emitter",
    "_FR_RETRY_ATTEMPT_SPAN_NAME",
    "_FR_HAS_RETRY_ATTEMPT_CHILD_KEY",
    "_LLM_PATH_SUFFIXES",
]

"""ST-10.4 retry-aware emission for the OpenAI direct SDK.

Per RETRY_LOOP.md §4.4 OpenAI row + §4.7 suppression discipline +
ST-10.0 hook-table addendum (in phase_st10_retryloop.txt):

  - Hook the SDK-internal private httpx wrapper classes
    ``openai._base_client.SyncHttpxClientWrapper.send`` and
    ``AsyncHttpxClientWrapper.send``. These run ONCE per HTTP attempt
    inside the SDK's internal retry loop, giving us a clean per-attempt
    boundary without globally monkey-patching ``httpx.Client`` (which
    would emit retry_attempt spans for every httpx caller in the
    process — including non-LLM ones).
  - The wrap is INSTANCE-level via ``wrapt.wrap_function_wrapper``
    against the class, so newly-constructed clients automatically pick
    it up. Unwrap is symmetric (``opentelemetry.instrumentation.utils.unwrap``).
  - Private-symbol guard: ``openai._base_client`` is private to the
    SDK and may change between versions. We wrap defensively via
    ``getattr`` + a try/except around the wrap call; if either
    ``SyncHttpxClientWrapper.send`` or ``AsyncHttpxClientWrapper.send``
    is missing or incompatibly typed, we log a warning and skip
    retry-attempt emission for that variant. Normal openai
    instrumentation continues unaffected.
  - Endpoint allow-list: only LLM endpoints emit retry_attempt spans
    (``/v1/chat/completions``, ``/v1/completions``, ``/v1/embeddings``,
    ``/v1/responses``, plus Azure ``/openai/deployments/.../chat/completions``).
    Non-LLM SDK traffic (e.g. ``/v1/models`` for token-refresh / model
    listing) does NOT emit retry_attempt — per §4.4.1 allow-listing
    requirement.
  - Suppression discipline (§4.7): before emitting, check BOTH the
    OTel context ``SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY`` AND
    the shared ``is_framework_owned()`` registry. If either says
    "suppress", we skip emission. This is the key invariant that
    prevents LiteLLM/LangChain/LlamaIndex framework retries from
    DOUBLE-emitting when both framework wrappers and direct-SDK
    wrappers are active.
  - Parent resolution: use the current OTel ambient span. If invalid
    or absent, gracefully skip (no orphan retry_attempt). The §4.5
    backend dedup degrades gracefully.
  - Per-attempt span lifetime: open on wrap entry, end on wrap exit
    (success OR exception). On non-2xx HTTP, set
    ``http.status_code`` + ``error.type`` and ERROR status; on 2xx,
    set best-effort response attrs + OK status.

Tests: see ``tests/test_retry_attempt_emission.py``.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Optional

from opentelemetry import context as context_api
from opentelemetry import trace
from opentelemetry.instrumentation.fortifyroot import (
    FR_HAS_ATTEMPT_CHILD_KEY,
    is_framework_owned,
    llm_attempt_attributes,
    next_llm_attempt,
)
from opentelemetry.instrumentation.openai.version import __version__
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.semconv_ai import SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY
from opentelemetry.trace import SpanKind, Status, StatusCode
from wrapt import wrap_function_wrapper

logger = logging.getLogger(__name__)


# ST-10 §4.4 / §4.5 constants.
_FR_SPAN_ROLE_KEY = "fortifyroot.span.role"
_FR_LLM_ATTEMPT_SPAN_NAME_PREFIX = "fortifyroot.openai"
_FR_SPAN_ROLE_LLM_ATTEMPT = "llm_attempt"
_FR_HAS_ATTEMPT_CHILD_KEY = FR_HAS_ATTEMPT_CHILD_KEY

# Package-local context key set by the OpenAI logical-call wrappers
# (currently `chat_wrappers.chat_wrapper` / `achat_wrapper`) BEFORE
# they invoke the wrapped OpenAI SDK call. Used by ``_is_suppressed()``
# to distinguish:
#
#   - "openai wrapper is making its OWN internal HTTP attempt under
#      its OWN logical span" — retry_attempt MUST emit (the openai
#      wrapper sets ``SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY``
#      to protect against OTHER instrumentors double-counting; we are
#      NOT one of those — we are part of the openai instrumentor).
#
#   - "external suppression" (user set the key globally, OR a framework
#     wrapper set it to disable LLM instrumentation for a scope) —
#     retry_attempt MUST be suppressed.
#
# The override key is read via ``context_api.get_value`` and is True
# only inside the chat_wrapper's `wrapped(*args, **kwargs)` call. See
# ``opentelemetry.instrumentation.openai.shared.chat_wrappers``.
OPENAI_DIRECT_RETRY_PARENT_ACTIVE_KEY = "fortifyroot.openai.direct_retry_parent_active"

# Private symbols we wrap. Recorded here as module-level constants so the
# instrument/uninstrument code paths share one source of truth and the
# private-symbol guard tests can introspect them.
_OPENAI_BASE_CLIENT_MODULE = "openai._base_client"
_SYNC_WRAPPER_CLASS = "SyncHttpxClientWrapper"
_ASYNC_WRAPPER_CLASS = "AsyncHttpxClientWrapper"
_WRAPPED_METHOD = "send"

# §4.4.1 endpoint allow-list: matched against the request URL's path
# suffix. Hits any of these → LLM endpoint → emit retry_attempt.
# Misses → non-LLM SDK traffic (e.g. /v1/models, auth refresh) →
# skip emission.
_LLM_PATH_SUFFIXES = (
    "/chat/completions",   # /v1/chat/completions, Azure /openai/deployments/.../chat/completions
    "/completions",        # /v1/completions (legacy completion); also matches /chat/completions
    "/embeddings",         # /v1/embeddings
    "/responses",          # /v1/responses (Responses API)
    "/messages",           # /v1/messages (Azure-style messages endpoints, OpenAI parity layer)
)


# Tracks whether instrumentation is currently installed, so the
# private-symbol guard test path can verify symmetry without double-wrap
# / double-unwrap explosions when called by ``instrument()`` machinery
# (the BaseInstrumentor singleton has its own state, but this module
# also exposes a direct install/uninstall for tests).
_state_lock = threading.Lock()
_installed = False

# Tracer provider the wrap was installed with. ``None`` → fall back to
# the global ``TracerProvider`` (matches behaviour pre-2026-05-16). Set
# via ``instrument_retry_emitter(tracer_provider=...)`` so retry spans
# go to the SAME provider as the openai logical span emitted by
# ``chat_wrapper`` etc. Otherwise a consumer that passes an explicit
# provider to ``OpenAIInstrumentor().instrument(tracer_provider=...)``
# gets the logical openai span on their provider but the retry_attempt
# span sent into a no-op tracer (when there is no global provider).
_tracer_provider = None


def _request_path(request: Any) -> str:
    """Return the URL path of an httpx.Request, or '' if unavailable.

    The OpenAI SDK passes an ``httpx.Request`` as the first positional
    argument to ``SyncHttpxClientWrapper.send``. ``httpx.URL.path`` is
    the percent-decoded path.
    """
    try:
        url = getattr(request, "url", None)
        if url is None:
            return ""
        path = getattr(url, "path", None)
        if isinstance(path, str):
            return path
        # Older httpx versions: stringify and rely on the .path attribute
        # being populated. If not present, fall back to str(url) which
        # includes the path.
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
    """Map a URL path to a ``gen_ai.operation.name`` value."""
    if path.endswith("/embeddings"):
        return "embeddings"
    if path.endswith("/chat/completions") or path.endswith("/messages"):
        return "chat"
    if path.endswith("/responses"):
        return "chat"  # Responses API is chat-shaped
    if path.endswith("/completions"):
        return "text_completion"
    return "chat"


def _resolve_model_from_request(request: Any) -> Optional[str]:
    """Best-effort: pull the ``model`` field from the request body. The
    OpenAI SDK serialises the JSON payload before constructing the
    httpx.Request, so ``request.content`` is bytes carrying the JSON.

    On failure (binary streamed body, non-JSON, parse error), return
    None — we omit the attribute rather than guessing.
    """
    try:
        content = getattr(request, "content", None)
        if content is None or not isinstance(content, (bytes, bytearray)):
            return None
        # Cheap, defensive JSON decode.
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
    """Extract ``server.address`` and ``server.port`` from an httpx.Request."""
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
    """§4.7: skip emission if EITHER suppression signal is active.

    Subtle: the OpenAI ``chat_wrapper`` (and ``achat_wrapper``) sets
    ``SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY`` around its own
    wrapped SDK call to protect against OTHER (external) LLM
    instrumentors double-counting the same call. That is NOT the
    contract our retry handler should respect — we are part of the
    openai instrumentor itself and want to emit retry_attempt under
    the openai logical span. The chat_wrapper signals this case by
    ALSO setting ``OPENAI_DIRECT_RETRY_PARENT_ACTIVE_KEY`` in the
    same context; we treat that as an override.

    For all OTHER scenarios where ``SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY``
    is True without the override key (user explicitly disabled LLM
    instrumentation, framework wrapper suppressing direct-SDK, etc.),
    suppression still applies.
    """
    try:
        if context_api.get_value(SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY):
            # Override: openai-wrapper-internal suppression — emit anyway.
            if not context_api.get_value(OPENAI_DIRECT_RETRY_PARENT_ACTIVE_KEY):
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
    """§4.5: mark the parent as 'has llm_attempt child' AFTER the
    first child has successfully started. Idempotent — setting twice
    is a no-op."""
    try:
        parent_span.set_attribute(_FR_HAS_ATTEMPT_CHILD_KEY, True)
    except Exception:
        logger.debug("failed to set has_attempt_child on parent", exc_info=True)


def _start_attempt_span(request: Any, parent_span: "trace.Span") -> "trace.Span":
    """Open the llm_attempt sibling span under the given parent.
    Caller is responsible for adding response attrs + ending it.
    """
    path = _request_path(request)
    span_name, attempt_number, is_retry = next_llm_attempt(
        parent_span,
        _FR_LLM_ATTEMPT_SPAN_NAME_PREFIX,
    )
    attrs: dict[str, Any] = {
        "gen_ai.system": "openai",
        "gen_ai.operation.name": _operation_for_path(path),
    }
    attrs.update(llm_attempt_attributes(attempt_number, is_retry))
    model = _resolve_model_from_request(request)
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


def _finalize_success(span: "trace.Span", response: Any, *, is_streaming: bool = False) -> None:
    """Apply response-side attributes + status. Treats non-2xx HTTP
    responses as errors even when no exception is raised — httpx does
    NOT raise on non-2xx by default; the OpenAI SDK's high-level layers
    raise above us, but at the wrap layer we see the raw response.

    For non-streaming responses (regardless of 2xx vs non-2xx), parse
    the JSON body and extract usage tokens + response id + response
    model. Per RETRY_LOOP.md §4.4 token-usage rule (around line 164):
    wrappers MUST extract usage from the response body whenever it's
    present, regardless of whether the attempt succeeded — some
    failures (context-length-exceeded errors etc.) DO consume tokens
    and the provider returns usage in the error body. Backend §4.5
    dedup makes a qualifying retry_attempt canonical (even single-
    attempt) and reads token attrs from it
    (``fr-backend/internal/processing/proc_llm_extractor.go``
    ``isRetryAttempt`` block) — so usage attribution must live here.

    For streaming responses (``stream=True`` passed to ``send()``),
    body reading would consume the SSE stream before the SDK can
    iterate it. Usage attrs are therefore OMITTED on streaming
    retry_attempts in this MVP; full streaming usage capture would
    require intercepting the SSE chunk stream and is tracked as a
    follow-up.
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
                rid = headers.get("x-request-id") or headers.get("openai-request-id")
            if isinstance(rid, str) and rid:
                span.set_attribute("gen_ai.response.id", rid)
        except Exception:
            pass

        # Non-streaming body parse — applies to BOTH 2xx and non-2xx per the
        # §4.4 token-usage rule. ``_extract_usage_from_body`` is fully
        # defensive: missing fields, parse failure, and bodies without
        # ``usage`` all degrade silently to "no attrs set".
        if not is_streaming:
            _extract_usage_from_body(span, response)

        # Status + error attribution.
        if isinstance(status_code, int) and 200 <= status_code < 300:
            span.set_status(Status(StatusCode.OK))
        else:
            err_type = "openai.HTTPStatusError"
            if isinstance(status_code, int):
                if status_code == 429:
                    err_type = "openai.RateLimitError"
                elif 500 <= status_code < 600:
                    err_type = "openai.InternalServerError"
                elif status_code == 401:
                    err_type = "openai.AuthenticationError"
                elif status_code == 403:
                    err_type = "openai.PermissionDeniedError"
            span.set_attribute("error.type", err_type)
            span.set_status(Status(StatusCode.ERROR, f"http {status_code}"))
    except Exception:
        logger.debug("failed to set response attrs on openai llm_attempt", exc_info=True)


def _extract_usage_from_body(span: "trace.Span", response: Any) -> None:
    """Parse a non-streaming OpenAI response body and copy usage /
    response id / response model attrs to the llm_attempt span.

    All operations are wrapped in try/except — body parsing is
    best-effort. If anything fails (non-JSON body, malformed schema,
    missing fields), the span is left without those attrs rather than
    raising.
    """
    try:
        # ``.json()`` triggers ``.read()`` if the body isn't already
        # buffered, then parses. Subsequent SDK reads hit the cached
        # ``_content``.
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
            # OpenAI chat / completions schema: prompt_tokens / completion_tokens.
            # Newer Responses API uses input_tokens / output_tokens — tolerate both.
            pt = usage.get("prompt_tokens")
            if not isinstance(pt, int):
                pt = usage.get("input_tokens")
            ct = usage.get("completion_tokens")
            if not isinstance(ct, int):
                ct = usage.get("output_tokens")
            if isinstance(pt, int):
                span.set_attribute("gen_ai.usage.input_tokens", pt)
            if isinstance(ct, int):
                span.set_attribute("gen_ai.usage.output_tokens", ct)
    except Exception:
        logger.debug("failed to extract usage from openai response body", exc_info=True)


def _finalize_error(span: "trace.Span", error: BaseException) -> None:
    """Apply error-side attributes + status."""
    try:
        error_type = type(error).__name__
        # Fully-qualified type name when available.
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
        logger.debug("failed to set error attrs on openai llm_attempt", exc_info=True)


def _should_emit_for(request: Any) -> bool:
    """Combine all skip-emission guards into one decision. Returns
    True iff an llm_attempt span SHOULD be emitted for this request.
    """
    if _is_suppressed():
        return False
    path = _request_path(request)
    if not _is_llm_endpoint(path):
        return False
    return True


def _sync_send_wrapper(wrapped, instance, args, kwargs):
    """Wraps ``openai._base_client.SyncHttpxClientWrapper.send``.

    The ``send(request, **kwargs)`` signature is the standard httpx one;
    ``request`` is always the first positional argument; ``stream`` (bool)
    is passed by the OpenAI SDK when the high-level call is a streaming
    request.

    Framework-attempt registry: this wrapper does NOT register a token.
    The §4.7.1 registry's documented contract (see ``retry_registry.py``
    docstring) reserves registration for FRAMEWORK wrappers (LiteLLM /
    LangChain / LlamaIndex). Direct-SDK wrappers only CONSULT via
    ``is_framework_owned()``. Self-registering would falsely suppress
    other concurrent direct-SDK calls on the same OS thread — a real
    issue under asyncio where two tasks share a thread.

    Streaming skip (ST-10.4 review-driven 2026-05-17): when
    ``stream=True`` is passed to send, this wrapper SKIPS retry_attempt
    emission entirely. Rationale: streaming retry_attempts cannot
    carry token usage (the body is the SSE stream and reading it would
    break the SDK), but backend §4.5 dedup makes any qualifying
    retry_attempt the canonical LLMUsageEvent — producing a zero-token
    canonical event for streaming calls. By not emitting at all, the
    parent ``openai.chat`` span (which DOES get full usage attribution
    from ``ChatStream``'s stream-completion callback) remains the
    canonical event. Streaming retry-loop detection is the documented
    deferred follow-up ``ST-10.4-FOLLOWUP-streaming-usage``; this
    explicit skip is part of that deferral.
    """
    request = args[0] if args else kwargs.get("request")
    if request is None or not _should_emit_for(request):
        return wrapped(*args, **kwargs)

    # ST-10.4: skip retry_attempt emission for streaming calls.
    if bool(kwargs.get("stream", False)):
        return wrapped(*args, **kwargs)

    parent = _resolve_parent_span()
    if parent is None:
        return wrapped(*args, **kwargs)

    is_streaming = False  # only reached on non-streaming path
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
    """Wraps ``openai._base_client.AsyncHttpxClientWrapper.send``.

    Same no-self-registration contract and streaming-skip behaviour as
    ``_sync_send_wrapper``.
    """
    request = args[0] if args else kwargs.get("request")
    if request is None or not _should_emit_for(request):
        return await wrapped(*args, **kwargs)

    # ST-10.4: skip retry_attempt emission for streaming (see
    # _sync_send_wrapper docstring for rationale).
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
    """Import-time wrappability check. Returns True iff the named symbol
    exists, is a class, and has the named method as a function-like
    attribute. Used by ``instrument_retry_emitter`` to skip wrapping
    when private OpenAI symbols are missing or incompatible (without
    breaking normal instrumentation).
    """
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
    """Install the OpenAI retry_attempt emitter. Idempotent — calling
    twice without an intervening uninstrument is a no-op.

    Wraps:
      - openai._base_client.SyncHttpxClientWrapper.send
      - openai._base_client.AsyncHttpxClientWrapper.send

    ``tracer_provider`` is the TracerProvider retry spans should be
    emitted through. Caller should pass the same provider the parent
    ``OpenAIInstrumentor`` was configured with so retry_attempt spans
    land in the same exporter chain as the openai logical span. If
    ``None``, falls back to the global TracerProvider (pre-2026-05-16
    behaviour; correct when the consumer has set a global provider).

    If a private symbol is missing or incompatible (newer/older openai
    SDK has refactored ``_base_client``), logs a warning and skips
    wrapping that variant. Normal openai instrumentation is NOT
    affected.
    """
    global _installed, _tracer_provider
    with _state_lock:
        if _installed:
            return
        _tracer_provider = tracer_provider
        sync_ok = _has_wrappable_symbol(
            _OPENAI_BASE_CLIENT_MODULE, _SYNC_WRAPPER_CLASS, _WRAPPED_METHOD,
        )
        async_ok = _has_wrappable_symbol(
            _OPENAI_BASE_CLIENT_MODULE, _ASYNC_WRAPPER_CLASS, _WRAPPED_METHOD,
        )
        if not sync_ok:
            logger.warning(
                "ST-10.4: openai._base_client.%s.%s missing/incompatible; "
                "skipping sync retry_attempt emission. Normal openai "
                "instrumentation is unaffected.",
                _SYNC_WRAPPER_CLASS, _WRAPPED_METHOD,
            )
        else:
            try:
                wrap_function_wrapper(
                    _OPENAI_BASE_CLIENT_MODULE,
                    f"{_SYNC_WRAPPER_CLASS}.{_WRAPPED_METHOD}",
                    _sync_send_wrapper,
                )
            except Exception as e:
                logger.warning(
                    "ST-10.4: failed to wrap openai sync httpx send (%s); "
                    "retry_attempt emission disabled for sync path",
                    e,
                )
        if not async_ok:
            logger.warning(
                "ST-10.4: openai._base_client.%s.%s missing/incompatible; "
                "skipping async retry_attempt emission. Normal openai "
                "instrumentation is unaffected.",
                _ASYNC_WRAPPER_CLASS, _WRAPPED_METHOD,
            )
        else:
            try:
                wrap_function_wrapper(
                    _OPENAI_BASE_CLIENT_MODULE,
                    f"{_ASYNC_WRAPPER_CLASS}.{_WRAPPED_METHOD}",
                    _async_send_wrapper,
                )
            except Exception as e:
                logger.warning(
                    "ST-10.4: failed to wrap openai async httpx send (%s); "
                    "retry_attempt emission disabled for async path",
                    e,
                )
        _installed = True


def uninstrument_retry_emitter() -> None:
    """Remove the OpenAI retry_attempt wraps. Idempotent — calling
    twice or before install is a no-op.
    """
    global _installed, _tracer_provider
    with _state_lock:
        if not _installed:
            return
        _tracer_provider = None
        for cls in (_SYNC_WRAPPER_CLASS, _ASYNC_WRAPPER_CLASS):
            try:
                unwrap(f"{_OPENAI_BASE_CLIENT_MODULE}.{cls}", _WRAPPED_METHOD)
            except Exception:
                logger.debug(
                    "ST-10.4: openai unwrap of %s.%s failed (likely "
                    "wrap was never installed for this variant)",
                    cls, _WRAPPED_METHOD,
                    exc_info=True,
                )
        _installed = False


def _is_installed_for_test() -> bool:
    """Test-only helper: True iff retry-emitter wraps are currently
    installed. Not part of the public contract; do not import outside
    tests."""
    with _state_lock:
        return _installed


__all__ = [
    "instrument_retry_emitter",
    "uninstrument_retry_emitter",
    "_FR_LLM_ATTEMPT_SPAN_NAME_PREFIX",
    "_FR_HAS_ATTEMPT_CHILD_KEY",
    "_LLM_PATH_SUFFIXES",
]

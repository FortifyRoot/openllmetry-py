from __future__ import annotations

import asyncio
import copy
import logging
import threading
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from opentelemetry.trace import Span

SAFETY_EVENT_NAME = "fortifyroot.safety.violation"
GEN_AI_SYSTEM = "gen_ai.system"
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"

logger = logging.getLogger(__name__)


class SafetyDecision(str, Enum):
    ALLOW = "ALLOW"
    MASK = "MASK"


class SafetyLocation(str, Enum):
    PROMPT = "PROMPT"
    COMPLETION = "COMPLETION"


@dataclass(frozen=True, slots=True)
class SafetyFinding:
    """A single safety match emitted by a prompt or completion scan."""

    category: str
    severity: str
    action: str
    rule_name: str
    start: int | str
    end: int | str


@dataclass(frozen=True, slots=True)
class SafetyContext:
    """Context passed to registered prompt and completion safety handlers."""

    provider: str
    text: str
    location: SafetyLocation
    span_name: str
    request_type: str | None = None
    segment_index: int | None = None
    segment_role: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SafetyResult:
    """Normalized handler output returned to provider integrations."""

    text: str
    findings: Sequence[SafetyFinding] = ()
    overall_action: str = SafetyDecision.ALLOW.value


PromptSafetyHandler = Callable[[SafetyContext], SafetyResult | None]
CompletionSafetyHandler = Callable[[SafetyContext], SafetyResult | None]


HANDLER_LOCK = threading.RLock()
_handler_lock = HANDLER_LOCK
_prompt_handler: PromptSafetyHandler | None = None
_completion_handler: CompletionSafetyHandler | None = None


def build_safety_metadata(
    metadata: Mapping[str, Any] | None = None,
    *,
    provider: Any | None = None,
    request_model: Any | None = None,
    response_model: Any | None = None,
) -> dict[str, Any]:
    """Build raw LLM context metadata for safety handlers.

    These are raw semantic-convention attributes. FortifyRoot backend owns
    canonical provider-role enrichment, so SDK instrumentation should not emit
    routing_provider or billing_provider directly.
    """

    attrs = dict(metadata or {})
    _set_metadata_if_present(attrs, GEN_AI_SYSTEM, provider)
    _set_metadata_if_present(attrs, GEN_AI_REQUEST_MODEL, request_model)
    _set_metadata_if_present(attrs, GEN_AI_RESPONSE_MODEL, response_model)
    return attrs


def _set_metadata_if_present(attrs: dict[str, Any], key: str, value: Any | None) -> None:
    if value is None:
        return
    text = str(value).strip()
    if text:
        attrs.setdefault(key, text)


def register_prompt_safety_handler(handler: PromptSafetyHandler | None) -> None:
    """Register the global prompt safety handler."""

    global _prompt_handler
    with _handler_lock:
        _prompt_handler = handler


def register_completion_safety_handler(handler: CompletionSafetyHandler | None) -> None:
    """Register the global completion safety handler."""

    global _completion_handler
    with _handler_lock:
        _completion_handler = handler


def clear_safety_handlers() -> None:
    """Clear all registered FortifyRoot safety hooks."""

    from opentelemetry.instrumentation.fortifyroot.streaming import (
        clear_completion_safety_stream_factory,
    )

    register_prompt_safety_handler(None)
    register_completion_safety_handler(None)
    clear_completion_safety_stream_factory()


def get_prompt_safety_handler() -> PromptSafetyHandler | None:
    """Return the currently registered prompt safety handler, if any."""

    with _handler_lock:
        return _prompt_handler


def get_completion_safety_handler() -> CompletionSafetyHandler | None:
    """Return the currently registered completion safety handler, if any."""

    with _handler_lock:
        return _completion_handler


def run_prompt_safety(
    *,
    span: Span | None,
    provider: str,
    span_name: str,
    text: str | None,
    location: SafetyLocation,
    request_type: str | None = None,
    segment_index: int | None = None,
    segment_role: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> SafetyResult | None:
    """Execute prompt safety for provider request text."""

    return _run_safety(
        handler=get_prompt_safety_handler(),
        span=span,
        provider=provider,
        span_name=span_name,
        text=text,
        location=location,
        request_type=request_type,
        segment_index=segment_index,
        segment_role=segment_role,
        metadata=metadata,
    )


def run_completion_safety(
    *,
    span: Span | None,
    provider: str,
    span_name: str,
    text: str | None,
    location: SafetyLocation,
    request_type: str | None = None,
    segment_index: int | None = None,
    segment_role: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> SafetyResult | None:
    """Execute completion safety for provider response text."""

    return _run_safety(
        handler=get_completion_safety_handler(),
        span=span,
        provider=provider,
        span_name=span_name,
        text=text,
        location=location,
        request_type=request_type,
        segment_index=segment_index,
        segment_role=segment_role,
        metadata=metadata,
    )


async def run_prompt_safety_async(
    *,
    span: Span | None,
    provider: str,
    span_name: str,
    text: str | None,
    location: SafetyLocation,
    request_type: str | None = None,
    segment_index: int | None = None,
    segment_role: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> SafetyResult | None:
    """Execute prompt safety off the event loop via asyncio.to_thread."""

    handler = get_prompt_safety_handler()
    if handler is None or text is None or text == "":
        return None
    return await asyncio.to_thread(
        _run_safety,
        handler=handler,
        span=span,
        provider=provider,
        span_name=span_name,
        text=text,
        location=location,
        request_type=request_type,
        segment_index=segment_index,
        segment_role=segment_role,
        metadata=metadata,
    )


async def run_completion_safety_async(
    *,
    span: Span | None,
    provider: str,
    span_name: str,
    text: str | None,
    location: SafetyLocation,
    request_type: str | None = None,
    segment_index: int | None = None,
    segment_role: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> SafetyResult | None:
    """Execute completion safety off the event loop via asyncio.to_thread."""

    handler = get_completion_safety_handler()
    if handler is None or text is None or text == "":
        return None
    return await asyncio.to_thread(
        _run_safety,
        handler=handler,
        span=span,
        provider=provider,
        span_name=span_name,
        text=text,
        location=location,
        request_type=request_type,
        segment_index=segment_index,
        segment_role=segment_role,
        metadata=metadata,
    )


def clone_value(value: Any) -> Any:
    """Best-effort deep copy that falls back to the original object."""

    try:
        return copy.deepcopy(value)
    except Exception:
        return value


def get_object_value(obj: Any, key: str, default: Any = None) -> Any:
    """Read a field from either a mapping or attribute-based object."""

    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def set_object_value(obj: Any, key: str, value: Any) -> bool:
    """Write a field onto either a mapping or attribute-based object."""

    if isinstance(obj, MutableMapping):
        obj[key] = value
        return True
    try:
        setattr(obj, key, value)
        return True
    except Exception:
        return False


def _run_safety(
    *,
    handler: Callable[[SafetyContext], SafetyResult | None] | None,
    span: Span | None,
    provider: str,
    span_name: str,
    text: str | None,
    location: SafetyLocation,
    request_type: str | None,
    segment_index: int | None,
    segment_role: str | None,
    metadata: Mapping[str, Any] | None,
) -> SafetyResult | None:
    if handler is None or text is None or text == "":
        return None

    context = SafetyContext(
        provider=provider,
        text=text,
        location=location,
        span_name=span_name,
        request_type=request_type,
        segment_index=segment_index,
        segment_role=segment_role,
        metadata=metadata or {},
    )
    try:
        result = handler(context)
    except Exception:
        logger.warning("Safety handler execution failed", exc_info=True)
        return None
    if result is None:
        return None

    normalized = _normalize_result(text, result)
    _emit_findings(span, context, normalized)
    return normalized


def _normalize_result(original_text: str, result: SafetyResult) -> SafetyResult:
    text = result.text if result.text is not None else original_text
    findings = tuple(_normalize_finding(finding) for finding in result.findings)
    overall_action = _normalize_decision(result.overall_action)
    return SafetyResult(text=text, findings=findings, overall_action=overall_action)


def _normalize_finding(finding: SafetyFinding) -> SafetyFinding:
    return SafetyFinding(
        category=str(finding.category).upper(),
        severity=str(finding.severity).upper(),
        action=_normalize_decision(finding.action),
        rule_name=finding.rule_name,
        start=int(finding.start),
        end=int(finding.end),
    )


def _normalize_decision(value: str) -> str:
    raw = str(value).strip().upper()
    if raw == SafetyDecision.MASK.value:
        return SafetyDecision.MASK.value
    return SafetyDecision.ALLOW.value


# ---------------------------------------------------------------------------
# FR: Deferred finding buffer — thread-local queue for safety findings that
# are produced when no valid OTel span is current (e.g. LangChain/LlamaIndex
# safety pre-wrappers run before the callback handler creates the span).
#
# Thread-local so concurrent requests on different threads don't interfere.
#
# Four operations on the buffer:
#
#   discard   — throw away all findings (data lost). Use at wrapper entry
#               to prevent stale findings from a prior request leaking.
#
#   emit      — write all findings as span events on a given span, then
#               clear the buffer. Use after the real span is created.
#
#   drain     — remove and return all findings from THIS thread's buffer.
#               Use on a worker thread (asyncio.to_thread) to extract
#               findings before returning to the calling thread.
#
#   inject    — append findings INTO this thread's buffer. Use on the
#               calling thread to re-insert findings drained from a worker.
#
# Async pattern (drain + inject):
#   Worker threads created by asyncio.to_thread get their own thread-local
#   buffer. Callers must drain on the worker and inject on the event-loop
#   thread so that the callback handler's emit() sees the findings.
# ---------------------------------------------------------------------------

_deferred = threading.local()


def discard_deferred_findings() -> None:
    """Throw away all buffered findings on this thread — data is lost.

    Call at the START of a new safety wrapper invocation to prevent stale
    findings from a prior request on the same thread from leaking into the
    current request's span.
    """
    _deferred.items = []


def emit_deferred_findings(span: Span | None) -> None:
    """Write all buffered findings as span events on *span*, then clear.

    Each finding becomes a ``fortifyroot.safety.violation`` span event.
    Safe to call when the buffer is empty or *span* is None/ended (no-op).
    """
    items: list[dict[str, Any]] = getattr(_deferred, "items", [])
    _deferred.items = []
    if not span or not span.is_recording() or not items:
        return
    for attrs in items:
        span.add_event(SAFETY_EVENT_NAME, attributes=attrs)


def drain_deferred_findings() -> list[dict[str, Any]]:
    """Remove and return all findings from this thread's buffer.

    Use on a worker thread after safety runs, then pass the returned list
    to ``inject_deferred_findings()`` on the calling thread::

        def _on_worker():
            apply_safety(...)
            return drain_deferred_findings()

        findings = await asyncio.to_thread(_on_worker)
        inject_deferred_findings(findings)
    """
    items: list[dict[str, Any]] = getattr(_deferred, "items", [])
    _deferred.items = []
    return items


def inject_deferred_findings(items: list[dict[str, Any]]) -> None:
    """Append findings (from another thread) into this thread's buffer.

    Counterpart to ``drain_deferred_findings()``.  No-op if *items* is
    empty.
    """
    if not items:
        return
    if not hasattr(_deferred, "items"):
        _deferred.items = []
    _deferred.items.extend(items)


def _emit_findings(
    span: Span | None,
    context: SafetyContext,
    result: SafetyResult,
) -> None:
    if not result.findings:
        return

    attrs_list = []
    for finding in result.findings:
        attributes: dict[str, Any] = {
            "fortifyroot.safety.category": finding.category,
            "fortifyroot.safety.severity": finding.severity,
            "fortifyroot.safety.action": finding.action,
            "fortifyroot.safety.location": context.location.value,
            "fortifyroot.safety.rule_name": finding.rule_name,
            "fortifyroot.safety.start": finding.start,
            "fortifyroot.safety.end": finding.end,
        }
        if context.segment_index is not None:
            attributes["fortifyroot.safety.segment_index"] = context.segment_index
        if context.segment_role:
            attributes["fortifyroot.safety.segment_role"] = context.segment_role
        attrs_list.append(attributes)

    # If the span is valid and recording, emit immediately
    if span is not None and span.is_recording():
        for attributes in attrs_list:
            span.add_event(SAFETY_EVENT_NAME, attributes=attributes)
        return

    # Otherwise buffer for deferred emission
    if not hasattr(_deferred, "items"):
        _deferred.items = []
    _deferred.items.extend(attrs_list)

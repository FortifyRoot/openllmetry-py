"""FR: Per-chunk streaming safety for LlamaIndex LLM streaming responses.

Uses the FR holdback algorithm (CompletionTextStreamGroup) to detect patterns
that span chunk boundaries in real time -- e.g. an email address split across
several yielded deltas.  Each chunk's ``delta`` field is passed through the
holdback buffer before being yielded to the caller.

This file is FR-authored and does not exist in upstream openllmetry.
"""
from __future__ import annotations

import time
from typing import Any, Generator

from opentelemetry.instrumentation.fortifyroot.text_streaming import (
    CompletionTextStreamGroup,
)

PROVIDER = "LlamaIndex"
FR_STREAMING_TIME_TO_FIRST_TOKEN_MS = "fortifyroot.llm.streaming.time_to_first_token_ms"
FR_STREAMING_TIME_TO_GENERATE_MS = "fortifyroot.llm.streaming.time_to_generate_ms"


class LlamaIndexStreamingSafety:
    """Per-call streaming safety evaluator backed by CompletionTextStreamGroup.

    One instance is created per streaming LLM call.  It owns the holdback
    buffer for that call and routes each delta through it.
    """

    def __init__(self, span: Any, span_name: str, request_type: str) -> None:
        self._span = span
        self._stream_started_at = time.perf_counter()
        self._first_token_at: float | None = None
        self._streams = CompletionTextStreamGroup(
            span=span,
            provider=PROVIDER,
            span_name=span_name,
            request_type=request_type,
        )

    def process_delta(
        self,
        delta: str,
        *,
        segment_index: int = 0,
        segment_role: str = "assistant",
    ) -> str:
        """Pass one chunk delta through the holdback buffer.

        Returns the portion of text that is safe to release now (may be empty
        if the buffer is still accumulating) or the masked text if a finding
        was fully matched and lies within the release boundary.
        """
        return self._streams.process(
            key=segment_index,
            text=delta,
            segment_index=segment_index,
            segment_role=segment_role,
        )

    def flush(self, *, segment_index: int = 0) -> str:
        """Release all remaining held-back text at stream end.

        Must be called exactly once per segment after the generator is
        exhausted, to ensure the trailing holdback_chars are evaluated and
        any findings in them are masked/emitted.
        """
        return self._streams.flush(key=segment_index) or ""

    def record_first_token(self) -> None:
        """Record TTFT once, when the first output-bearing chunk arrives."""
        if self._first_token_at is not None:
            return
        self._first_token_at = time.perf_counter()
        _set_streaming_latency_attribute(
            self._span,
            FR_STREAMING_TIME_TO_FIRST_TOKEN_MS,
            self._first_token_at - self._stream_started_at,
        )

    def record_completion(self) -> None:
        """Record STTG if at least one output-bearing chunk arrived."""
        if self._first_token_at is None:
            return
        _set_streaming_latency_attribute(
            self._span,
            FR_STREAMING_TIME_TO_GENERATE_MS,
            time.perf_counter() - self._first_token_at,
        )


# ---------------------------------------------------------------------------
# Internal helpers (best-effort, never raise)
# ---------------------------------------------------------------------------

def _patch_delta(response: Any, safety: LlamaIndexStreamingSafety) -> None:
    """Mutate response.delta in-place via the safety holdback buffer."""
    delta = getattr(response, "delta", None)
    if isinstance(delta, str) and delta:
        try:
            response.delta = safety.process_delta(delta)
        except Exception:
            pass


def _flush_into_last(response: Any, safety: LlamaIndexStreamingSafety) -> None:
    """Append any held-back tail to the last response's delta."""
    try:
        tail = safety.flush()
    except Exception:
        return
    if not tail:
        return
    try:
        response.delta = (getattr(response, "delta", None) or "") + tail
    except Exception:
        pass


def _set_streaming_latency_attribute(span: Any, name: str, seconds: float) -> None:
    if span is not None and span.is_recording() and seconds is not None:
        span.set_attribute(name, int(round(max(0, seconds) * 1000)))


def _response_has_output_delta(response: Any) -> bool:
    delta = getattr(response, "delta", None)
    return isinstance(delta, str) and delta != ""


# ---------------------------------------------------------------------------
# Sync streaming wrapper
# ---------------------------------------------------------------------------

def wrap_stream(gen: Generator, safety: LlamaIndexStreamingSafety) -> Generator:
    """Wrap a sync ChatResponseGen / CompletionResponseGen with streaming safety.

    Uses a pending-item pattern: each response object is held until the next
    one arrives so that flush() output can be appended to the very last item
    before it is yielded.
    """
    pending = None
    for response in gen:
        if pending is not None:
            yield pending
        if _response_has_output_delta(response):
            safety.record_first_token()
        _patch_delta(response, safety)
        pending = response
    if pending is not None:
        _flush_into_last(pending, safety)
        safety.record_completion()
        yield pending


# ---------------------------------------------------------------------------
# Async streaming wrapper
# ---------------------------------------------------------------------------

def make_async_stream(agen: Any, safety: LlamaIndexStreamingSafety) -> Any:
    """Return an async generator that applies streaming safety to each chunk.

    ``agen`` must be an async iterable of LlamaIndex response objects that
    carry a ``delta`` attribute (ChatResponse or CompletionResponse).

    Returns an async generator object directly so the caller can write:
        async for response in make_async_stream(agen, safety): ...
    """

    async def _gen():
        pending = None
        async for response in agen:
            if pending is not None:
                yield pending
            if _response_has_output_delta(response):
                safety.record_first_token()
            _patch_delta(response, safety)
            pending = response
        if pending is not None:
            _flush_into_last(pending, safety)
            safety.record_completion()
            yield pending

    return _gen()

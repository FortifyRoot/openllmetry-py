"""FR: Tests for LlamaIndex per-chunk streaming safety.

Covers:
- wrap_stream / make_async_stream helpers in streaming_safety.py
- llm_stream_chat_wrapper / llm_astream_chat_wrapper in safety.py
- llm_stream_complete_wrapper / llm_astream_complete_wrapper in safety.py
- dispatcher_wrapper.py: apply_chat_end_safety / apply_completion_end_safety
  are skipped when waiting_for_streaming=True (no duplicate span events)
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    CompletionResponse,
    MessageRole,
)
from llama_index.core.instrumentation.events.llm import (
    LLMChatEndEvent,
    LLMCompletionEndEvent,
)

from opentelemetry.instrumentation.fortifyroot import (
    SafetyFinding,
    SafetyLocation,
    SafetyResult,
    clear_safety_handlers,
    register_prompt_safety_handler,
)
from opentelemetry.instrumentation.llamaindex.dispatcher_wrapper import SpanHolder
from opentelemetry.instrumentation.llamaindex.safety import (
    llm_stream_chat_wrapper,
    llm_stream_complete_wrapper,
    llm_astream_chat_wrapper,
    llm_astream_complete_wrapper,
    uninstrument_llm_safety_wrappers,
)
from opentelemetry.instrumentation.llamaindex.streaming_safety import (
    LlamaIndexStreamingSafety,
    make_async_stream,
    wrap_stream,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

pytestmark = pytest.mark.fr


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeLLM:
    class __class__:
        __name__ = "FakeLLM"


def _chat_response(delta: str | None) -> ChatResponse:
    return ChatResponse(
        message=ChatMessage(role=MessageRole.ASSISTANT, content=delta or ""),
        delta=delta,
    )


def _completion_response(delta: str | None) -> CompletionResponse:
    return CompletionResponse(text=delta or "", delta=delta)


def _mask_all(context) -> SafetyResult | None:
    if context.location == SafetyLocation.PROMPT:
        return SafetyResult(
            text="[MASKED]",
            overall_action="MASK",
            findings=[SafetyFinding("PII", "HIGH", "MASK", "test", 0, len(context.text))],
        )
    return None


def _mock_safety(*, process_side_effect=None, flush_return="") -> LlamaIndexStreamingSafety:
    safety = MagicMock(spec=LlamaIndexStreamingSafety)
    if process_side_effect is not None:
        safety.process_delta.side_effect = process_side_effect
    else:
        safety.process_delta.side_effect = lambda delta, **kw: delta
    safety.flush.return_value = flush_return
    return safety


def _test_span():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer(__name__)
    return exporter, tracer


def setup_function():
    clear_safety_handlers()
    uninstrument_llm_safety_wrappers()


def teardown_function():
    clear_safety_handlers()
    uninstrument_llm_safety_wrappers()


# ===========================================================================
# streaming_safety.py: wrap_stream (sync)
# ===========================================================================

class TestWrapStream:
    def setup_method(self):
        clear_safety_handlers()
        uninstrument_llm_safety_wrappers()

    def teardown_method(self):
        clear_safety_handlers()
        uninstrument_llm_safety_wrappers()
    def test_passthrough_when_no_delta(self):
        """Responses with None delta are yielded unchanged."""
        responses = [_chat_response(None), _chat_response(None)]
        safety = _mock_safety()
        result = list(wrap_stream(iter(responses), safety))
        assert len(result) == 2
        safety.process_delta.assert_not_called()

    def test_empty_generator_yields_nothing(self):
        safety = _mock_safety()
        result = list(wrap_stream(iter([]), safety))
        assert result == []
        safety.flush.assert_not_called()

    def test_single_chunk_flush_appended_to_delta(self):
        """flush() tail is appended to the last chunk's delta."""
        r = _chat_response("hello")
        safety = _mock_safety(flush_return=" world")
        result = list(wrap_stream(iter([r]), safety))
        assert len(result) == 1
        assert result[0].delta == "hello world"
        safety.flush.assert_called_once()

    def test_delta_masked_by_process(self):
        """process_delta return value replaces the original delta."""
        responses = [_chat_response("secret"), _chat_response(" more")]
        safety = _mock_safety(
            process_side_effect=lambda d, **kw: d.replace("secret", "[PII]"),
            flush_return="",
        )
        result = list(wrap_stream(iter(responses), safety))
        assert result[0].delta == "[PII]"
        assert result[1].delta == " more"

    def test_multi_chunk_flush_appended_only_to_last(self):
        """flush() tail goes onto the last chunk, not earlier ones."""
        responses = [_chat_response("a"), _chat_response("b"), _chat_response("c")]
        safety = _mock_safety(flush_return="TAIL")
        result = list(wrap_stream(iter(responses), safety))
        assert result[0].delta == "a"
        assert result[1].delta == "b"
        assert result[2].delta == "cTAIL"

    def test_exception_in_process_delta_does_not_crash(self):
        """If process_delta raises, the original delta is kept and iteration continues."""
        r1 = _chat_response("first")
        r2 = _chat_response("second")
        safety = _mock_safety()
        safety.process_delta.side_effect = RuntimeError("boom")
        result = list(wrap_stream(iter([r1, r2]), safety))
        # The wrapper catches the error; original deltas survive unchanged
        assert len(result) == 2
        assert result[0].delta == "first"
        assert result[1].delta == "second"

    def test_exception_in_flush_does_not_crash(self):
        """If flush raises, the last chunk is still yielded."""
        r = _chat_response("text")
        safety = _mock_safety()
        safety.flush.side_effect = RuntimeError("boom")
        result = list(wrap_stream(iter([r]), safety))
        assert len(result) == 1
        assert result[0].delta == "text"

    def test_completion_response_delta_patched(self):
        """Works with CompletionResponse (stream_complete path)."""
        r = _completion_response("secret")
        safety = _mock_safety(
            process_side_effect=lambda d, **kw: "[MASKED]",
            flush_return="",
        )
        result = list(wrap_stream(iter([r]), safety))
        assert result[0].delta == "[MASKED]"

    def test_pending_item_pattern_yields_every_item(self):
        """All N items are yielded (not N-1) when flush tail is empty."""
        responses = [_chat_response(f"t{i}") for i in range(5)]
        safety = _mock_safety(flush_return="")
        result = list(wrap_stream(iter(responses), safety))
        assert len(result) == 5


# ===========================================================================
# streaming_safety.py: make_async_stream
# ===========================================================================

class TestMakeAsyncStream:
    def setup_method(self):
        clear_safety_handlers()
        uninstrument_llm_safety_wrappers()

    def teardown_method(self):
        clear_safety_handlers()
        uninstrument_llm_safety_wrappers()
    @pytest.mark.asyncio
    async def test_basic_async_passthrough(self):
        async def _source():
            for d in ["a", "b", "c"]:
                yield _chat_response(d)

        safety = _mock_safety(flush_return="")
        result = []
        async for r in make_async_stream(_source(), safety):
            result.append(r.delta)
        assert result == ["a", "b", "c"]

    @pytest.mark.asyncio
    async def test_async_delta_masked(self):
        async def _source():
            yield _chat_response("secret")
            yield _chat_response(" data")

        safety = _mock_safety(
            process_side_effect=lambda d, **kw: d.replace("secret", "[PII]"),
            flush_return="",
        )
        result = []
        async for r in make_async_stream(_source(), safety):
            result.append(r.delta)
        assert result == ["[PII]", " data"]

    @pytest.mark.asyncio
    async def test_async_flush_appended_to_last(self):
        async def _source():
            yield _chat_response("hello")

        safety = _mock_safety(flush_return=" tail")
        result = []
        async for r in make_async_stream(_source(), safety):
            result.append(r.delta)
        assert result == ["hello tail"]

    @pytest.mark.asyncio
    async def test_empty_async_generator(self):
        async def _source():
            return
            yield  # pragma: no cover

        safety = _mock_safety()
        result = [r async for r in make_async_stream(_source(), safety)]
        assert result == []
        safety.flush.assert_not_called()


# ===========================================================================
# safety.py: llm_stream_chat_wrapper
# ===========================================================================

class TestStreamChatWrapper:
    def setup_method(self):
        clear_safety_handlers()
        uninstrument_llm_safety_wrappers()

    def teardown_method(self):
        clear_safety_handlers()
        uninstrument_llm_safety_wrappers()
    def test_prompt_safety_applied_before_wrapped_call(self):
        register_prompt_safety_handler(_mask_all)
        captured = {}

        def wrapped(messages, **kwargs):
            captured["messages"] = messages
            return iter([])

        messages = [ChatMessage(content="secret", role=MessageRole.USER)]
        list(llm_stream_chat_wrapper(wrapped, _FakeLLM(), (messages,), {}))

        assert captured["messages"][0].content == "[MASKED]"
        assert messages[0].content == "secret"  # original untouched

    def test_returns_generator(self):
        import types
        def wrapped(messages, **kwargs):
            return iter([_chat_response("hi")])

        result = llm_stream_chat_wrapper(wrapped, _FakeLLM(), ([],), {})
        assert isinstance(result, types.GeneratorType)

    def test_chunks_yielded_with_safety_applied(self):
        """Full integration: stream wrapper applies safety to each delta."""
        with patch(
            "opentelemetry.instrumentation.llamaindex.streaming_safety.LlamaIndexStreamingSafety",
            autospec=True,
        ) as MockSafety:
            instance = MockSafety.return_value
            instance.process_delta.side_effect = lambda d, **kw: d.replace("x", "Y")
            instance.flush.return_value = ""

            def wrapped(messages, **kwargs):
                yield _chat_response("ax")
                yield _chat_response("bx")

            result = list(llm_stream_chat_wrapper(wrapped, _FakeLLM(), ([],), {}))

        assert result[0].delta == "aY"
        assert result[1].delta == "bY"

    def test_no_prompt_handler_passes_original_messages(self):
        captured = {}

        def wrapped(messages, **kwargs):
            captured["messages"] = messages
            return iter([])

        messages = [ChatMessage(content="hello", role=MessageRole.USER)]
        list(llm_stream_chat_wrapper(wrapped, _FakeLLM(), (messages,), {}))
        assert captured["messages"][0].content == "hello"


# ===========================================================================
# safety.py: llm_stream_complete_wrapper
# ===========================================================================

class TestStreamCompleteWrapper:
    def setup_method(self):
        clear_safety_handlers()
        uninstrument_llm_safety_wrappers()

    def teardown_method(self):
        clear_safety_handlers()
        uninstrument_llm_safety_wrappers()
    def test_prompt_safety_applied(self):
        register_prompt_safety_handler(_mask_all)
        captured = {}

        def wrapped(prompt, **kwargs):
            captured["prompt"] = prompt
            return iter([])

        list(llm_stream_complete_wrapper(wrapped, _FakeLLM(), ("secret",), {}))
        assert captured["prompt"] == "[MASKED]"

    def test_completion_deltas_processed(self):
        with patch(
            "opentelemetry.instrumentation.llamaindex.streaming_safety.LlamaIndexStreamingSafety",
            autospec=True,
        ) as MockSafety:
            instance = MockSafety.return_value
            instance.process_delta.side_effect = lambda d, **kw: "[DONE]"
            instance.flush.return_value = ""

            def wrapped(prompt, **kwargs):
                yield _completion_response("raw")

            result = list(llm_stream_complete_wrapper(wrapped, _FakeLLM(), ("prompt",), {}))

        assert result[0].delta == "[DONE]"


# ===========================================================================
# safety.py: llm_astream_chat_wrapper (async)
# ===========================================================================

class TestAStreamChatWrapper:
    def setup_method(self):
        clear_safety_handlers()
        uninstrument_llm_safety_wrappers()

    def teardown_method(self):
        clear_safety_handlers()
        uninstrument_llm_safety_wrappers()
    @pytest.mark.asyncio
    async def test_prompt_safety_applied(self):
        register_prompt_safety_handler(_mask_all)
        captured = {}

        async def wrapped(messages, **kwargs):
            captured["messages"] = messages
            return
            yield  # pragma: no cover  -- make it an async gen func pattern

        # Simulate the LI pattern: async def returning an async gen
        async def wrapped_coro(messages, **kwargs):
            captured["messages"] = messages

            async def _gen():
                return
                yield  # pragma: no cover

            return _gen()

        messages = [ChatMessage(content="secret", role=MessageRole.USER)]
        agen = await llm_astream_chat_wrapper(wrapped_coro, _FakeLLM(), (messages,), {})
        async for _ in agen:
            pass
        assert captured["messages"][0].content == "[MASKED]"

    @pytest.mark.asyncio
    async def test_returns_async_generator(self):
        import collections.abc

        async def wrapped_coro(messages, **kwargs):
            async def _gen():
                yield _chat_response("hi")

            return _gen()

        agen = await llm_astream_chat_wrapper(wrapped_coro, _FakeLLM(), ([],), {})
        assert isinstance(agen, collections.abc.AsyncGenerator)

    @pytest.mark.asyncio
    async def test_handles_async_gen_function_directly(self):
        """astream_chat may be an async gen function (not a coroutine)."""
        import collections.abc

        async def wrapped_agen(messages, **kwargs):
            yield _chat_response("chunk1")
            yield _chat_response("chunk2")

        # wrapped_agen(messages) returns an async generator (not a coroutine)
        agen = await llm_astream_chat_wrapper(wrapped_agen, _FakeLLM(), ([],), {})
        assert isinstance(agen, collections.abc.AsyncGenerator)
        results = [r async for r in agen]
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_delta_masked_through_safety(self):
        with patch(
            "opentelemetry.instrumentation.llamaindex.streaming_safety.LlamaIndexStreamingSafety",
            autospec=True,
        ) as MockSafety:
            instance = MockSafety.return_value
            instance.process_delta.side_effect = lambda d, **kw: "[M]"
            instance.flush.return_value = ""

            async def wrapped_coro(messages, **kwargs):
                async def _gen():
                    yield _chat_response("secret")

                return _gen()

            agen = await llm_astream_chat_wrapper(wrapped_coro, _FakeLLM(), ([],), {})
            results = [r async for r in agen]

        assert results[0].delta == "[M]"


# ===========================================================================
# safety.py: llm_astream_complete_wrapper (async)
# ===========================================================================

class TestAStreamCompleteWrapper:
    def setup_method(self):
        clear_safety_handlers()
        uninstrument_llm_safety_wrappers()

    def teardown_method(self):
        clear_safety_handlers()
        uninstrument_llm_safety_wrappers()
    @pytest.mark.asyncio
    async def test_prompt_safety_applied(self):
        register_prompt_safety_handler(_mask_all)
        captured = {}

        async def wrapped_coro(prompt, **kwargs):
            captured["prompt"] = prompt

            async def _gen():
                return
                yield  # pragma: no cover

            return _gen()

        agen = await llm_astream_complete_wrapper(
            wrapped_coro, _FakeLLM(), ("secret",), {}
        )
        async for _ in agen:
            pass
        assert captured["prompt"] == "[MASKED]"

    @pytest.mark.asyncio
    async def test_completion_delta_masked(self):
        with patch(
            "opentelemetry.instrumentation.llamaindex.streaming_safety.LlamaIndexStreamingSafety",
            autospec=True,
        ) as MockSafety:
            instance = MockSafety.return_value
            instance.process_delta.side_effect = lambda d, **kw: "[C]"
            instance.flush.return_value = ""

            async def wrapped_coro(prompt, **kwargs):
                async def _gen():
                    yield _completion_response("raw")

                return _gen()

            agen = await llm_astream_complete_wrapper(
                wrapped_coro, _FakeLLM(), ("prompt",), {}
            )
            results = [r async for r in agen]

        assert results[0].delta == "[C]"


# ===========================================================================
# dispatcher_wrapper.py: no duplicate events when streaming
# ===========================================================================

class TestDispatcherStreamingGuard:
    def setup_method(self):
        clear_safety_handlers()
        uninstrument_llm_safety_wrappers()

    def teardown_method(self):
        clear_safety_handlers()
        uninstrument_llm_safety_wrappers()
    def test_chat_end_safety_skipped_when_waiting_for_streaming(self):
        """apply_chat_end_safety must not be called when the span is streaming."""
        _, tracer = _test_span()
        with tracer.start_as_current_span("test") as otel_span:
            holder = SpanHolder(span_id="test-1", otel_span=otel_span)
            holder.waiting_for_streaming = True  # simulates a streaming response

            with patch(
                "opentelemetry.instrumentation.llamaindex.dispatcher_wrapper.apply_chat_end_safety"
            ) as mock_safety:
                holder.update_span_for_event(
                    LLMChatEndEvent(
                        messages=[],
                        response=ChatResponse(
                            message=ChatMessage(role=MessageRole.ASSISTANT, content="text")
                        ),
                        span_id="test-1",
                    )
                )

        mock_safety.assert_not_called()

    def test_chat_end_safety_called_when_not_streaming(self):
        """apply_chat_end_safety IS called for non-streaming LLMChatEndEvent."""
        _, tracer = _test_span()
        with tracer.start_as_current_span("test") as otel_span:
            holder = SpanHolder(span_id="test-2", otel_span=otel_span)
            # waiting_for_streaming defaults to False

            with patch(
                "opentelemetry.instrumentation.llamaindex.dispatcher_wrapper.apply_chat_end_safety"
            ) as mock_safety:
                holder.update_span_for_event(
                    LLMChatEndEvent(
                        messages=[],
                        response=ChatResponse(
                            message=ChatMessage(role=MessageRole.ASSISTANT, content="text")
                        ),
                        span_id="test-2",
                    )
                )

        mock_safety.assert_called_once()

    def test_completion_end_safety_skipped_when_waiting_for_streaming(self):
        _, tracer = _test_span()
        with tracer.start_as_current_span("test") as otel_span:
            holder = SpanHolder(span_id="test-3", otel_span=otel_span)
            holder.waiting_for_streaming = True

            with patch(
                "opentelemetry.instrumentation.llamaindex.dispatcher_wrapper.apply_completion_end_safety"
            ) as mock_safety:
                holder.update_span_for_event(
                    LLMCompletionEndEvent(
                        prompt="p",
                        response=CompletionResponse(text="out"),
                        span_id="test-3",
                    )
                )

        mock_safety.assert_not_called()

    def test_completion_end_safety_called_when_not_streaming(self):
        _, tracer = _test_span()
        with tracer.start_as_current_span("test") as otel_span:
            holder = SpanHolder(span_id="test-4", otel_span=otel_span)

            with patch(
                "opentelemetry.instrumentation.llamaindex.dispatcher_wrapper.apply_completion_end_safety"
            ) as mock_safety:
                holder.update_span_for_event(
                    LLMCompletionEndEvent(
                        prompt="p",
                        response=CompletionResponse(text="out"),
                        span_id="test-4",
                    )
                )

        mock_safety.assert_called_once()


# ===========================================================================
# LlamaIndexStreamingSafety unit tests
# ===========================================================================

class TestLlamaIndexStreamingSafety:
    def setup_method(self):
        clear_safety_handlers()
        uninstrument_llm_safety_wrappers()

    def teardown_method(self):
        clear_safety_handlers()
        uninstrument_llm_safety_wrappers()
    def test_process_delta_delegates_to_streams(self):
        span = MagicMock()
        with patch(
            "opentelemetry.instrumentation.llamaindex.streaming_safety.CompletionTextStreamGroup",
            autospec=True,
        ) as MockGroup:
            mock_group = MockGroup.return_value
            mock_group.process.return_value = "processed"

            safety = LlamaIndexStreamingSafety(span, "test.stream", "CHAT")
            result = safety.process_delta("hello", segment_index=0, segment_role="assistant")

        assert result == "processed"
        mock_group.process.assert_called_once_with(
            key=0, text="hello", segment_index=0, segment_role="assistant"
        )

    def test_flush_delegates_to_streams(self):
        span = MagicMock()
        with patch(
            "opentelemetry.instrumentation.llamaindex.streaming_safety.CompletionTextStreamGroup",
            autospec=True,
        ) as MockGroup:
            mock_group = MockGroup.return_value
            mock_group.flush.return_value = "tail"

            safety = LlamaIndexStreamingSafety(span, "test.stream", "CHAT")
            result = safety.flush(segment_index=0)

        assert result == "tail"
        mock_group.flush.assert_called_once_with(key=0)

    def test_flush_returns_empty_string_when_none(self):
        span = MagicMock()
        with patch(
            "opentelemetry.instrumentation.llamaindex.streaming_safety.CompletionTextStreamGroup",
            autospec=True,
        ) as MockGroup:
            MockGroup.return_value.flush.return_value = None

            safety = LlamaIndexStreamingSafety(span, "test.stream", "CHAT")
            assert safety.flush() == ""

# NOTE:
# This file has been modified by FortifyRoot.
# Original source: https://github.com/traceloop/openllmetry

import inspect
import json
import re
from dataclasses import dataclass, field
from functools import singledispatchmethod
from typing import Any, AsyncGenerator, Dict, Generator, List, Optional

from llama_index.core.agent.workflow.workflow_events import (
    ToolCall as WorkflowToolCall,
)
from llama_index.core.base.response.schema import StreamingResponse
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.instrumentation.event_handlers import BaseEventHandler
from llama_index.core.instrumentation.events import BaseEvent
from llama_index.core.instrumentation.events.agent import AgentToolCallEvent
from llama_index.core.instrumentation.events.chat_engine import (
    StreamChatEndEvent,
)
from llama_index.core.instrumentation.events.embedding import (
    EmbeddingStartEvent,
)
from llama_index.core.instrumentation.events.llm import (
    LLMChatEndEvent,
    LLMChatStartEvent,
    LLMCompletionEndEvent,
    LLMCompletionStartEvent,
    LLMPredictEndEvent,
)
from llama_index.core.instrumentation.events.rerank import ReRankStartEvent
from llama_index.core.instrumentation.span_handlers import BaseSpanHandler
from llama_index.core.workflow import Workflow
from opentelemetry import context as context_api
from opentelemetry.instrumentation.llamaindex.event_emitter import (
    emit_chat_message_events,
    emit_chat_response_events,
    emit_rerank_message_event,
)
from opentelemetry.instrumentation.llamaindex.safety import (
    apply_chat_end_safety,
    apply_completion_end_safety,
    apply_completion_start_span_attributes,
    apply_predict_end_safety,
    instrument_llm_safety_wrappers,
)
from opentelemetry.instrumentation.llamaindex.span_utils import (
    set_embedding,
    set_llm_chat_request,
    set_llm_chat_request_model_attributes,
    set_llm_chat_response,
    set_llm_chat_response_model_attributes,
    set_llm_predict_response,
    set_rerank,
    set_rerank_model_attributes,
    set_tool,
)
from opentelemetry.instrumentation.llamaindex.utils import (
    JSONEncoder,
    should_emit_events,
    should_send_prompts,
)
from opentelemetry.semconv_ai import (
    SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
    SpanAttributes,
    TraceloopSpanKindValues,
)
from opentelemetry.trace import Span, Tracer, set_span_in_context

# For these spans, instead of creating a span using data from LlamaIndex,
# we use the regular OpenLLMetry instrumentations
AVAILABLE_OPENLLMETRY_INSTRUMENTATIONS = ["OpenAI"]

# FR: marker role set on the LlamaIndex "*.workflow" span when it
# delegates full LLM attribute extraction to a child provider span.
# The FR backend LLM-usage extractor skips spans carrying this role so
# we don't double-count the wrapper + the child provider span. Safety
# findings are still emitted on the wrapper (that's where safety fires
# before the child provider span exists), which is why we ALSO need
# the model-attribution helpers below.
_FR_LLM_WRAPPER_ROLE_KEY = "fortifyroot.span.role"
_FR_LLM_WRAPPER_ROLE_VALUE = "llm_wrapper"


def _infer_provider_from_model(model) -> Optional[str]:
    name = str(model or "").lower()
    if "claude" in name or "anthropic" in name:
        return "anthropic"
    if "gpt" in name or name.startswith(("o1", "o3", "o4")):
        return "openai"
    return None


def _stamp_llm_model_for_safety(event: BaseEvent, span) -> None:
    """Set gen_ai.request.model / gen_ai.system on a delegated wrapper
    span so that safety findings emitted on it have model attribution.

    Leaves other attributes (prompts, params, usage) alone — those
    belong to the child provider span. The FR backend LLM-usage
    extractor skips wrapper spans by the ``fortifyroot.span.role``
    marker, so adding gen_ai.request.model here does NOT cause
    double-counting.
    """
    from opentelemetry.semconv._incubating.attributes import (
        gen_ai_attributes as GenAIAttributes,
    )
    try:
        model_dict = event.model_dict or {}
    except Exception:  # pragma: no cover — defensive; event shape may vary
        return
    if "llm" in model_dict:
        model_dict = model_dict.get("llm", {})
    model = model_dict.get("model") if isinstance(model_dict, dict) else None
    if model:
        span.set_attribute(GenAIAttributes.GEN_AI_REQUEST_MODEL, model)
        provider = _infer_provider_from_model(model)
        if provider:
            span.set_attribute(GenAIAttributes.GEN_AI_SYSTEM, provider)


def _stamp_llm_response_model_for_safety(event: BaseEvent, span) -> None:
    """Set gen_ai.response.model on a delegated wrapper span when the
    LLM response carries a model different from the request (e.g.
    Anthropic aliasing ``claude-4-sonnet`` → ``claude-sonnet-4``).

    Used for completion-location safety findings that are emitted on
    the wrapper span after the response has arrived.
    """
    from opentelemetry.semconv._incubating.attributes import (
        gen_ai_attributes as GenAIAttributes,
    )
    response = getattr(event, "response", None)
    raw = getattr(response, "raw", None) if response is not None else None
    model = None
    try:
        if raw is not None:
            model = getattr(raw, "model", None)
            if not model and isinstance(raw, dict):
                model = raw.get("model")
    except Exception:  # pragma: no cover
        model = None
    if model:
        span.set_attribute(GenAIAttributes.GEN_AI_RESPONSE_MODEL, model)

CLASS_ANDMETHOD_NAME_FROM_ID_REGEX = re.compile(r"([a-zA-Z]+)\.([a-zA-Z_]+)-")
STREAMING_END_EVENTS = (
    LLMChatEndEvent,
    LLMCompletionEndEvent,
    StreamChatEndEvent,
)


def instrument_with_dispatcher(tracer: Tracer):
    instrument_llm_safety_wrappers()
    dispatcher = get_dispatcher()
    # Register the FR retry-attempt handler FIRST, before
    # OpenLLMetrySpanHandler. Order matters because span handlers
    # fire in registration order — and our handler reads the
    # ambient OTel context to decide the parent of the
    # retry_attempt span. If OpenLLMetrySpanHandler ran first, it
    # would have already swapped the OTel ambient context to its
    # per-call SpanHolder span, meaning each retry attempt would
    # land under a DIFFERENT parent → siblings invariant broken →
    # RetryDetectorProc grouping fails. Running FR's handler first
    # preserves the user's enclosing OTel span as the shared parent
    # for all retry_attempts under one logical retry loop.
    from opentelemetry.instrumentation.llamaindex.retry_handler import (
        _FortifyRootRetryHandler,
    )
    dispatcher.add_span_handler(_FortifyRootRetryHandler())
    openllmetry_span_handler = OpenLLMetrySpanHandler(tracer)
    dispatcher.add_span_handler(openllmetry_span_handler)
    dispatcher.add_event_handler(OpenLLMetryEventHandler(openllmetry_span_handler))


@dataclass
class SpanHolder:
    span_id: str
    parent: Optional["SpanHolder"] = None
    otel_span: Optional[Span] = None
    token: Optional[Any] = None
    context: Optional[context_api.context.Context] = None
    waiting_for_streaming: bool = field(init=False, default=False)
    # FR: True when the underlying LLM class has its own OpenLLMetry instrumentor
    # (e.g. OpenAI).  Event handlers should skip setting LLM-specific attributes
    # on this span since the provider instrumentor sets them on its own child span.
    delegates_to_provider: bool = field(init=False, default=False)

    _active: bool = field(init=False, default=True)

    def process_event(self, event: BaseEvent) -> List["SpanHolder"]:
        self.update_span_for_event(event)

        if self.waiting_for_streaming and isinstance(event, STREAMING_END_EVENTS):
            self.end()
            return [self] + self.notify_parent()

        return []

    def notify_parent(self) -> List["SpanHolder"]:
        if self.parent:
            self.parent.end()
            return [self.parent] + self.parent.notify_parent()
        return []

    def end(self, should_detach_context: bool = True):
        if not self._active:
            return

        self._active = False
        if self.otel_span:
            self.otel_span.end()
        if self.token and should_detach_context:
            context_api.detach(self.token)

    @singledispatchmethod
    def update_span_for_event(self, event: BaseEvent):
        pass

    @update_span_for_event.register
    def _(self, event: LLMChatStartEvent):
        # FR: skip LLM attribute-setting when provider instrumentor handles it
        if not self.delegates_to_provider:
            set_llm_chat_request_model_attributes(event, self.otel_span)
            if should_emit_events():
                emit_chat_message_events(event)
            else:
                set_llm_chat_request(event, self.otel_span)
        else:
            # FR: even when the provider instrumentor owns the full LLM
            # attribute set on its own child span, we still stamp
            # ``gen_ai.request.model`` + ``gen_ai.system`` on this wrapper
            # span so that safety violation events emitted here (via
            # ``emit_deferred_findings``) carry model attribution. The
            # wrapper span is marked ``fortifyroot.span.role="llm_wrapper"``
            # at creation time so the backend LLM-usage extractor skips
            # it for event counting and avoids double-counting with the
            # child provider span.
            _stamp_llm_model_for_safety(event, self.otel_span)

    @update_span_for_event.register
    def _(self, event: LLMChatEndEvent):
        # FR: skip LLM attribute-setting when provider instrumentor handles it
        if not self.delegates_to_provider:
            # FR: when streaming, per-chunk safety already ran and emitted findings;
            # skip here to avoid duplicate span events on the assembled response.
            if not self.waiting_for_streaming:
                apply_chat_end_safety(event, self.otel_span)
            set_llm_chat_response_model_attributes(event, self.otel_span)
            if should_emit_events():
                emit_chat_response_events(event)
            else:
                set_llm_chat_response(event, self.otel_span)  # noqa: F821
        else:
            # FR: completion-location safety findings (non-streaming) are
            # emitted here for delegated providers too, so we need model
            # attribution on the wrapper span for those events.
            if not self.waiting_for_streaming:
                apply_chat_end_safety(event, self.otel_span)
            _stamp_llm_response_model_for_safety(event, self.otel_span)

    @update_span_for_event.register
    def _(self, event: LLMCompletionStartEvent):
        if not self.delegates_to_provider:
            apply_completion_start_span_attributes(event, self.otel_span)
        else:
            _stamp_llm_model_for_safety(event, self.otel_span)

    @update_span_for_event.register
    def _(self, event: LLMCompletionEndEvent):
        if not self.delegates_to_provider:
            # FR: same as LLMChatEndEvent -- skip when streaming safety is active.
            if not self.waiting_for_streaming:
                apply_completion_end_safety(event, self.otel_span)
        else:
            # FR: delegated-provider completion safety still emits
            # findings on the wrapper span; stamp response model on the
            # wrapper so those findings carry model attribution.
            if not self.waiting_for_streaming:
                apply_completion_end_safety(event, self.otel_span)
            _stamp_llm_response_model_for_safety(event, self.otel_span)

    @update_span_for_event.register
    def _(self, event: LLMPredictEndEvent):
        if not self.delegates_to_provider:
            apply_predict_end_safety(event, self.otel_span)
            if not should_emit_events():
                set_llm_predict_response(event, self.otel_span)

    @update_span_for_event.register
    def _(self, event: EmbeddingStartEvent):
        set_embedding(event, self.otel_span)

    @update_span_for_event.register
    def _(self, event: ReRankStartEvent):
        set_rerank_model_attributes(event, self.otel_span)
        if should_emit_events():
            emit_rerank_message_event(event)
        else:
            set_rerank(event, self.otel_span)

    @update_span_for_event.register
    def _(self, event: AgentToolCallEvent):
        set_tool(event, self.otel_span)


class OpenLLMetrySpanHandler(BaseSpanHandler[SpanHolder]):
    waiting_for_streaming_spans: Dict[str, SpanHolder] = {}
    _tracer: Tracer = PrivateAttr()

    def __init__(self, tracer: Tracer):
        super().__init__()
        self._tracer = tracer

    def new_span(
        self,
        id_: str,
        bound_args: inspect.BoundArguments,
        instance: Optional[Any] = None,
        parent_span_id: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Optional[SpanHolder]:
        """Create a span."""
        # Take the class name and method name from id_ where id_ is e.g.
        # 'SentenceSplitter.split_text_metadata_aware-a2f2a780-2fa6-4682-a88e-80dc1f1ebe6a'
        matches = CLASS_ANDMETHOD_NAME_FROM_ID_REGEX.match(id_)
        class_name = matches.groups()[0]
        method_name = matches.groups()[1]

        parent = self.open_spans.get(parent_span_id)

        # FR: always create a span, even for OpenLLMetry-instrumented classes.
        # Previously this returned early for OpenAI (no span created), which
        # meant safety findings from the pre-wrapper had no valid span to be
        # emitted on.  Now we create a LlamaIndex task/workflow span and set
        # SUPPRESS=False so the provider instrumentor still creates its own
        # child span with full enrichment (tokens, cost, etc.).
        is_openllmetry_class = class_name in AVAILABLE_OPENLLMETRY_INSTRUMENTATIONS

        kind = (
            TraceloopSpanKindValues.TASK.value
            if parent
            else TraceloopSpanKindValues.WORKFLOW.value
        )

        if isinstance(instance, Workflow):
            span_name = (
                f"{instance.__class__.__name__}.{kind}"
                if not parent_span_id
                else f"{method_name}.{kind}"
            )
        else:
            span_name = f"{class_name}.{kind}"

        span = self._tracer.start_span(
            span_name,
            context=parent.context if parent else None,
        )
        current_context = set_span_in_context(
            span, context=parent.context if parent else None
        )
        # FR: for OpenLLMetry classes, set SUPPRESS=False so the provider
        # instrumentor (OpenAI, etc.) still runs and creates its child span.
        current_context = context_api.set_value(
            SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
            not is_openllmetry_class,
            current_context,
        )
        token = context_api.attach(current_context)

        # FR: emit deferred prompt safety findings now that the span exists
        from opentelemetry.instrumentation.fortifyroot import emit_deferred_findings
        emit_deferred_findings(span)

        span.set_attribute(SpanAttributes.TRACELOOP_SPAN_KIND, kind)
        span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_NAME, span_name)
        # FR: mark delegated LlamaIndex wrapper spans so the backend
        # LLM-usage extractor skips them for event counting. Pair with
        # ``_stamp_llm_model_for_safety()`` to keep safety findings
        # emitted on this span (via emit_deferred_findings) correctly
        # attributed to model / provider. See
        # framework safety attribution regression coverage.
        if is_openllmetry_class:
            span.set_attribute(_FR_LLM_WRAPPER_ROLE_KEY, _FR_LLM_WRAPPER_ROLE_VALUE)
        try:
            if should_send_prompts():
                span.set_attribute(
                    SpanAttributes.TRACELOOP_ENTITY_INPUT,
                    json.dumps(bound_args.arguments, cls=JSONEncoder),
                )
        except Exception:
            pass

        # Extract tool information for call_tool spans (workflow-based agents)
        if method_name == "call_tool":
            try:
                # The 'ev' argument is a WorkflowToolCall event
                ev = bound_args.arguments.get("ev")
                if ev and isinstance(ev, WorkflowToolCall):
                    span.set_attribute("tool.name", ev.tool_name)
                    span.set_attribute(
                        "tool.arguments",
                        json.dumps(ev.tool_kwargs, cls=JSONEncoder)
                    )
            except Exception:
                pass

        holder = SpanHolder(id_, parent, span, token, current_context)
        holder.delegates_to_provider = is_openllmetry_class
        return holder

    def prepare_to_exit_span(
        self,
        id_: str,
        instance: Optional[Any] = None,
        result: Optional[Any] = None,
        **kwargs,
    ) -> SpanHolder:
        """Logic for preparing to drop a span."""
        span_holder = self.open_spans[id_]
        # I know it's messy, but the typing of result is messy and couldn't find a better way
        # to get a dictionary I can then use to remove keys
        try:
            serialized_output = json.dumps(result, cls=JSONEncoder)
            # we need to remove some keys like source_nodes as they can be very large
            output = json.loads(serialized_output)
            if "source_nodes" in output:
                del output["source_nodes"]
            if should_send_prompts():
                span_holder.otel_span.set_attribute(
                    SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
                    json.dumps(output, cls=JSONEncoder),
                )
        except Exception:
            pass

        if isinstance(result, (Generator, AsyncGenerator, StreamingResponse)):
            # This is a streaming response, we want to wait for the streaming end event before ending the span
            span_holder.waiting_for_streaming = True
            with self.lock:
                self.waiting_for_streaming_spans[id_] = span_holder
            return span_holder
        else:
            should_detach_context = not isinstance(instance, Workflow)
            span_holder.end(should_detach_context)
            return span_holder

    def prepare_to_drop_span(
        self, id_: str, err: Optional[Exception], **kwargs
    ) -> Optional[SpanHolder]:
        """Logic for dropping a span."""
        if id_ in self.open_spans:
            with self.lock:
                span_holder = self.open_spans[id_]
            return span_holder
        return None


class OpenLLMetryEventHandler(BaseEventHandler):
    _span_handler: OpenLLMetrySpanHandler = PrivateAttr()

    def __init__(self, span_handler: OpenLLMetrySpanHandler):
        super().__init__()
        self._span_handler = span_handler

    def handle(self, event: BaseEvent, **kwargs) -> Any:
        span = self._span_handler.open_spans.get(event.span_id)
        if not span:
            span = self._span_handler.waiting_for_streaming_spans.get(event.span_id)
        if not span:
            print(f"No span found for event {event}")
            return

        finished_spans = span.process_event(event)

        with self._span_handler.lock:
            for span in finished_spans:
                self._span_handler.waiting_for_streaming_spans.pop(span.span_id)

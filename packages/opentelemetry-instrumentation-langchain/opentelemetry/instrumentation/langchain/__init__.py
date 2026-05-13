"""OpenTelemetry Langchain instrumentation"""

import logging
from typing import Any, Collection, Optional

from opentelemetry import context as context_api


from opentelemetry._logs import get_logger
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.langchain.callback_handler import (
    TraceloopCallbackHandler,
)
from opentelemetry.instrumentation.langchain.config import Config
from opentelemetry.instrumentation.langchain.retry_handler import (
    _FortifyRootRetryHandler,
)
from opentelemetry.instrumentation.langchain.safety import (
    instrument_safety_wrappers,
    uninstrument_safety_wrappers,
)
from opentelemetry.instrumentation.langchain.utils import is_package_available
from opentelemetry.instrumentation.langchain.version import __version__
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.metrics import get_meter
from opentelemetry.semconv_ai import Meters, SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY, SpanAttributes
from opentelemetry.trace import get_tracer
from opentelemetry.trace.propagation import set_span_in_context
from opentelemetry.trace.propagation.tracecontext import (
    TraceContextTextMapPropagator,
)
from wrapt import wrap_function_wrapper

logger = logging.getLogger(__name__)

_instruments = ("langchain-core > 0.1.0", )


class LangchainInstrumentor(BaseInstrumentor):
    """An instrumentor for Langchain SDK."""

    def __init__(
        self,
        exception_logger=None,
        disable_trace_context_propagation=False,
        use_legacy_attributes: bool = True,
        metadata_key_prefix: str = SpanAttributes.TRACELOOP_ASSOCIATION_PROPERTIES
    ):
        """Create a Langchain instrumentor instance.

        Args:
            exception_logger: A callable that takes an Exception as input. This will be
                used to log exceptions that occur during instrumentation. If None, exceptions will not be logged.
            disable_trace_context_propagation: If True, disables trace context propagation to LLM providers.
            use_legacy_attributes: If True, uses span attributes for Inputs/Outputs instead of events.
            metadata_key_prefix: Prefix for metadata keys added to spans. Defaults to
                `SpanAttributes.TRACELOOP_ASSOCIATION_PROPERTIES`.
                Useful for using with other backends.
        """
        super().__init__()
        Config.exception_logger = exception_logger
        Config.use_legacy_attributes = use_legacy_attributes
        Config.metadata_key_prefix = metadata_key_prefix
        self.disable_trace_context_propagation = disable_trace_context_propagation

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(__name__, __version__, tracer_provider)

        # Add meter creation
        meter_provider = kwargs.get("meter_provider")
        meter = get_meter(__name__, __version__, meter_provider)

        # Create duration histogram
        duration_histogram = meter.create_histogram(
            name=Meters.LLM_OPERATION_DURATION,
            unit="s",
            description="GenAI operation duration",
        )

        # Create token histogram
        token_histogram = meter.create_histogram(
            name=Meters.LLM_TOKEN_USAGE,
            unit="token",
            description="Measures number of input and output tokens used",
        )

        if not Config.use_legacy_attributes:
            logger_provider = kwargs.get("logger_provider")
            Config.event_logger = get_logger(
                __name__, __version__, logger_provider=logger_provider
            )

        traceloopCallbackHandler = TraceloopCallbackHandler(
            tracer, duration_histogram, token_histogram
        )
        # ST-10.2: register the FR retry-attempt handler alongside the
        # existing Traceloop handler. Per ST-10.0 C2 POC findings, this
        # captures per-HTTP-attempt callbacks on framework-layer retry
        # paths (e.g. Runnable.with_retry) and emits one
        # fortifyroot.langchain.retry_attempt sibling span per attempt
        # under the parent_run_id's span. The handler gets the Traceloop
        # handler as a CONSTRUCTION-time reference so it can resolve the
        # workflow parent via Traceloop's run_id-keyed ``spans`` dict
        # without mutating shared state per-callback-manager-init
        # (review-round-2 Major 5).
        fortifyrootRetryHandler = _FortifyRootRetryHandler(
            traceloop_handler=traceloopCallbackHandler,
        )
        wrap_function_wrapper(
            "langchain_core.callbacks",
            "BaseCallbackManager.__init__",
            _BaseCallbackManagerInitWrapper(
                traceloopCallbackHandler,
                fortifyrootRetryHandler,
            ),
        )
        instrument_safety_wrappers()

        if not self.disable_trace_context_propagation:
            self._wrap_openai_functions_for_tracing(traceloopCallbackHandler)

    def _wrap_openai_functions_for_tracing(self, traceloopCallbackHandler):
        openai_tracing_wrapper = _OpenAITracingWrapper(traceloopCallbackHandler)

        if is_package_available("langchain_community"):
            # Wrap langchain_community.llms.openai.BaseOpenAI
            wrap_function_wrapper(
                "langchain_community.llms.openai",
                "BaseOpenAI._generate",
                openai_tracing_wrapper,
            )

            wrap_function_wrapper(
                "langchain_community.llms.openai",
                "BaseOpenAI._agenerate",
                openai_tracing_wrapper,
            )

            wrap_function_wrapper(
                "langchain_community.llms.openai",
                "BaseOpenAI._stream",
                openai_tracing_wrapper,
            )

            wrap_function_wrapper(
                "langchain_community.llms.openai",
                "BaseOpenAI._astream",
                openai_tracing_wrapper,
            )

        if is_package_available("langchain_openai"):
            # Wrap langchain_openai.llms.base.BaseOpenAI
            wrap_function_wrapper(
                "langchain_openai.llms.base",
                "BaseOpenAI._generate",
                openai_tracing_wrapper,
            )

            wrap_function_wrapper(
                "langchain_openai.llms.base",
                "BaseOpenAI._agenerate",
                openai_tracing_wrapper,
            )

            wrap_function_wrapper(
                "langchain_openai.llms.base",
                "BaseOpenAI._stream",
                openai_tracing_wrapper,
            )

            wrap_function_wrapper(
                "langchain_openai.llms.base",
                "BaseOpenAI._astream",
                openai_tracing_wrapper,
            )

            # langchain_openai.chat_models.base.BaseOpenAI
            wrap_function_wrapper(
                "langchain_openai.chat_models.base",
                "BaseChatOpenAI._generate",
                openai_tracing_wrapper,
            )

            wrap_function_wrapper(
                "langchain_openai.chat_models.base",
                "BaseChatOpenAI._agenerate",
                openai_tracing_wrapper,
            )

            # Doesn't work :(
            # wrap_function_wrapper(
            #     module="langchain_openai.chat_models.base",
            #     name="BaseChatOpenAI._stream",
            #     wrapper=openai_tracing_wrapper,
            # )
            # wrap_function_wrapper(
            #     module="langchain_openai.chat_models.base",
            #     name="BaseChatOpenAI._astream",
            #     wrapper=openai_tracing_wrapper,
            # )

    def _uninstrument(self, **kwargs):
        unwrap("langchain_core.callbacks", "BaseCallbackManager.__init__")
        uninstrument_safety_wrappers()
        if not self.disable_trace_context_propagation:
            if is_package_available("langchain_community"):
                unwrap("langchain_community.llms.openai", "BaseOpenAI._generate")
                unwrap("langchain_community.llms.openai", "BaseOpenAI._agenerate")
                unwrap("langchain_community.llms.openai", "BaseOpenAI._stream")
                unwrap("langchain_community.llms.openai", "BaseOpenAI._astream")
            if is_package_available("langchain_openai"):
                unwrap("langchain_openai.llms.base", "BaseOpenAI._generate")
                unwrap("langchain_openai.llms.base", "BaseOpenAI._agenerate")
                unwrap("langchain_openai.llms.base", "BaseOpenAI._stream")
                unwrap("langchain_openai.llms.base", "BaseOpenAI._astream")
                unwrap("langchain_openai.chat_models.base", "BaseOpenAI._generate")
                unwrap("langchain_openai.chat_models.base", "BaseOpenAI._agenerate")
                # unwrap("langchain_openai.chat_models.base", "BaseOpenAI._stream")
                # unwrap("langchain_openai.chat_models.base", "BaseOpenAI._astream")


class _BaseCallbackManagerInitWrapper:
    def __init__(
        self,
        callback_handler: "TraceloopCallbackHandler",
        retry_handler: "_FortifyRootRetryHandler" = None,
    ):
        self._callback_handler = callback_handler
        self._retry_handler = retry_handler

    def __call__(
        self,
        wrapped,
        instance,
        args,
        kwargs,
    ) -> None:
        wrapped(*args, **kwargs)
        # Register the existing Traceloop handler first (existing
        # behavior, preserved as the canonical order). The FR retry
        # handler is registered AFTER Traceloop and uses parent_run_id
        # to resolve the workflow parent via Traceloop's spans dict
        # — see _resolve_parent_span in retry_handler.py. This avoids
        # the test-isolation bug observed when the FR retry handler
        # was registered FIRST: running before Traceloop caused
        # Traceloop's context-attach/detach discipline to break
        # downstream (LiteLLM tests in the same pytest session
        # observed stale OTel ambient context leaking from LangChain
        # — review-batch-1 v6 trace-id-shared-across-tests bug,
        # 2026-05-11).
        for handler in instance.inheritable_handlers:
            if isinstance(handler, type(self._callback_handler)):
                break
        else:
            # Add a property to the handler which indicates the CallbackManager instance.
            # Since the CallbackHandler only propagates context for sync callbacks,
            # we need a way to determine the type of CallbackManager being wrapped.
            self._callback_handler._callback_manager = instance
            instance.add_handler(self._callback_handler, True)
        # ST-10.2: register the FR retry-attempt handler AFTER
        # Traceloop. Idempotent registration. The handler already
        # holds a CONSTRUCTION-time reference to the Traceloop handler
        # (set in LangchainInstrumentor._instrument). We deliberately
        # do NOT mutate any shared attribute on the retry handler per
        # callback-manager-init — that previously raced when concurrent
        # Runnable invocations created BaseCallbackManagers on
        # different threads (review-round-2 Major 5).
        if self._retry_handler is not None:
            for handler in instance.inheritable_handlers:
                if isinstance(handler, type(self._retry_handler)):
                    break
            else:
                instance.add_handler(self._retry_handler, True)


# This class wraps a function call to inject tracing information (trace headers) into
# OpenAI client requests. It assumes the following:
# 1. The wrapped function includes a `run_manager` keyword argument that contains a `run_id`.
#    The `run_id` is used to look up a corresponding tracing span from the callback manager.
# 2. The `kwargs` passed to the wrapped function are forwarded to the OpenAI client. This
#    allows us to add extra headers (including tracing headers) to the OpenAI request by
#    modifying the `extra_headers` argument in `kwargs`.
class _OpenAITracingWrapper:
    def __init__(self, callback_manager: "TraceloopCallbackHandler"):
        self._callback_manager = callback_manager

    def __call__(
        self,
        wrapped,
        instance,
        args,
        kwargs,
    ) -> None:
        run_manager = kwargs.get("run_manager")
        if run_manager:
            run_id = run_manager.run_id
            span_holder = self._callback_manager.spans.get(run_id)

            if span_holder:
                extra_headers = kwargs.get("extra_headers", {})
                ctx = set_span_in_context(span_holder.span)
                TraceContextTextMapPropagator().inject(extra_headers, context=ctx)
                kwargs["extra_headers"] = extra_headers
            else:
                logger.debug(
                    "No span found for run_id %s, skipping header injection",
                    run_id
                )

        # In legacy chains like LLMChain, suppressing model instrumentations
        # within create_llm_span doesn't work, so this should helps as a fallback.
        #
        # ST-10 review-round-2 fix (2026-05-11): capture the attach token
        # and detach in ``finally`` so this suppression layer doesn't leak
        # into the OTel context stack indefinitely. The pre-fix code
        # never detached, accumulating SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY
        # frames on every wrapped openai call, which compounded across
        # LangChain tests in long pytest sessions and risked corrupting
        # later instrumentor behaviour. See review-round-2 Blocker 3.
        suppression_token: Optional[Any] = None
        try:
            suppression_token = context_api.attach(
                context_api.set_value(SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY, True)
            )
        except Exception:
            # If context setting fails, continue without suppression
            # This is not critical for core functionality
            suppression_token = None

        try:
            return wrapped(*args, **kwargs)
        finally:
            if suppression_token is not None:
                try:
                    context_api.detach(suppression_token)
                except Exception:
                    # Detach can fail in async/concurrent edge cases —
                    # safe to ignore; the context value's lifetime
                    # is bounded by the surrounding context frame anyway.
                    pass

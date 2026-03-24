import logging
from typing import Dict, Optional, Any, cast

from opentelemetry.exporter.otlp.proto.grpc._log_exporter import (
    OTLPLogExporter as GRPCExporter,
)
from opentelemetry.exporter.otlp.proto.http._log_exporter import (
    OTLPLogExporter as HTTPExporter,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk._logs.export import LogExporter, BatchLogRecordProcessor
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler

from opentelemetry.instrumentation.logging import LoggingInstrumentor


_FORTIFYROOT_LOGGING_HANDLER_MARKER = "_fortifyroot_logging_handler"


class LoggerWrapper(object):
    resource_attributes: Dict[Any, Any] = {}
    endpoint: Optional[str] = None
    headers: Dict[str, str] = {}
    __logging_exporter: Optional[LogExporter] = None
    __logging_provider: Optional[LoggerProvider] = None

    def __new__(cls, exporter: Optional[LogExporter] = None) -> "LoggerWrapper":
        if not hasattr(cls, "instance"):
            obj = cls.instance = super(LoggerWrapper, cls).__new__(cls)
            if not LoggerWrapper.endpoint:
                return obj

            obj.__logging_exporter = (
                exporter
                if exporter
                else init_logging_exporter(LoggerWrapper.endpoint, LoggerWrapper.headers)
            )
            obj.__logging_provider = init_logging_provider(
                obj.__logging_exporter, LoggerWrapper.resource_attributes
            )
            instrumentor = LoggingInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument(set_logging_format=True)

        return cls.instance

    @staticmethod
    def set_static_params(
        resource_attributes: dict,
        endpoint: str,
        headers: Dict[str, str],
    ) -> None:
        LoggerWrapper.resource_attributes = resource_attributes
        LoggerWrapper.endpoint = endpoint
        LoggerWrapper.headers = headers

    @classmethod
    def get_logging_provider(cls) -> Optional[LoggerProvider]:
        if not hasattr(cls, "instance"):
            return None
        return cls.instance.__logging_provider


def init_logging_exporter(endpoint: str, headers: Dict[str, str]) -> LogExporter:
    if "http" in endpoint.lower() or "https" in endpoint.lower():
        return cast(LogExporter, HTTPExporter(endpoint=f"{endpoint}/v1/logs", headers=headers))
    else:
        return cast(LogExporter, GRPCExporter(endpoint=endpoint, headers=headers))


def init_logging_provider(
    exporter: LogExporter, resource_attributes: Optional[Dict[Any, Any]] = None
) -> LoggerProvider:
    resource = (
        Resource.create(resource_attributes)
        if resource_attributes
        else Resource.create()
    )

    logger_provider = LoggerProvider(resource=resource)
    logger_provider.add_log_record_processor(BatchLogRecordProcessor(exporter))

    logging_handler = LoggingHandler(level=logging.NOTSET, logger_provider=logger_provider)
    _attach_root_logging_handler(logging_handler)

    return logger_provider


def is_fortifyroot_logging_handler(handler: logging.Handler) -> bool:
    return bool(getattr(handler, _FORTIFYROOT_LOGGING_HANDLER_MARKER, False))


def _attach_root_logging_handler(logging_handler: LoggingHandler) -> None:
    root_logger = logging.getLogger()
    had_non_fortifyroot_handlers = any(
        not is_fortifyroot_logging_handler(handler) for handler in root_logger.handlers
    )

    for handler in list(root_logger.handlers):
        if is_fortifyroot_logging_handler(handler):
            root_logger.removeHandler(handler)

    setattr(logging_handler, _FORTIFYROOT_LOGGING_HANDLER_MARKER, True)
    root_logger.addHandler(logging_handler)

    # Preserve existing app logging levels/formatters when they are already
    # configured. If the app has not configured logging at all, keep the prior
    # default of exporting INFO and above.
    if not had_non_fortifyroot_handlers and root_logger.level > logging.INFO:
        root_logger.setLevel(logging.INFO)


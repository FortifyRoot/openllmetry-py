# NOTE:
# This file has been modified by FortifyRoot.
# Original source: https://github.com/traceloop/openllmetry

import ipaddress
import logging
from typing import Dict, Optional, Any, cast
from urllib.parse import urlparse

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk._logs.export import LogExporter, BatchLogRecordProcessor
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler

from opentelemetry.instrumentation.logging import LoggingInstrumentor
from traceloop.sdk.exporters.auth_warnings import (
    AUTH_WARNING_LOGGER_NAME,
    FortifyRootGRPCLogExporter as GRPCExporter,
    FortifyRootHTTPLogExporter as HTTPExporter,
)
from traceloop.sdk.exporters.headers import grpc_metadata_headers

LOCAL_EXPORT_HOSTS = {"localhost"}


def _is_local_export_host(host: Optional[str]) -> bool:
    if not host:
        return False
    normalized = host.strip().lower()
    if normalized in LOCAL_EXPORT_HOSTS:
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def _resolve_grpc_exporter_endpoint(endpoint: str) -> tuple[str, bool]:
    trimmed_endpoint = endpoint.strip()
    if "://" not in trimmed_endpoint:
        return trimmed_endpoint, False

    parsed = urlparse(trimmed_endpoint)
    if parsed.username or parsed.password:
        raise ValueError("OTLP exporter endpoints must not include username or password")
    scheme = parsed.scheme.lower()
    if scheme == "grpcs":
        return parsed.netloc, False
    if scheme == "grpc":
        if not _is_local_export_host(parsed.hostname):
            raise ValueError(
                "grpc:// OTLP export is insecure and is allowed only for local "
                "development endpoints; use grpcs:// or https:// for remote collectors"
            )
        return parsed.netloc, True

    raise ValueError(
        f"Unsupported OTLP exporter endpoint scheme {parsed.scheme!r}; "
        "use https://, http://, grpcs://, grpc://localhost, or a bare secure gRPC host:port"
    )


def _validate_http_exporter_endpoint(endpoint: str) -> None:
    parsed = urlparse(endpoint.strip())
    if parsed.username or parsed.password:
        raise ValueError("OTLP exporter endpoints must not include username or password")
    if parsed.scheme.lower() != "http":
        return
    if _is_local_export_host(parsed.hostname):
        return
    raise ValueError(
        "http:// OTLP export is insecure and is allowed only for local "
        "development endpoints; use https:// for remote collectors"
    )


_FORTIFYROOT_LOGGING_HANDLER_MARKER = "_fortifyroot_logging_handler"
_FORTIFYROOT_INTERNAL_EXPORTER_LOGGER_PREFIX = "fortifyroot.sdk.exporters."


class _FortifyRootInternalLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if record.name == AUTH_WARNING_LOGGER_NAME:
            return False
        return not record.name.startswith(_FORTIFYROOT_INTERNAL_EXPORTER_LOGGER_PREFIX)


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
    trimmed_endpoint = endpoint.strip()
    scheme = urlparse(trimmed_endpoint).scheme.lower()
    if scheme in {"http", "https"}:
        _validate_http_exporter_endpoint(trimmed_endpoint)
        base_url = trimmed_endpoint.rstrip("/")
        if not base_url.endswith("/v1/logs"):
            base_url = f"{base_url}/v1/logs"
        return cast(LogExporter, HTTPExporter(endpoint=base_url, headers=headers))
    grpc_endpoint, insecure = _resolve_grpc_exporter_endpoint(trimmed_endpoint)
    return cast(
        LogExporter,
        GRPCExporter(
            endpoint=grpc_endpoint,
            headers=grpc_metadata_headers(headers),
            insecure=insecure,
        ),
    )


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
    logging_handler.addFilter(_FortifyRootInternalLogFilter())
    root_logger.addHandler(logging_handler)

    # Preserve existing app logging levels/formatters when they are already
    # configured. If the app has not configured logging at all, keep the prior
    # default of exporting INFO and above.
    if not had_non_fortifyroot_handlers and root_logger.level > logging.INFO:
        root_logger.setLevel(logging.INFO)

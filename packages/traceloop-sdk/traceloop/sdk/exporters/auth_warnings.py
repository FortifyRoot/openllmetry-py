# NOTE:
# This file has been added by FortifyRoot.
# Original project: https://github.com/traceloop/openllmetry

import logging
import threading
import time
from typing import Any, Protocol, cast
from urllib.parse import urlparse

from grpc import RpcError, StatusCode
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import (
    OTLPLogExporter as BaseGRPCLogExporter,
)
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
    OTLPMetricExporter as BaseGRPCMetricExporter,
)
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
    OTLPSpanExporter as BaseGRPCSpanExporter,
)
from opentelemetry.exporter.otlp.proto.http._log_exporter import (
    OTLPLogExporter as BaseHTTPLogExporter,
)
from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
    OTLPMetricExporter as BaseHTTPMetricExporter,
)
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter as BaseHTTPSpanExporter,
)

AUTH_WARNING_LOGGER_NAME = "fortifyroot.sdk.exporters.auth_warnings"

logger = logging.getLogger(AUTH_WARNING_LOGGER_NAME)

_HTTP_AUTH_STATUS_CODES = {401, 403}
_GRPC_AUTH_STATUS_CODES = {
    StatusCode.UNAUTHENTICATED,
    StatusCode.PERMISSION_DENIED,
}
_AUTH_WARNING_TTL_SECONDS = 60 * 60
_WARNED_AUTH_FAILURES: dict[tuple[str, str, str], float] = {}
_WARNED_AUTH_FAILURES_LOCK = threading.Lock()


class _HTTPExporterProtocol(Protocol):
    def _export(self, *args: Any, **kwargs: Any) -> Any: ...


def _endpoint_label(endpoint: Any) -> str:
    raw_endpoint = str(endpoint or "").strip()
    if not raw_endpoint:
        return "unknown endpoint"
    parsed = urlparse(raw_endpoint)
    if parsed.netloc:
        return parsed.netloc
    return raw_endpoint


def _warn_once(signal: str, status: str, endpoint: Any) -> None:
    endpoint_label = _endpoint_label(endpoint)
    key = (signal, status, endpoint_label)
    now = time.monotonic()
    with _WARNED_AUTH_FAILURES_LOCK:
        last_warned_at = _WARNED_AUTH_FAILURES.get(key)
        if (
            last_warned_at is not None
            and now - last_warned_at < _AUTH_WARNING_TTL_SECONDS
        ):
            return
        _WARNED_AUTH_FAILURES[key] = now

    logger.warning(
        "FortifyRoot Ocelle SDK auth warning: %s export was rejected by the configured OTLP endpoint (%s) with %s. "
        "If this endpoint is FortifyRoot, the SDK API key may be invalid, revoked, deleted, or missing permissions. "
        "Telemetry will not reach the OTLP endpoint until a valid credential is configured.",
        signal,
        endpoint_label,
        status,
    )


def _warn_for_http_auth_failure(signal: str, endpoint: Any, response: Any) -> None:
    status_code = getattr(response, "status_code", None)
    if status_code in _HTTP_AUTH_STATUS_CODES:
        _warn_once(signal, f"HTTP {status_code}", endpoint)


def _call_next_http_export(exporter: Any, *args: Any, **kwargs: Any) -> Any:
    next_exporter = cast(
        _HTTPExporterProtocol,
        super(_HTTPAuthWarningMixin, exporter),
    )
    return next_exporter._export(*args, **kwargs)


class _AuthWarningClientProxy:
    def __init__(self, client: Any, signal: str, endpoint: Any) -> None:
        self._client = client
        self._signal = signal
        self._endpoint = endpoint

    def Export(self, *args: Any, **kwargs: Any) -> Any:  # noqa: N802
        try:
            return self._client.Export(*args, **kwargs)
        except RpcError as exc:
            code = exc.code()
            if code in _GRPC_AUTH_STATUS_CODES:
                _warn_once(self._signal, f"gRPC {code.name}", self._endpoint)
            raise

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


class _HTTPAuthWarningMixin:
    _fortifyroot_export_signal = "telemetry"

    def _export(self, *args: Any, **kwargs: Any) -> Any:
        response = _call_next_http_export(self, *args, **kwargs)
        _warn_for_http_auth_failure(
            self._fortifyroot_export_signal,
            getattr(self, "_endpoint", ""),
            response,
        )
        return response


class _GRPCAuthWarningMixin:
    _fortifyroot_export_signal = "telemetry"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        client = getattr(self, "_client", None)
        if client is not None:
            # Upstream OTel currently only invokes `.Export` on `_client`.
            # This proxy intentionally does not subclass the generated stub.
            self._client = _AuthWarningClientProxy(
                client,
                self._fortifyroot_export_signal,
                getattr(self, "_endpoint", ""),
            )


class FortifyRootHTTPSpanExporter(_HTTPAuthWarningMixin, BaseHTTPSpanExporter):
    _fortifyroot_export_signal = "traces"


class FortifyRootGRPCSpanExporter(_GRPCAuthWarningMixin, BaseGRPCSpanExporter):
    _fortifyroot_export_signal = "traces"


class FortifyRootHTTPMetricExporter(_HTTPAuthWarningMixin, BaseHTTPMetricExporter):
    _fortifyroot_export_signal = "metrics"


class FortifyRootGRPCMetricExporter(_GRPCAuthWarningMixin, BaseGRPCMetricExporter):
    _fortifyroot_export_signal = "metrics"


class FortifyRootHTTPLogExporter(_HTTPAuthWarningMixin, BaseHTTPLogExporter):
    _fortifyroot_export_signal = "logs"


class FortifyRootGRPCLogExporter(_GRPCAuthWarningMixin, BaseGRPCLogExporter):
    _fortifyroot_export_signal = "logs"


def reset_auth_warning_state_for_tests() -> None:
    with _WARNED_AUTH_FAILURES_LOCK:
        _WARNED_AUTH_FAILURES.clear()

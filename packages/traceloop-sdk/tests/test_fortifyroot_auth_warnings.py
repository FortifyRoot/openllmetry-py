from __future__ import annotations

import logging
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from grpc import RpcError, StatusCode
from opentelemetry.sdk._logs.export import LogExportResult, LogExporter
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from traceloop.sdk.exporters.auth_warnings import (
    _AuthWarningClientProxy,
    reset_auth_warning_state_for_tests,
)

pytestmark = pytest.mark.fr


class _FakeHTTPResponse:
    ok = False
    reason = "Unauthorized"
    text = "unauthorized"

    def __init__(self, status_code):
        self.status_code = status_code


class _FakeAuthRpcError(RpcError):
    def __init__(self, code):
        super().__init__()
        self._code = code

    def code(self):
        return self._code


class _CapturingLogExporter(LogExporter):
    def __init__(self):
        self.bodies = []

    def export(self, batch):
        for record in batch:
            self.bodies.append(str(record.log_record.body))
        return LogExportResult.SUCCESS

    def shutdown(self):
        return None


def _make_traces_http_exporter(endpoint="http://localhost:4318"):
    from traceloop.sdk.tracing.tracing import init_spans_exporter

    return init_spans_exporter(endpoint, {})


def _make_metrics_http_exporter(endpoint="http://localhost:4318"):
    from traceloop.sdk.metrics.metrics import init_metrics_exporter

    return init_metrics_exporter(endpoint, {})


def _make_logs_http_exporter(endpoint="http://localhost:4318"):
    from traceloop.sdk.logging.logging import init_logging_exporter

    return init_logging_exporter(endpoint, {})


def _make_traces_grpc_exporter(endpoint="grpc://localhost:4317"):
    from traceloop.sdk.tracing.tracing import init_spans_exporter

    return init_spans_exporter(endpoint, {})


def _make_metrics_grpc_exporter(endpoint="localhost:4317"):
    from traceloop.sdk.metrics.metrics import init_metrics_exporter

    return init_metrics_exporter(endpoint, {})


def _make_logs_grpc_exporter(endpoint="localhost:4317"):
    from traceloop.sdk.logging.logging import init_logging_exporter

    return init_logging_exporter(endpoint, {})


def _rejecting_post(status_code):
    def post(*args, **kwargs):
        return _FakeHTTPResponse(status_code)

    return post


def _auth_warnings(caplog):
    return [
        record
        for record in caplog.records
        if "FortifyRoot SDK auth warning" in record.getMessage()
    ]


@pytest.mark.parametrize(
    "factory,signal,status_code",
    [
        (_make_traces_http_exporter, "traces", 401),
        (_make_traces_http_exporter, "traces", 403),
        (_make_metrics_http_exporter, "metrics", 401),
        (_make_metrics_http_exporter, "metrics", 403),
        (_make_logs_http_exporter, "logs", 401),
        (_make_logs_http_exporter, "logs", 403),
    ],
)
def test_http_exporters_warn_on_auth_failure_once(factory, signal, status_code, caplog):
    reset_auth_warning_state_for_tests()
    exporter = factory()
    assert hasattr(exporter, "_session")
    exporter._session.post = _rejecting_post(status_code)

    with caplog.at_level("WARNING"):
        exporter._export(b"payload")
        exporter._export(b"payload")

    auth_warnings = _auth_warnings(caplog)
    assert len(auth_warnings) == 1
    warning = auth_warnings[0].getMessage()
    assert signal in warning
    assert f"HTTP {status_code}" in warning
    assert "invalid, revoked, deleted, or missing permissions" in warning
    assert "Telemetry will not reach the OTLP endpoint" in warning
    assert "Telemetry will not reach FortifyRoot" not in warning


def test_http_exporter_dedupes_by_endpoint(caplog):
    reset_auth_warning_state_for_tests()
    exporter_one = _make_traces_http_exporter("http://collector-one:4318")
    exporter_two = _make_traces_http_exporter("http://collector-two:4318")
    assert hasattr(exporter_one, "_session")
    assert hasattr(exporter_two, "_session")
    exporter_one._session.post = _rejecting_post(401)
    exporter_two._session.post = _rejecting_post(401)

    with caplog.at_level("WARNING"):
        exporter_one._export(b"payload")
        exporter_two._export(b"payload")

    auth_warnings = _auth_warnings(caplog)
    assert len(auth_warnings) == 2
    messages = [record.getMessage() for record in auth_warnings]
    assert any("collector-one:4318" in message for message in messages)
    assert any("collector-two:4318" in message for message in messages)


@pytest.mark.parametrize(
    "factory,signal,code",
    [
        (_make_traces_grpc_exporter, "traces", StatusCode.UNAUTHENTICATED),
        (_make_traces_grpc_exporter, "traces", StatusCode.PERMISSION_DENIED),
        (_make_metrics_grpc_exporter, "metrics", StatusCode.UNAUTHENTICATED),
        (_make_metrics_grpc_exporter, "metrics", StatusCode.PERMISSION_DENIED),
        (_make_logs_grpc_exporter, "logs", StatusCode.UNAUTHENTICATED),
        (_make_logs_grpc_exporter, "logs", StatusCode.PERMISSION_DENIED),
    ],
)
def test_grpc_exporter_client_proxy_warns_on_auth_failure(
    factory,
    signal,
    code,
    caplog,
):
    reset_auth_warning_state_for_tests()

    class FakeClient:
        def Export(self, *args, **kwargs):  # noqa: N802
            raise _FakeAuthRpcError(code)

    exporter = factory()
    assert isinstance(exporter._client, _AuthWarningClientProxy)
    exporter._client._client = FakeClient()

    with caplog.at_level("WARNING"), pytest.raises(_FakeAuthRpcError):
        exporter._client.Export()

    assert "FortifyRoot SDK auth warning" in caplog.text
    assert signal in caplog.text
    assert f"gRPC {code.name}" in caplog.text


def test_trace_http_exporter_warns_end_to_end_with_mock_collector(caplog):
    reset_auth_warning_state_for_tests()

    class RejectingCollector(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            self.send_response(401)
            self.end_headers()
            self.wfile.write(b"unauthorized")

        def log_message(self, format, *args):
            return None

    server = HTTPServer(("127.0.0.1", 0), RejectingCollector)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        endpoint = f"http://127.0.0.1:{server.server_port}"
        exporter = _make_traces_http_exporter(endpoint)
        with caplog.at_level("WARNING"):
            exporter.export([])
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert "FortifyRoot SDK auth warning" in caplog.text
    assert "traces" in caplog.text
    assert "HTTP 401" in caplog.text


def test_auth_warning_log_is_not_reexported_through_otel_logging_handler(caplog):
    reset_auth_warning_state_for_tests()

    from traceloop.sdk import Traceloop
    from traceloop.sdk.logging.logging import (
        LoggerWrapper,
        is_fortifyroot_logging_handler,
    )
    from traceloop.sdk.tracing.tracing import TracerWrapper

    saved_tracer_instance = getattr(TracerWrapper, "instance", None)
    saved_logger_instance = getattr(LoggerWrapper, "instance", None)
    saved_logging_enabled = os.environ.get("TRACELOOP_LOGGING_ENABLED")
    root_logger = logging.getLogger()
    log_exporter = _CapturingLogExporter()

    if hasattr(TracerWrapper, "instance"):
        del TracerWrapper.instance
    if hasattr(LoggerWrapper, "instance"):
        del LoggerWrapper.instance

    try:
        os.environ["TRACELOOP_LOGGING_ENABLED"] = "true"
        Traceloop.init(
            app_name="test-auth-warning-log-filter",
            api_endpoint="http://localhost:4318",
            api_key="fr-test",
            disable_batch=True,
            exporter=InMemorySpanExporter(),
            logging_exporter=log_exporter,
        )
        exporter = _make_traces_http_exporter("http://localhost:4318")
        assert hasattr(exporter, "_session")
        exporter._session.post = _rejecting_post(401)

        with caplog.at_level("WARNING"):
            exporter._export(b"payload")

        provider = LoggerWrapper.get_logging_provider()
        assert provider is not None
        provider.force_flush()

        assert "FortifyRoot SDK auth warning" in caplog.text
        assert not any(
            "FortifyRoot SDK auth warning" in body for body in log_exporter.bodies
        )
    finally:
        for handler in list(root_logger.handlers):
            if is_fortifyroot_logging_handler(handler):
                root_logger.removeHandler(handler)
        if hasattr(LoggerWrapper, "instance"):
            del LoggerWrapper.instance
        if saved_logger_instance is not None:
            LoggerWrapper.instance = saved_logger_instance
        if hasattr(TracerWrapper, "instance"):
            del TracerWrapper.instance
        if saved_tracer_instance is not None:
            TracerWrapper.instance = saved_tracer_instance
        if saved_logging_enabled is None:
            os.environ.pop("TRACELOOP_LOGGING_ENABLED", None)
        else:
            os.environ["TRACELOOP_LOGGING_ENABLED"] = saved_logging_enabled

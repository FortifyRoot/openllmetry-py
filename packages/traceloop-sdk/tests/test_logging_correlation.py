from __future__ import annotations

import logging

import pytest
from opentelemetry.trace import (
    NonRecordingSpan,
    SpanContext,
    TraceFlags,
    TraceState,
    use_span,
)

from traceloop.sdk.logging.logging import (
    init_logging_provider,
)

pytestmark = pytest.mark.fr

try:
    from opentelemetry.sdk._logs.export import LogExportResult as _LogExportResult
except ImportError:  # pragma: no cover - future OTel may move log export APIs.
    _LogExportResult = None


class _CapturingLogExporter:
    def __init__(self) -> None:
        self.records = []

    def export(self, batch):
        for record in batch:
            self.records.append(record.log_record)
        if _LogExportResult is not None:
            return _LogExportResult.SUCCESS
        return None

    def shutdown(self):
        return None


def _restore_root_logger(
    original_handlers: list[logging.Handler],
    original_level: int,
) -> None:
    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
    for handler in original_handlers:
        root_logger.addHandler(handler)
    root_logger.setLevel(original_level)


def test_stdlib_logs_correlate_only_inside_active_span():
    root_logger = logging.getLogger()
    original_handlers = list(root_logger.handlers)
    original_level = root_logger.level
    logger_provider = None

    try:
        exporter = _CapturingLogExporter()
        logger_provider = init_logging_provider(exporter)

        app_logger = logging.getLogger("traceloop.tests.custom_logging")
        app_logger.setLevel(logging.INFO)

        app_logger.info("outside span")

        span_context = SpanContext(
            trace_id=0x1234567890ABCDEF1234567890ABCDEF,
            span_id=0x1234567890ABCDEF,
            is_remote=False,
            trace_flags=TraceFlags(TraceFlags.SAMPLED),
            trace_state=TraceState(),
        )
        with use_span(NonRecordingSpan(span_context), end_on_exit=False):
            app_logger.info("inside span")

        logger_provider.force_flush()

        records_by_body = {record.body: record for record in exporter.records}
        outside = records_by_body["outside span"]
        inside = records_by_body["inside span"]

        assert outside.trace_id == 0
        assert outside.span_id == 0
        assert inside.trace_id == span_context.trace_id
        assert inside.span_id == span_context.span_id
    finally:
        if logger_provider is not None:
            logger_provider.shutdown()
        _restore_root_logger(original_handlers, original_level)

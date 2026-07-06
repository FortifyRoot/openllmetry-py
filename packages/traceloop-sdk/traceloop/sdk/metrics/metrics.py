# NOTE:
# This file has been modified by FortifyRoot.
# Original source: https://github.com/traceloop/openllmetry

import ipaddress
from collections.abc import Sequence
from typing import Dict, Optional, Any
from urllib.parse import urlparse

from opentelemetry.semconv_ai import Meters
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    PeriodicExportingMetricReader,
    MetricExporter,
)
from opentelemetry.sdk.metrics.view import View, ExplicitBucketHistogramAggregation
from opentelemetry.sdk.resources import Resource

from opentelemetry import metrics
from traceloop.sdk.exporters.auth_warnings import (
    FortifyRootGRPCMetricExporter as GRPCExporter,
    FortifyRootHTTPMetricExporter as HTTPExporter,
)

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


class MetricsWrapper(object):
    resource_attributes: Dict[Any, Any] = {}
    endpoint: Optional[str] = None
    # if it needs headers?
    headers: Dict[str, str] = {}
    __metrics_exporter: Optional[MetricExporter] = None
    __metrics_provider: Optional[MeterProvider] = None

    def __new__(cls, exporter: Optional[MetricExporter] = None) -> "MetricsWrapper":
        if not hasattr(cls, "instance"):
            obj = cls.instance = super(MetricsWrapper, cls).__new__(cls)
            if not MetricsWrapper.endpoint:
                return obj

            obj.__metrics_exporter = (
                exporter
                if exporter
                else init_metrics_exporter(
                    MetricsWrapper.endpoint, MetricsWrapper.headers
                )
            )

            obj.__metrics_provider = init_metrics_provider(
                obj.__metrics_exporter, MetricsWrapper.resource_attributes
            )

        return cls.instance

    @staticmethod
    def set_static_params(
        resource_attributes: dict,
        endpoint: str,
        headers: Dict[str, str],
    ) -> None:
        MetricsWrapper.resource_attributes = resource_attributes
        MetricsWrapper.endpoint = endpoint
        MetricsWrapper.headers = headers


def init_metrics_exporter(endpoint: str, headers: Dict[str, str]) -> MetricExporter:
    trimmed_endpoint = endpoint.strip()
    scheme = urlparse(trimmed_endpoint).scheme.lower()
    if scheme in {"http", "https"}:
        _validate_http_exporter_endpoint(trimmed_endpoint)
        base_url = trimmed_endpoint.rstrip("/")
        if not base_url.endswith("/v1/metrics"):
            base_url = f"{base_url}/v1/metrics"
        return HTTPExporter(endpoint=base_url, headers=headers)
    grpc_endpoint, insecure = _resolve_grpc_exporter_endpoint(trimmed_endpoint)
    return GRPCExporter(endpoint=grpc_endpoint, headers=headers, insecure=insecure)


def init_metrics_provider(
    exporter: MetricExporter, resource_attributes: Optional[Dict[Any, Any]] = None
) -> MeterProvider:
    resource = (
        Resource.create(resource_attributes)
        if resource_attributes
        else Resource.create()
    )
    reader = PeriodicExportingMetricReader(exporter)
    provider = MeterProvider(
        metric_readers=[reader],
        resource=resource,
        views=metric_views(),
    )

    metrics.set_meter_provider(provider)
    return provider


def metric_views() -> Sequence[View]:
    return [
        View(
            instrument_name=Meters.LLM_TOKEN_USAGE,
            aggregation=ExplicitBucketHistogramAggregation(
                [
                    1,
                    4,
                    16,
                    64,
                    256,
                    1024,
                    4096,
                    16384,
                    65536,
                    262144,
                    1048576,
                    4194304,
                    16777216,
                    67108864,
                ]
            ),
        ),
        View(
            instrument_name=Meters.PINECONE_DB_QUERY_DURATION,
            aggregation=ExplicitBucketHistogramAggregation(
                [
                    0.01,
                    0.02,
                    0.04,
                    0.08,
                    0.16,
                    0.32,
                    0.64,
                    1.28,
                    2.56,
                    5.12,
                    10.24,
                    20.48,
                    40.96,
                    81.92,
                ]
            ),
        ),
        View(
            instrument_name=Meters.PINECONE_DB_QUERY_SCORES,
            aggregation=ExplicitBucketHistogramAggregation(
                [
                    -1,
                    -0.875,
                    -0.75,
                    -0.625,
                    -0.5,
                    -0.375,
                    -0.25,
                    -0.125,
                    0,
                    0.125,
                    0.25,
                    0.375,
                    0.5,
                    0.625,
                    0.75,
                    0.875,
                    1,
                ]
            ),
        ),
    ]

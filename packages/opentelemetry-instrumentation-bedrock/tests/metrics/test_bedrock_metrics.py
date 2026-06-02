# NOTE:
# This file has been modified by FortifyRoot.
# Original source: https://github.com/traceloop/openllmetry

import json

import pytest
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import Meters


@pytest.mark.vcr
def test_invoke_model_metrics(test_context, brt):
    if brt is None:
        print("test_invoke_model_metrics test skipped.")
        return

    _, _, reader = test_context

    body = json.dumps(
        {
            "inputText": "Tell me a joke about opentelemetry",
            "textGenerationConfig": {
                "maxTokenCount": 200,
                "temperature": 0.5,
                "topP": 0.5,
            },
        }
    )

    brt.invoke_model(
        body=body,
        modelId="amazon.titan-text-express-v1",
        accept="application/json",
        contentType="application/json",
    )

    metrics_data = reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics
    assert len(resource_metrics) > 0

    found_token_metric = False
    found_duration_metric = False
    expected_request_models = {
        "amazon.titan-text-express-v1",
        "titan-text-express-v1",
    }

    for rm in resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:

                if metric.name == Meters.LLM_TOKEN_USAGE:
                    found_token_metric = True
                    for data_point in metric.data.data_points:
                        assert data_point.attributes[GenAIAttributes.GEN_AI_TOKEN_TYPE] in [
                            "output",
                            "input",
                        ]
                        assert (
                            data_point.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL]
                            in expected_request_models
                        )
                        assert data_point.sum > 0

                if metric.name == Meters.LLM_OPERATION_DURATION:
                    found_duration_metric = True
                    assert any(
                        data_point.count > 0 for data_point in metric.data.data_points
                    )
                    assert any(
                        data_point.sum > 0 for data_point in metric.data.data_points
                    )
                    model_points = [
                        data_point
                        for data_point in metric.data.data_points
                        if GenAIAttributes.GEN_AI_REQUEST_MODEL
                        in data_point.attributes
                    ]
                    assert model_points
                    assert all(
                        data_point.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL]
                        in expected_request_models
                        for data_point in model_points
                    )

                if metric.name in (Meters.LLM_TOKEN_USAGE, Meters.LLM_OPERATION_DURATION):
                    assert (
                        metric.data.data_points[0].attributes[GenAIAttributes.GEN_AI_SYSTEM]
                        == "bedrock"
                    )

    assert found_token_metric is True
    assert found_duration_metric is True

"""
FR VCR cassette tests for safety rule enforcement on Bedrock instrumentation.

These tests record real AWS Bedrock API interactions via VCR cassettes and verify
that the safety callback system correctly detects and masks sensitive content
in both prompts and completions using the Converse API.

Safety rule flavours covered:
  - RegEx:  PCI.credit_card, PII.email
  - List:   PII.competitor_org (list of company names)
  - UDF:    custom_compliance.project_codename (user-defined function
            returning 2 sub-rule findings)

North-star: This is a NEW file (FR-owned). Zero delta on TL files.

Note: AWS Bedrock credentials are loaded via load-test-secrets.sh with FR_TEST_
prefix. The brt_safety fixture creates a boto3 client using these prefixed vars.
"""

import json
import os
import re

import boto3
import pytest
from opentelemetry.instrumentation.fortifyroot import (
    SafetyFinding,
    SafetyLocation,
    SafetyResult,
    clear_safety_handlers,
    register_completion_safety_handler,
    register_prompt_safety_handler,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)


# ---------------------------------------------------------------------------
# Safety rule definitions (mirror what real SDK rules would do)
# ---------------------------------------------------------------------------

# RegEx patterns
_CREDIT_CARD_RE = re.compile(r"\b(?:\d[ -]*?){13,19}\b")
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")

# List of competitor org names (case-insensitive)
_COMPETITOR_ORGS = ["acme corp", "globex industries", "initech", "umbrella inc"]

# UDF: detects two fictional project codenames
_PROJECT_CODENAMES = ["project-phoenix", "operation-atlas"]

# Module-level action setting (switched by fixtures)
_SAFETY_ACTION = "MASK"


def _build_findings_and_mask(text, findings_list, action="MASK"):
    """Apply all collected findings to produce masked (or allowed) text."""
    if not findings_list:
        return None
    if action == "MASK":
        masked = text
        for f in sorted(findings_list, key=lambda x: x.start, reverse=True):
            masked = masked[: f.start] + f"[{f.rule_name}]" + masked[f.end :]
        return SafetyResult(
            text=masked,
            overall_action="MASK",
            findings=findings_list,
        )
    return SafetyResult(
        text=text,
        overall_action="ALLOW",
        findings=findings_list,
    )


def _scan_text(text, location):
    """Run all safety rule types against text, return findings."""
    if not isinstance(text, str) or not text:
        return []

    action = _SAFETY_ACTION
    findings = []

    for m in _CREDIT_CARD_RE.finditer(text):
        findings.append(
            SafetyFinding(
                category="PCI",
                severity="HIGH",
                action=action,
                rule_name="PCI.credit_card",
                start=m.start(),
                end=m.end(),
            )
        )

    for m in _EMAIL_RE.finditer(text):
        findings.append(
            SafetyFinding(
                category="PII",
                severity="MEDIUM",
                action=action,
                rule_name="PII.email",
                start=m.start(),
                end=m.end(),
            )
        )

    text_lower = text.lower()
    for org in _COMPETITOR_ORGS:
        idx = text_lower.find(org)
        while idx != -1:
            findings.append(
                SafetyFinding(
                    category="PII",
                    severity="LOW",
                    action=action,
                    rule_name="PII.competitor_org",
                    start=idx,
                    end=idx + len(org),
                )
            )
            idx = text_lower.find(org, idx + len(org))

    for codename in _PROJECT_CODENAMES:
        idx = text_lower.find(codename)
        while idx != -1:
            findings.append(
                SafetyFinding(
                    category="COMPLIANCE",
                    severity="HIGH",
                    action=action,
                    rule_name=f"custom_compliance.{codename}",
                    start=idx,
                    end=idx + len(codename),
                )
            )
            idx = text_lower.find(codename, idx + len(codename))

    return findings


def _prompt_handler(context):
    if context.location != SafetyLocation.PROMPT:
        return None
    findings = _scan_text(context.text, SafetyLocation.PROMPT)
    return _build_findings_and_mask(context.text, findings, action=_SAFETY_ACTION)


def _completion_handler(context):
    if context.location != SafetyLocation.COMPLETION:
        return None
    findings = _scan_text(context.text, SafetyLocation.COMPLETION)
    return _build_findings_and_mask(context.text, findings, action=_SAFETY_ACTION)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _register_safety_handlers():
    """Register all safety handlers before each test, clear after."""
    clear_safety_handlers()
    register_prompt_safety_handler(_prompt_handler)
    register_completion_safety_handler(_completion_handler)
    yield
    clear_safety_handlers()


@pytest.fixture
def brt_safety():
    """Bedrock runtime client using FR_TEST_ prefixed credentials.

    load-test-secrets.sh exports AWS creds with FR_TEST_ prefix to avoid
    clobbering the caller's AWS identity. In VCR replay mode, the actual
    credentials don't matter (dummy values are used by conftest.py).
    """
    return boto3.client(
        service_name="bedrock-runtime",
        aws_access_key_id=os.environ.get("FR_TEST_AWS_ACCESS_KEY_ID", os.environ.get("AWS_ACCESS_KEY_ID", "test")),
        aws_secret_access_key=os.environ.get("FR_TEST_AWS_SECRET_ACCESS_KEY", os.environ.get("AWS_SECRET_ACCESS_KEY", "test")),
        region_name=os.environ.get("FR_TEST_AWS_DEFAULT_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-1")),
    )


# ---------------------------------------------------------------------------
# T3-B3: RegEx rule — PII.email in prompt (Converse)
# ---------------------------------------------------------------------------


@pytest.mark.vcr
@pytest.mark.fr
def test_bedrock_safety_regex_email_prompt(
    instrument_legacy, span_exporter, brt_safety
):
    """Prompt contains an email address. Safety should detect and mask it."""
    brt_safety.converse(
        modelId="anthropic.claude-3-haiku-20240307-v1:0",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "text": (
                            "Summarize the following customer inquiry from "
                            "john.doe@example.com about their recent order."
                        )
                    }
                ],
            }
        ],
        inferenceConfig={"maxTokens": 50},
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) >= 1
    span = spans[0]

    prompt_content = span.attributes.get(
        f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"
    )
    assert prompt_content is not None
    assert "john.doe@example.com" not in prompt_content
    assert "[PII.email]" in prompt_content

    safety_events = [
        e for e in span.events if "fortifyroot.safety" in (e.name or "")
    ]
    assert len(safety_events) >= 1
    event_attrs = safety_events[0].attributes
    assert event_attrs["fortifyroot.safety.rule_name"] == "PII.email"
    assert event_attrs["fortifyroot.safety.action"] == "MASK"
    assert event_attrs["fortifyroot.safety.location"] == "PROMPT"


# ---------------------------------------------------------------------------
# T3-B3: RegEx rule — PCI.credit_card in prompt (Converse)
# ---------------------------------------------------------------------------


@pytest.mark.vcr
@pytest.mark.fr
def test_bedrock_safety_regex_credit_card_prompt(
    instrument_legacy, span_exporter, brt_safety
):
    """Prompt contains a credit card number. Safety should detect and mask it."""
    brt_safety.converse(
        modelId="anthropic.claude-3-haiku-20240307-v1:0",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "text": "Check if order for card 4111 1111 1111 1111 was processed."
                    }
                ],
            }
        ],
        inferenceConfig={"maxTokens": 50},
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) >= 1
    span = spans[0]

    prompt_content = span.attributes.get(
        f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"
    )
    assert prompt_content is not None
    assert "4111 1111 1111 1111" not in prompt_content
    assert "[PCI.credit_card]" in prompt_content

    safety_events = [
        e for e in span.events if "fortifyroot.safety" in (e.name or "")
    ]
    found_cc = any(
        e.attributes.get("fortifyroot.safety.rule_name") == "PCI.credit_card"
        for e in safety_events
    )
    assert found_cc, "Expected PCI.credit_card safety finding"


# ---------------------------------------------------------------------------
# T3-B3: List rule — PII.competitor_org in prompt (Converse)
# ---------------------------------------------------------------------------


@pytest.mark.vcr
@pytest.mark.fr
def test_bedrock_safety_list_competitor_org_prompt(
    instrument_legacy, span_exporter, brt_safety
):
    """Prompt mentions a competitor org from the blocklist."""
    brt_safety.converse(
        modelId="anthropic.claude-3-haiku-20240307-v1:0",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "text": (
                            "Compare our product roadmap with Acme Corp's latest "
                            "offering and Globex Industries' pricing strategy."
                        )
                    }
                ],
            }
        ],
        inferenceConfig={"maxTokens": 50},
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) >= 1
    span = spans[0]

    prompt_content = span.attributes.get(
        f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"
    )
    assert prompt_content is not None
    assert "acme corp" not in prompt_content.lower()
    assert "globex industries" not in prompt_content.lower()
    assert "[PII.competitor_org]" in prompt_content

    safety_events = [
        e for e in span.events if "fortifyroot.safety" in (e.name or "")
    ]
    found_org = any(
        e.attributes.get("fortifyroot.safety.rule_name") == "PII.competitor_org"
        for e in safety_events
    )
    assert found_org, "Expected PII.competitor_org safety finding"


# ---------------------------------------------------------------------------
# T3-B3: UDF rule — custom_compliance with 2 sub-rules (Converse)
# ---------------------------------------------------------------------------


@pytest.mark.vcr
@pytest.mark.fr
def test_bedrock_safety_udf_project_codenames_prompt(
    instrument_legacy, span_exporter, brt_safety
):
    """Prompt contains two project codenames detected by UDF rule."""
    brt_safety.converse(
        modelId="anthropic.claude-3-haiku-20240307-v1:0",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "text": (
                            "Prepare a status update for Project-Phoenix and "
                            "Operation-Atlas including timeline and budget."
                        )
                    }
                ],
            }
        ],
        inferenceConfig={"maxTokens": 50},
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) >= 1
    span = spans[0]

    prompt_content = span.attributes.get(
        f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"
    )
    assert prompt_content is not None
    assert "[custom_compliance.project-phoenix]" in prompt_content
    assert "[custom_compliance.operation-atlas]" in prompt_content
    assert "Project-Phoenix" not in prompt_content
    assert "Operation-Atlas" not in prompt_content

    safety_events = [
        e for e in span.events if "fortifyroot.safety" in (e.name or "")
    ]
    udf_findings = [
        e
        for e in safety_events
        if (e.attributes.get("fortifyroot.safety.rule_name") or "").startswith(
            "custom_compliance."
        )
    ]
    assert len(udf_findings) >= 2, (
        f"Expected 2 UDF findings, got {len(udf_findings)}"
    )


# ---------------------------------------------------------------------------
# T3-B3: Completion safety — email in LLM response (Converse)
# ---------------------------------------------------------------------------


@pytest.mark.vcr
@pytest.mark.fr
def test_bedrock_safety_completion_email_masked(
    instrument_legacy, span_exporter, brt_safety
):
    """
    Ask the LLM to generate fictional contact info. The response should
    contain an email-like pattern that completion safety detects and masks.
    """
    brt_safety.converse(
        modelId="anthropic.claude-3-haiku-20240307-v1:0",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "text": (
                            "Create a fictional contact card. The person's name is "
                            "Jane Smith, company is Widgets Inc, role is VP of Sales. "
                            "Invent a plausible work email and phone. Format as:\n"
                            "Name: ...\nEmail: ...\nPhone: ..."
                        )
                    }
                ],
            }
        ],
        inferenceConfig={"maxTokens": 100},
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) >= 1
    span = spans[0]

    completion_content = span.attributes.get(
        f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content"
    )
    assert completion_content is not None, "Completion content should be recorded"
    assert "[PII.email]" in completion_content, (
        "LLM-generated email in response should be masked by completion safety"
    )

    safety_events = [
        e for e in span.events if "fortifyroot.safety" in (e.name or "")
    ]
    completion_findings = [
        e
        for e in safety_events
        if e.attributes.get("fortifyroot.safety.location") == "COMPLETION"
    ]
    assert len(completion_findings) >= 1, "Should have completion safety finding"
    assert any(
        e.attributes.get("fortifyroot.safety.rule_name") == "PII.email"
        for e in completion_findings
    ), "Completion safety should detect PII.email"


# ---------------------------------------------------------------------------
# T3-B3: Combined — multiple rule types in single prompt (Converse)
# ---------------------------------------------------------------------------


@pytest.mark.vcr
@pytest.mark.fr
def test_bedrock_safety_combined_rules_prompt(
    instrument_legacy, span_exporter, brt_safety
):
    """Prompt triggers all rule types: RegEx (email), List (org), UDF (codename)."""
    brt_safety.converse(
        modelId="anthropic.claude-3-haiku-20240307-v1:0",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "text": (
                            "Send the Project-Phoenix briefing to contact@acme-corp.com "
                            "and CC the Initech team."
                        )
                    }
                ],
            }
        ],
        inferenceConfig={"maxTokens": 50},
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) >= 1
    span = spans[0]

    prompt_content = span.attributes.get(
        f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"
    )
    assert prompt_content is not None

    safety_events = [
        e for e in span.events if "fortifyroot.safety" in (e.name or "")
    ]
    rule_names = {
        e.attributes.get("fortifyroot.safety.rule_name") for e in safety_events
    }
    assert "PII.email" in rule_names, "Expected PII.email finding"
    assert "PII.competitor_org" in rule_names or any(
        "competitor_org" in (r or "") for r in rule_names
    ), "Expected PII.competitor_org finding"
    assert any(
        (r or "").startswith("custom_compliance.") for r in rule_names
    ), "Expected custom_compliance UDF finding"


# ---------------------------------------------------------------------------
# T3-B3: Streaming with PII — verify safety on streamed completion (Converse)
# ---------------------------------------------------------------------------


@pytest.mark.vcr
@pytest.mark.fr
def test_bedrock_safety_streaming_with_pii(
    instrument_legacy, span_exporter, brt_safety
):
    """Streaming converse with PII in prompt. Verify prompt safety applied."""
    response = brt_safety.converse_stream(
        modelId="anthropic.claude-3-haiku-20240307-v1:0",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "text": (
                            "Acknowledge receipt of message from "
                            "alice.jones@example.org about Project-Phoenix."
                        )
                    }
                ],
            }
        ],
        inferenceConfig={"maxTokens": 50},
    )

    # Consume the stream
    for _ in response.get("stream", []):
        pass

    spans = span_exporter.get_finished_spans()
    assert len(spans) >= 1
    span = spans[0]

    prompt_content = span.attributes.get(
        f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"
    )
    assert prompt_content is not None
    assert "alice.jones@example.org" not in prompt_content
    assert "[PII.email]" in prompt_content


# ===========================================================================
# ALLOW action tests
# ===========================================================================


@pytest.fixture
def _set_allow_action():
    """Switch safety handlers to ALLOW mode for the test."""
    global _SAFETY_ACTION
    _SAFETY_ACTION = "ALLOW"
    yield
    _SAFETY_ACTION = "MASK"


@pytest.mark.vcr
@pytest.mark.fr
def test_bedrock_safety_allow_email_not_masked(
    _set_allow_action, instrument_legacy, span_exporter, brt_safety
):
    """ALLOW action: email detected but text passes through unchanged."""
    brt_safety.converse(
        modelId="anthropic.claude-3-haiku-20240307-v1:0",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "text": (
                            "Forward this message from alice.jones@example.org "
                            "to the support team."
                        )
                    }
                ],
            }
        ],
        inferenceConfig={"maxTokens": 50},
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) >= 1
    span = spans[0]

    prompt_content = span.attributes.get(
        f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"
    )
    assert prompt_content is not None
    assert "alice.jones@example.org" in prompt_content
    assert "[PII.email]" not in prompt_content

    safety_events = [
        e for e in span.events if "fortifyroot.safety" in (e.name or "")
    ]
    assert len(safety_events) >= 1, "ALLOW should still emit safety findings"
    event_attrs = safety_events[0].attributes
    assert event_attrs["fortifyroot.safety.rule_name"] == "PII.email"
    assert event_attrs["fortifyroot.safety.action"] == "ALLOW"
    assert event_attrs["fortifyroot.safety.location"] == "PROMPT"


@pytest.mark.vcr
@pytest.mark.fr
def test_bedrock_safety_allow_credit_card_not_masked(
    _set_allow_action, instrument_legacy, span_exporter, brt_safety
):
    """ALLOW action: credit card detected but text passes through unchanged."""
    brt_safety.converse(
        modelId="anthropic.claude-3-haiku-20240307-v1:0",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "text": "Verify payment for card 4111 1111 1111 1111 was received."
                    }
                ],
            }
        ],
        inferenceConfig={"maxTokens": 50},
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) >= 1
    span = spans[0]

    prompt_content = span.attributes.get(
        f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"
    )
    assert prompt_content is not None
    assert "4111 1111 1111 1111" in prompt_content

    safety_events = [
        e for e in span.events if "fortifyroot.safety" in (e.name or "")
    ]
    found_cc = any(
        e.attributes.get("fortifyroot.safety.rule_name") == "PCI.credit_card"
        and e.attributes.get("fortifyroot.safety.action") == "ALLOW"
        for e in safety_events
    )
    assert found_cc, "Expected PCI.credit_card finding with ALLOW action"

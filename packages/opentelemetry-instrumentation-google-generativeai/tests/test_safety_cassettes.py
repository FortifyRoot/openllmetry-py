"""
FR VCR cassette tests for safety rule enforcement on Google GenAI instrumentation.

These tests record real Google GenAI API interactions via VCR cassettes and verify
that the safety callback system correctly detects and masks sensitive content
in both prompts and completions.

Safety rule flavours covered:
  - RegEx:  PCI.credit_card, PII.email
  - List:   PII.competitor_org (list of company names)
  - UDF:    custom_compliance.project_codename (user-defined function
            returning 2 sub-rule findings)

North-star: This is a NEW file (FR-owned). Zero delta on TL files.
"""

import re

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
        # Sort findings by start position descending so replacements don't shift indices
        for f in sorted(findings_list, key=lambda x: x.start, reverse=True):
            masked = masked[: f.start] + f"[{f.rule_name}]" + masked[f.end :]
        return SafetyResult(
            text=masked,
            overall_action="MASK",
            findings=findings_list,
        )
    # ALLOW: text unchanged, findings still recorded
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

    # --- RegEx: PCI.credit_card ---
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

    # --- RegEx: PII.email ---
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

    # --- List: PII.competitor_org ---
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

    # --- UDF: custom_compliance.project_codename (returns 2 sub-rules) ---
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
    """Prompt safety handler covering RegEx, List, and UDF rules."""
    if context.location != SafetyLocation.PROMPT:
        return None
    findings = _scan_text(context.text, SafetyLocation.PROMPT)
    return _build_findings_and_mask(context.text, findings, action=_SAFETY_ACTION)


def _completion_handler(context):
    """Completion safety handler covering RegEx, List, and UDF rules."""
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


# ---------------------------------------------------------------------------
# T3-A3: RegEx rule — PII.email in prompt
# ---------------------------------------------------------------------------


@pytest.mark.vcr
@pytest.mark.fr
def test_google_genai_safety_regex_email_prompt(exporter, genai_client):
    """Prompt contains an email address. Safety should detect and mask it."""
    exporter.clear()
    genai_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=(
            "Summarize the following customer inquiry from "
            "john.doe@example.com about their recent order."
        ),
        config={"max_output_tokens": 50},
    )

    spans = exporter.get_finished_spans()
    assert len(spans) >= 1
    span = spans[0]

    # The prompt content should have been masked (email replaced)
    prompt_content = span.attributes.get("gen_ai.prompt.0.content")
    assert prompt_content is not None
    assert "john.doe@example.com" not in prompt_content
    assert "[PII.email]" in prompt_content

    # Safety finding should be recorded as a span event
    safety_events = [
        e for e in span.events if "fortifyroot.safety" in (e.name or "")
    ]
    assert len(safety_events) >= 1
    event_attrs = safety_events[0].attributes
    assert event_attrs["fortifyroot.safety.rule_name"] == "PII.email"
    assert event_attrs["fortifyroot.safety.action"] == "MASK"
    assert event_attrs["fortifyroot.safety.location"] == "PROMPT"


# ---------------------------------------------------------------------------
# T3-A3: RegEx rule — PCI.credit_card in prompt
# ---------------------------------------------------------------------------


@pytest.mark.vcr
@pytest.mark.fr
def test_google_genai_safety_regex_credit_card_prompt(exporter, genai_client):
    """Prompt contains a credit card number. Safety should detect and mask it."""
    exporter.clear()
    genai_client.models.generate_content(
        model="gemini-2.5-flash",
        contents="Check if order for card 4111 1111 1111 1111 was processed.",
        config={"max_output_tokens": 50},
    )

    spans = exporter.get_finished_spans()
    assert len(spans) >= 1
    span = spans[0]

    prompt_content = span.attributes.get("gen_ai.prompt.0.content")
    assert prompt_content is not None
    assert "4111 1111 1111 1111" not in prompt_content
    assert "[PCI.credit_card]" in prompt_content

    safety_events = [
        e for e in span.events if "fortifyroot.safety" in (e.name or "")
    ]
    assert len(safety_events) >= 1
    found_cc = any(
        e.attributes.get("fortifyroot.safety.rule_name") == "PCI.credit_card"
        for e in safety_events
    )
    assert found_cc, "Expected PCI.credit_card safety finding"


# ---------------------------------------------------------------------------
# T3-A3: List rule — PII.competitor_org in prompt
# ---------------------------------------------------------------------------


@pytest.mark.vcr
@pytest.mark.fr
def test_google_genai_safety_list_competitor_org_prompt(exporter, genai_client):
    """Prompt mentions a competitor org from the blocklist."""
    exporter.clear()
    genai_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=(
            "Compare our product roadmap with Acme Corp's latest "
            "offering and Globex Industries' pricing strategy."
        ),
        config={"max_output_tokens": 50},
    )

    spans = exporter.get_finished_spans()
    assert len(spans) >= 1
    span = spans[0]

    prompt_content = span.attributes.get("gen_ai.prompt.0.content")
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
# T3-A3: UDF rule — custom_compliance with 2 sub-rules in prompt
# ---------------------------------------------------------------------------


@pytest.mark.vcr
@pytest.mark.fr
def test_google_genai_safety_udf_project_codenames_prompt(exporter, genai_client):
    """Prompt contains two project codenames detected by UDF rule."""
    exporter.clear()
    genai_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=(
            "Prepare a status update for Project-Phoenix and "
            "Operation-Atlas including timeline and budget."
        ),
        config={"max_output_tokens": 50},
    )

    spans = exporter.get_finished_spans()
    assert len(spans) >= 1
    span = spans[0]

    prompt_content = span.attributes.get("gen_ai.prompt.0.content")
    assert prompt_content is not None
    assert "[custom_compliance.project-phoenix]" in prompt_content
    assert "[custom_compliance.operation-atlas]" in prompt_content
    assert "Project-Phoenix" not in prompt_content
    assert "Operation-Atlas" not in prompt_content

    # UDF should produce 2 findings (one per codename)
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
# T3-A3: Completion safety — email in LLM response
# ---------------------------------------------------------------------------


@pytest.mark.vcr
@pytest.mark.fr
def test_google_genai_safety_completion_email_masked(exporter, genai_client):
    """
    Ask the LLM to generate fictional contact info. The response should
    contain an email-like pattern that completion safety detects and masks.
    """
    exporter.clear()
    genai_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=(
            "Reply with ONLY this text, filling in the blanks:\n"
            "Email: jane.smith@widgetsinc.com\n"
            "Phone: 555-0142\n"
            "Do not add anything else."
        ),
        config={
            "max_output_tokens": 200,
            "thinking_config": {"thinking_budget": 0},
        },
    )

    spans = exporter.get_finished_spans()
    assert len(spans) >= 1
    span = spans[0]

    # Completion should be recorded AND the LLM-generated email should be
    # masked by completion safety
    completion_content = span.attributes.get("gen_ai.completion.0.content")
    assert completion_content is not None, "Completion content should be recorded"
    assert "[PII.email]" in completion_content, (
        "LLM-generated email in response should be masked by completion safety"
    )

    # Verify safety events are present. The primary assertion is the
    # masking above; span events confirm the instrumentation layer
    # recorded the finding metadata.
    safety_events = [
        e for e in span.events if "fortifyroot.safety" in (e.name or "")
    ]
    # At minimum, the prompt safety handler fires (because the prompt
    # includes the email template text). Completion masking is verified
    # by the [PII.email] assertion on completion_content above.
    assert len(safety_events) >= 1, "Should have at least one safety finding"


# ---------------------------------------------------------------------------
# T3-A3: Combined — multiple rule types in single prompt
# ---------------------------------------------------------------------------


@pytest.mark.vcr
@pytest.mark.fr
def test_google_genai_safety_combined_rules_prompt(exporter, genai_client):
    """Prompt triggers all rule types: RegEx (email), List (org), UDF (codename)."""
    exporter.clear()
    genai_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=(
            "Send the Project-Phoenix briefing to contact@acme-corp.com "
            "and CC the Initech team."
        ),
        config={"max_output_tokens": 50},
    )

    spans = exporter.get_finished_spans()
    assert len(spans) >= 1
    span = spans[0]

    prompt_content = span.attributes.get("gen_ai.prompt.0.content")
    assert prompt_content is not None

    # Verify all 3 rule types fired
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
# T3-A3: Streaming with PII — verify safety on streamed completion
# ---------------------------------------------------------------------------


@pytest.mark.vcr
@pytest.mark.fr
def test_google_genai_safety_streaming_with_pii(exporter, genai_client):
    """Streaming chat with PII in prompt. Verify prompt safety applied."""
    exporter.clear()
    response = genai_client.models.generate_content_stream(
        model="gemini-2.5-flash",
        contents=(
            "Acknowledge receipt of message from "
            "alice.jones@example.org about Project-Phoenix."
        ),
        config={"max_output_tokens": 50},
    )

    # Consume the stream
    for _ in response:
        pass

    spans = exporter.get_finished_spans()
    assert len(spans) >= 1
    span = spans[0]

    # Prompt should be masked
    prompt_content = span.attributes.get("gen_ai.prompt.0.content")
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
def test_google_genai_safety_allow_email_not_masked(
    _set_allow_action, exporter, genai_client
):
    """ALLOW action: email detected but text passes through unchanged."""
    exporter.clear()
    genai_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=(
            "Forward this message from alice.jones@example.org "
            "to the support team."
        ),
        config={"max_output_tokens": 50},
    )

    spans = exporter.get_finished_spans()
    assert len(spans) >= 1
    span = spans[0]

    # With ALLOW, the original text should pass through unchanged
    prompt_content = span.attributes.get("gen_ai.prompt.0.content")
    assert prompt_content is not None
    assert "alice.jones@example.org" in prompt_content
    assert "[PII.email]" not in prompt_content

    # But findings should STILL be recorded as span events
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
def test_google_genai_safety_allow_credit_card_not_masked(
    _set_allow_action, exporter, genai_client
):
    """ALLOW action: credit card detected but text passes through unchanged."""
    exporter.clear()
    genai_client.models.generate_content(
        model="gemini-2.5-flash",
        contents="Verify payment for card 4111 1111 1111 1111 was received.",
        config={"max_output_tokens": 50},
    )

    spans = exporter.get_finished_spans()
    assert len(spans) >= 1
    span = spans[0]

    # With ALLOW, text unchanged — card number still visible
    prompt_content = span.attributes.get("gen_ai.prompt.0.content")
    assert prompt_content is not None
    assert "4111 1111 1111 1111" in prompt_content

    # Findings still recorded
    safety_events = [
        e for e in span.events if "fortifyroot.safety" in (e.name or "")
    ]
    found_cc = any(
        e.attributes.get("fortifyroot.safety.rule_name") == "PCI.credit_card"
        and e.attributes.get("fortifyroot.safety.action") == "ALLOW"
        for e in safety_events
    )
    assert found_cc, "Expected PCI.credit_card finding with ALLOW action"

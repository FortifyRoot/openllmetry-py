"""
FR VCR cassette tests for safety rule enforcement on LiteLLM instrumentation.

These tests record real LiteLLM API interactions via VCR cassettes and verify
that the safety callback system correctly detects and masks sensitive content
in both prompts and completions.

Safety rule flavours covered:
  - RegEx:  PCI.credit_card, PII.email
  - List:   PII.competitor_org (list of company names)
  - UDF:    custom_compliance.project_codename (user-defined function
            returning 2 sub-rule findings)

Since LiteLLM is FR-authored, this also tests the image_url passthrough
(non-text content safety no-op).

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
from opentelemetry.semconv_ai import SpanAttributes


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
            masked = masked[:f.start] + f"[{f.rule_name}]" + masked[f.end:]
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
        findings.append(SafetyFinding(
            category="PCI",
            severity="HIGH",
            action=action,
            rule_name="PCI.credit_card",
            start=m.start(),
            end=m.end(),
        ))

    # --- RegEx: PII.email ---
    for m in _EMAIL_RE.finditer(text):
        findings.append(SafetyFinding(
            category="PII",
            severity="MEDIUM",
            action=action,
            rule_name="PII.email",
            start=m.start(),
            end=m.end(),
        ))

    # --- List: PII.competitor_org ---
    text_lower = text.lower()
    for org in _COMPETITOR_ORGS:
        idx = text_lower.find(org)
        while idx != -1:
            findings.append(SafetyFinding(
                category="PII",
                severity="LOW",
                action=action,
                rule_name="PII.competitor_org",
                start=idx,
                end=idx + len(org),
            ))
            idx = text_lower.find(org, idx + len(org))

    # --- UDF: custom_compliance.project_codename (returns 2 sub-rules) ---
    for codename in _PROJECT_CODENAMES:
        idx = text_lower.find(codename)
        while idx != -1:
            findings.append(SafetyFinding(
                category="COMPLIANCE",
                severity="HIGH",
                action=action,
                rule_name=f"custom_compliance.{codename}",
                start=idx,
                end=idx + len(codename),
            ))
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
# RegEx rule — PII.email in prompt
# ---------------------------------------------------------------------------

@pytest.mark.vcr
@pytest.mark.fr
def test_litellm_safety_regex_email_prompt(instrument, span_exporter):
    """Prompt contains an email address. Safety should detect and mask it."""
    import litellm

    litellm.completion(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": (
                    "Summarize the following customer inquiry from "
                    "john.doe@example.com about their recent order."
                ),
            }
        ],
        max_tokens=50,
    )

    spans = span_exporter.get_finished_spans()
    fr_span = next(
        (s for s in spans if s.name == "fortifyroot.litellm.safety"), None
    )
    assert fr_span is not None

    prompt_content = fr_span.attributes.get(
        f"{SpanAttributes.LLM_PROMPTS}.0.content"
    )
    assert prompt_content is not None
    assert "john.doe@example.com" not in prompt_content
    assert "[PII.email]" in prompt_content

    safety_events = [
        e for e in fr_span.events
        if "fortifyroot.safety" in (e.name or "")
    ]
    assert len(safety_events) >= 1
    event_attrs = safety_events[0].attributes
    assert event_attrs["fortifyroot.safety.rule_name"] == "PII.email"
    assert event_attrs["fortifyroot.safety.action"] == "MASK"
    assert event_attrs["fortifyroot.safety.location"] == "PROMPT"


# ---------------------------------------------------------------------------
# RegEx rule — PCI.credit_card in prompt
# ---------------------------------------------------------------------------

@pytest.mark.vcr
@pytest.mark.fr
def test_litellm_safety_regex_credit_card_prompt(instrument, span_exporter):
    """Prompt contains a credit card number. Safety should detect and mask it."""
    import litellm

    litellm.completion(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": (
                    "Check if order for card 4111 1111 1111 1111 was processed."
                ),
            }
        ],
        max_tokens=50,
    )

    spans = span_exporter.get_finished_spans()
    fr_span = next(
        (s for s in spans if s.name == "fortifyroot.litellm.safety"), None
    )
    assert fr_span is not None

    prompt_content = fr_span.attributes.get(
        f"{SpanAttributes.LLM_PROMPTS}.0.content"
    )
    assert prompt_content is not None
    assert "4111 1111 1111 1111" not in prompt_content
    assert "[PCI.credit_card]" in prompt_content

    safety_events = [
        e for e in fr_span.events
        if "fortifyroot.safety" in (e.name or "")
    ]
    found_cc = any(
        e.attributes.get("fortifyroot.safety.rule_name") == "PCI.credit_card"
        for e in safety_events
    )
    assert found_cc, "Expected PCI.credit_card safety finding"


# ---------------------------------------------------------------------------
# List rule — PII.competitor_org in prompt
# ---------------------------------------------------------------------------

@pytest.mark.vcr
@pytest.mark.fr
def test_litellm_safety_list_competitor_org_prompt(instrument, span_exporter):
    """Prompt mentions a competitor org from the blocklist."""
    import litellm

    litellm.completion(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": (
                    "Compare our product roadmap with Acme Corp's latest "
                    "offering and Globex Industries' pricing strategy."
                ),
            }
        ],
        max_tokens=50,
    )

    spans = span_exporter.get_finished_spans()
    fr_span = next(
        (s for s in spans if s.name == "fortifyroot.litellm.safety"), None
    )
    assert fr_span is not None

    prompt_content = fr_span.attributes.get(
        f"{SpanAttributes.LLM_PROMPTS}.0.content"
    )
    assert prompt_content is not None
    assert "acme corp" not in prompt_content.lower()
    assert "globex industries" not in prompt_content.lower()
    assert "[PII.competitor_org]" in prompt_content

    safety_events = [
        e for e in fr_span.events
        if "fortifyroot.safety" in (e.name or "")
    ]
    found_org = any(
        e.attributes.get("fortifyroot.safety.rule_name") == "PII.competitor_org"
        for e in safety_events
    )
    assert found_org, "Expected PII.competitor_org safety finding"


# ---------------------------------------------------------------------------
# UDF rule — custom_compliance with 2 sub-rules in prompt
# ---------------------------------------------------------------------------

@pytest.mark.vcr
@pytest.mark.fr
def test_litellm_safety_udf_project_codenames_prompt(instrument, span_exporter):
    """Prompt contains two project codenames detected by UDF rule."""
    import litellm

    litellm.completion(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": (
                    "Prepare a status update for Project-Phoenix and "
                    "Operation-Atlas including timeline and budget."
                ),
            }
        ],
        max_tokens=50,
    )

    spans = span_exporter.get_finished_spans()
    fr_span = next(
        (s for s in spans if s.name == "fortifyroot.litellm.safety"), None
    )
    assert fr_span is not None

    prompt_content = fr_span.attributes.get(
        f"{SpanAttributes.LLM_PROMPTS}.0.content"
    )
    assert prompt_content is not None
    assert "[custom_compliance.project-phoenix]" in prompt_content
    assert "[custom_compliance.operation-atlas]" in prompt_content
    assert "Project-Phoenix" not in prompt_content
    assert "Operation-Atlas" not in prompt_content

    safety_events = [
        e for e in fr_span.events
        if "fortifyroot.safety" in (e.name or "")
    ]
    udf_findings = [
        e for e in safety_events
        if (e.attributes.get("fortifyroot.safety.rule_name") or "").startswith(
            "custom_compliance."
        )
    ]
    assert len(udf_findings) >= 2, (
        f"Expected 2 UDF findings, got {len(udf_findings)}"
    )


# ---------------------------------------------------------------------------
# Completion safety — email in LLM response
# ---------------------------------------------------------------------------

@pytest.mark.vcr
@pytest.mark.fr
def test_litellm_safety_completion_email_captured(instrument, span_exporter):
    """
    Ask the LLM to generate fictional contact info. Verify the completion
    content is captured in the span.

    NOTE: LiteLLM completion safety is applied via _FortifyRootCompletionLogger
    callback (litellm.callbacks[0]). During VCR replay, LiteLLM's async
    logging worker does not fire callbacks, so completion masking cannot be
    verified in cassette mode. Completion masking is thoroughly covered by
    mock-based tests in test_logger_integration.py.

    This test verifies: (1) completion content is captured, (2) prompt has
    no PII so no prompt masking occurs.
    """
    import litellm

    litellm.completion(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": (
                    "Create a fictional contact card. The person's name is "
                    "Jane Smith, company is Widgets Inc, role is VP of Sales. "
                    "Invent a plausible work email and phone. Format as:\n"
                    "Name: ...\nEmail: ...\nPhone: ..."
                ),
            }
        ],
        max_tokens=100,
    )

    spans = span_exporter.get_finished_spans()
    fr_span = next(
        (s for s in spans if s.name == "fortifyroot.litellm.safety"), None
    )
    assert fr_span is not None

    # Completion content should be captured (telemetry works)
    completion_content = fr_span.attributes.get(
        f"{SpanAttributes.LLM_COMPLETIONS}.0.content"
    )
    assert completion_content is not None, "Completion content should be recorded"
    # The LLM should have generated some contact info
    assert "Jane Smith" in completion_content or "Widgets" in completion_content, (
        "Completion should contain fictional contact info"
    )


# ---------------------------------------------------------------------------
# Combined — multiple rule types in single prompt
# ---------------------------------------------------------------------------

@pytest.mark.vcr
@pytest.mark.fr
def test_litellm_safety_combined_rules_prompt(instrument, span_exporter):
    """Prompt triggers all rule types: RegEx (email), List (org), UDF (codename)."""
    import litellm

    litellm.completion(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": (
                    "Send the Project-Phoenix briefing to contact@acme-corp.com "
                    "and CC the Initech team."
                ),
            }
        ],
        max_tokens=50,
    )

    spans = span_exporter.get_finished_spans()
    fr_span = next(
        (s for s in spans if s.name == "fortifyroot.litellm.safety"), None
    )
    assert fr_span is not None

    safety_events = [
        e for e in fr_span.events
        if "fortifyroot.safety" in (e.name or "")
    ]
    rule_names = {
        e.attributes.get("fortifyroot.safety.rule_name")
        for e in safety_events
    }
    assert "PII.email" in rule_names, "Expected PII.email finding"
    assert "PII.competitor_org" in rule_names or any(
        "competitor_org" in (r or "") for r in rule_names
    ), "Expected PII.competitor_org finding"
    assert any(
        (r or "").startswith("custom_compliance.") for r in rule_names
    ), "Expected custom_compliance UDF finding"


# ---------------------------------------------------------------------------
# Streaming with PII — verify safety on streamed completion
# ---------------------------------------------------------------------------

@pytest.mark.vcr
@pytest.mark.fr
def test_litellm_safety_streaming_with_pii(instrument, span_exporter):
    """Streaming chat with PII in prompt. Verify prompt safety applied."""
    import litellm

    response = litellm.completion(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": (
                    "Acknowledge receipt of message from "
                    "alice.jones@example.org about Project-Phoenix."
                ),
            }
        ],
        max_tokens=50,
        stream=True,
    )

    # Consume the stream
    for _ in response:
        pass

    spans = span_exporter.get_finished_spans()
    fr_span = next(
        (s for s in spans if s.name == "fortifyroot.litellm.safety"), None
    )
    assert fr_span is not None

    prompt_content = fr_span.attributes.get(
        f"{SpanAttributes.LLM_PROMPTS}.0.content"
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
def test_litellm_safety_allow_email_not_masked(
    _set_allow_action, instrument, span_exporter
):
    """ALLOW action: email detected but text passes through unchanged."""
    import litellm

    litellm.completion(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": (
                    "Forward this message from alice.jones@example.org "
                    "to the support team."
                ),
            }
        ],
        max_tokens=50,
    )

    spans = span_exporter.get_finished_spans()
    fr_span = next(
        (s for s in spans if s.name == "fortifyroot.litellm.safety"), None
    )
    assert fr_span is not None

    prompt_content = fr_span.attributes.get(
        f"{SpanAttributes.LLM_PROMPTS}.0.content"
    )
    assert prompt_content is not None
    assert "alice.jones@example.org" in prompt_content
    assert "[PII.email]" not in prompt_content

    safety_events = [
        e for e in fr_span.events
        if "fortifyroot.safety" in (e.name or "")
    ]
    assert len(safety_events) >= 1, "ALLOW should still emit safety findings"
    event_attrs = safety_events[0].attributes
    assert event_attrs["fortifyroot.safety.rule_name"] == "PII.email"
    assert event_attrs["fortifyroot.safety.action"] == "ALLOW"
    assert event_attrs["fortifyroot.safety.location"] == "PROMPT"


@pytest.mark.vcr
@pytest.mark.fr
def test_litellm_safety_allow_credit_card_not_masked(
    _set_allow_action, instrument, span_exporter
):
    """ALLOW action: credit card detected but text passes through unchanged."""
    import litellm

    litellm.completion(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": (
                    "Verify payment for card 4111 1111 1111 1111 was received."
                ),
            }
        ],
        max_tokens=50,
    )

    spans = span_exporter.get_finished_spans()
    fr_span = next(
        (s for s in spans if s.name == "fortifyroot.litellm.safety"), None
    )
    assert fr_span is not None

    prompt_content = fr_span.attributes.get(
        f"{SpanAttributes.LLM_PROMPTS}.0.content"
    )
    assert prompt_content is not None
    assert "4111 1111 1111 1111" in prompt_content

    safety_events = [
        e for e in fr_span.events
        if "fortifyroot.safety" in (e.name or "")
    ]
    found_cc = any(
        e.attributes.get("fortifyroot.safety.rule_name") == "PCI.credit_card"
        and e.attributes.get("fortifyroot.safety.action") == "ALLOW"
        for e in safety_events
    )
    assert found_cc, "Expected PCI.credit_card finding with ALLOW action"


# ---------------------------------------------------------------------------
# T4-C7: Non-text smoke — image_url pass-through
# ---------------------------------------------------------------------------

@pytest.mark.vcr
@pytest.mark.fr
def test_litellm_safety_image_url_passthrough(instrument, span_exporter):
    """
    Message with text + image_url blocks. Safety should mask PII in text
    but leave image_url block unchanged. Span should still be captured.
    """
    import litellm

    litellm.completion(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "PII email: john@example.com. Describe this image."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"
                        },
                    },
                ],
            }
        ],
        max_tokens=50,
    )

    spans = span_exporter.get_finished_spans()
    fr_span = next(
        (s for s in spans if s.name == "fortifyroot.litellm.safety"), None
    )
    assert fr_span is not None

    # Text part should be masked (email replaced)
    prompt_content = fr_span.attributes.get(
        f"{SpanAttributes.LLM_PROMPTS}.0.content"
    )
    assert prompt_content is not None
    assert "john@example.com" not in prompt_content
    assert "[PII.email]" in prompt_content

    # Span should still have correct attributes
    assert fr_span.attributes.get("fortifyroot.span.role") == "safety_wrapper"

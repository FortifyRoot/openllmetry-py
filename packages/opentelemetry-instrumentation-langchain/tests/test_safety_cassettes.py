"""
FR VCR cassette tests for safety rule enforcement on LangChain instrumentation.

These tests record real LangChain API interactions via VCR cassettes and verify
that the safety callback system correctly masks sensitive content in span
attributes for both prompts and completions.

Safety span events are emitted on the correct span via deferred_findings.py,
which queues findings during the safety handler and flushes them once the
span is available in the OTel context.

Safety rule flavours covered:
  - RegEx:  PCI.credit_card, PII.email
  - List:   PII.competitor_org (list of company names)
  - UDF:    custom_compliance.project_codename (user-defined function
            returning 2 sub-rule findings)

Includes LangGraph workflow safety test (T4-A4).

North-star: This is a NEW file (FR-owned). Zero delta on TL files.
"""

import re

import pytest
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
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
# Safety rule definitions (identical to T2 pattern)
# ---------------------------------------------------------------------------

_CREDIT_CARD_RE = re.compile(r"\b(?:\d[ -]*?){13,19}\b")
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
_COMPETITOR_ORGS = ["acme corp", "globex industries", "initech", "umbrella inc"]
_PROJECT_CODENAMES = ["project-phoenix", "operation-atlas"]
_SAFETY_ACTION = "MASK"


def _build_findings_and_mask(text, findings_list, action="MASK"):
    if not findings_list:
        return None
    if action == "MASK":
        masked = text
        for f in sorted(findings_list, key=lambda x: x.start, reverse=True):
            masked = masked[:f.start] + f"[{f.rule_name}]" + masked[f.end:]
        return SafetyResult(
            text=masked, overall_action="MASK", findings=findings_list,
        )
    return SafetyResult(
        text=text, overall_action="ALLOW", findings=findings_list,
    )


def _scan_text(text, location):
    if not isinstance(text, str) or not text:
        return []

    action = _SAFETY_ACTION
    findings = []

    for m in _CREDIT_CARD_RE.finditer(text):
        findings.append(SafetyFinding(
            category="PCI", severity="HIGH", action=action,
            rule_name="PCI.credit_card", start=m.start(), end=m.end(),
        ))

    for m in _EMAIL_RE.finditer(text):
        findings.append(SafetyFinding(
            category="PII", severity="MEDIUM", action=action,
            rule_name="PII.email", start=m.start(), end=m.end(),
        ))

    text_lower = text.lower()
    for org in _COMPETITOR_ORGS:
        idx = text_lower.find(org)
        while idx != -1:
            findings.append(SafetyFinding(
                category="PII", severity="LOW", action=action,
                rule_name="PII.competitor_org", start=idx, end=idx + len(org),
            ))
            idx = text_lower.find(org, idx + len(org))

    for codename in _PROJECT_CODENAMES:
        idx = text_lower.find(codename)
        while idx != -1:
            findings.append(SafetyFinding(
                category="COMPLIANCE", severity="HIGH", action=action,
                rule_name=f"custom_compliance.{codename}",
                start=idx, end=idx + len(codename),
            ))
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
# Helpers
# ---------------------------------------------------------------------------

def _find_prompt_span(spans):
    """Find the span containing prompt content attributes."""
    for s in spans:
        if s.attributes.get(f"{GenAIAttributes.GEN_AI_PROMPT}.0.content") is not None:
            return s
    return spans[0] if spans else None


def _find_completion_span(spans):
    """Find the span containing completion content attributes."""
    for s in spans:
        if s.attributes.get(f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content") is not None:
            return s
    return spans[0] if spans else None


def _collect_safety_events(spans):
    """Collect all safety-related span events across all spans."""
    events = []
    for s in spans:
        for e in s.events:
            if "fortifyroot.safety" in (e.name or ""):
                events.append(e)
    return events


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _register_safety_handlers():
    clear_safety_handlers()
    register_prompt_safety_handler(_prompt_handler)
    register_completion_safety_handler(_completion_handler)
    yield
    clear_safety_handlers()


@pytest.fixture
def llm():
    return ChatOpenAI(model="gpt-4o-mini", max_tokens=50)


# ---------------------------------------------------------------------------
# T4-A3: RegEx rule — PII.email in prompt
# ---------------------------------------------------------------------------

@pytest.mark.vcr
@pytest.mark.fr
def test_langchain_safety_regex_email_prompt(
    instrument_legacy, span_exporter, llm
):
    """Prompt contains an email address. Safety should detect and mask it."""
    llm.invoke([HumanMessage(
        content=(
            "Summarize the following customer inquiry from "
            "john.doe@example.com about their recent order."
        )
    )])

    spans = span_exporter.get_finished_spans()
    assert len(spans) >= 1
    span = _find_prompt_span(spans)

    prompt_content = span.attributes.get(
        f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"
    )
    assert prompt_content is not None
    assert "john.doe@example.com" not in prompt_content
    assert "[PII.email]" in prompt_content

    # Safety finding should be recorded as a span event
    safety_events = _collect_safety_events(spans)
    assert len(safety_events) >= 1
    event_attrs = safety_events[0].attributes
    assert event_attrs["fortifyroot.safety.rule_name"] == "PII.email"
    assert event_attrs["fortifyroot.safety.action"] == "MASK"
    assert event_attrs["fortifyroot.safety.location"] == "PROMPT"


# ---------------------------------------------------------------------------
# T4-A3: RegEx rule — PCI.credit_card in prompt
# ---------------------------------------------------------------------------

@pytest.mark.vcr
@pytest.mark.fr
def test_langchain_safety_regex_credit_card_prompt(
    instrument_legacy, span_exporter, llm
):
    """Prompt contains a credit card number. Safety should detect and mask it."""
    llm.invoke([HumanMessage(
        content="Check if order for card 4111 1111 1111 1111 was processed."
    )])

    spans = span_exporter.get_finished_spans()
    assert len(spans) >= 1
    span = _find_prompt_span(spans)

    prompt_content = span.attributes.get(
        f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"
    )
    assert prompt_content is not None
    assert "4111 1111 1111 1111" not in prompt_content
    assert "[PCI.credit_card]" in prompt_content

    safety_events = _collect_safety_events(spans)
    assert len(safety_events) >= 1
    found_cc = any(
        e.attributes.get("fortifyroot.safety.rule_name") == "PCI.credit_card"
        for e in safety_events
    )
    assert found_cc, "Expected PCI.credit_card safety finding"


# ---------------------------------------------------------------------------
# T4-A3: List rule — PII.competitor_org in prompt
# ---------------------------------------------------------------------------

@pytest.mark.vcr
@pytest.mark.fr
def test_langchain_safety_list_competitor_org_prompt(
    instrument_legacy, span_exporter, llm
):
    """Prompt mentions a competitor org from the blocklist."""
    llm.invoke([HumanMessage(
        content=(
            "Compare our product roadmap with Acme Corp's latest "
            "offering and Globex Industries' pricing strategy."
        )
    )])

    spans = span_exporter.get_finished_spans()
    assert len(spans) >= 1
    span = _find_prompt_span(spans)

    prompt_content = span.attributes.get(
        f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"
    )
    assert prompt_content is not None
    assert "acme corp" not in prompt_content.lower()
    assert "globex industries" not in prompt_content.lower()
    assert "[PII.competitor_org]" in prompt_content

    safety_events = _collect_safety_events(spans)
    found_org = any(
        e.attributes.get("fortifyroot.safety.rule_name") == "PII.competitor_org"
        for e in safety_events
    )
    assert found_org, "Expected PII.competitor_org safety finding"


# ---------------------------------------------------------------------------
# T4-A3: UDF rule — custom_compliance with 2 sub-rules in prompt
# ---------------------------------------------------------------------------

@pytest.mark.vcr
@pytest.mark.fr
def test_langchain_safety_udf_project_codenames_prompt(
    instrument_legacy, span_exporter, llm
):
    """Prompt contains two project codenames detected by UDF rule."""
    llm.invoke([HumanMessage(
        content=(
            "Prepare a status update for Project-Phoenix and "
            "Operation-Atlas including timeline and budget."
        )
    )])

    spans = span_exporter.get_finished_spans()
    assert len(spans) >= 1
    span = _find_prompt_span(spans)

    prompt_content = span.attributes.get(
        f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"
    )
    assert prompt_content is not None
    assert "[custom_compliance.project-phoenix]" in prompt_content
    assert "[custom_compliance.operation-atlas]" in prompt_content
    assert "Project-Phoenix" not in prompt_content
    assert "Operation-Atlas" not in prompt_content

    # UDF should produce 2 findings (one per codename)
    safety_events = _collect_safety_events(spans)
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
# T4-A3: Completion safety — email in LLM response
# ---------------------------------------------------------------------------

@pytest.mark.vcr
@pytest.mark.fr
def test_langchain_safety_completion_email_masked(
    instrument_legacy, span_exporter, llm
):
    """
    Ask the LLM to generate fictional contact info. Completion safety should
    detect and mask the LLM-generated email.
    """
    llm.invoke([HumanMessage(
        content=(
            "Create a fictional contact card. The person's name is "
            "Jane Smith, company is Widgets Inc, role is VP of Sales. "
            "Invent a plausible work email and phone. Format as:\n"
            "Name: ...\nEmail: ...\nPhone: ..."
        )
    )])

    spans = span_exporter.get_finished_spans()
    assert len(spans) >= 1
    span = _find_completion_span(spans)

    completion_content = span.attributes.get(
        f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content"
    )
    assert completion_content is not None, "Completion content should be recorded"
    assert "[PII.email]" in completion_content, (
        "LLM-generated email should be masked by completion safety"
    )


# ---------------------------------------------------------------------------
# T4-A3: Combined — multiple rule types in single prompt
# ---------------------------------------------------------------------------

@pytest.mark.vcr
@pytest.mark.fr
def test_langchain_safety_combined_rules_prompt(
    instrument_legacy, span_exporter, llm
):
    """Prompt triggers all rule types: RegEx (email), List (org), UDF (codename)."""
    llm.invoke([HumanMessage(
        content=(
            "Send the Project-Phoenix briefing to contact@acme-corp.com "
            "and CC the Initech team."
        )
    )])

    spans = span_exporter.get_finished_spans()
    assert len(spans) >= 1
    span = _find_prompt_span(spans)

    prompt_content = span.attributes.get(
        f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"
    )
    assert prompt_content is not None
    # All three rule types should have produced mask tokens
    assert "[PII.email]" in prompt_content
    assert "[PII.competitor_org]" in prompt_content
    assert "[custom_compliance.project-phoenix]" in prompt_content
    # Original PII should be absent
    assert "contact@acme-corp.com" not in prompt_content
    assert "initech" not in prompt_content.lower()
    assert "Project-Phoenix" not in prompt_content

    # Verify all 3 rule types fired as span events
    safety_events = _collect_safety_events(spans)
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
# T4-A3: Streaming with PII
# ---------------------------------------------------------------------------

@pytest.mark.vcr
@pytest.mark.fr
def test_langchain_safety_streaming_with_pii(
    instrument_legacy, span_exporter, llm
):
    """
    Verify prompt safety masking with PII in prompt.

    NOTE: The VCR cassette was recorded with a non-streaming call, so we use
    llm.invoke() here to avoid a stream/non-stream mismatch.  Streaming-
    specific safety behaviour is verified in mock-based test_safety_hooks.py.
    """
    llm.invoke([HumanMessage(
        content=(
            "Acknowledge receipt of message from "
            "alice.jones@example.org about Project-Phoenix."
        )
    )])

    spans = span_exporter.get_finished_spans()
    assert len(spans) >= 1
    span = _find_prompt_span(spans)

    prompt_content = span.attributes.get(
        f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"
    )
    assert prompt_content is not None
    assert "alice.jones@example.org" not in prompt_content
    assert "[PII.email]" in prompt_content

    safety_events = _collect_safety_events(spans)
    assert len(safety_events) >= 1


# ===========================================================================
# ALLOW action tests
# ===========================================================================

@pytest.fixture
def _set_allow_action():
    global _SAFETY_ACTION
    _SAFETY_ACTION = "ALLOW"
    yield
    _SAFETY_ACTION = "MASK"


@pytest.mark.vcr
@pytest.mark.fr
def test_langchain_safety_allow_email_not_masked(
    _set_allow_action, instrument_legacy, span_exporter, llm
):
    """ALLOW action: email detected but text passes through unchanged."""
    llm.invoke([HumanMessage(
        content=(
            "Forward this message from alice.jones@example.org "
            "to the support team."
        )
    )])

    spans = span_exporter.get_finished_spans()
    assert len(spans) >= 1
    span = _find_prompt_span(spans)

    prompt_content = span.attributes.get(
        f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"
    )
    assert prompt_content is not None
    assert "alice.jones@example.org" in prompt_content
    assert "[PII.email]" not in prompt_content

    # But findings should STILL be recorded as span events
    safety_events = _collect_safety_events(spans)
    assert len(safety_events) >= 1, "ALLOW should still emit safety findings"
    event_attrs = safety_events[0].attributes
    assert event_attrs["fortifyroot.safety.rule_name"] == "PII.email"
    assert event_attrs["fortifyroot.safety.action"] == "ALLOW"
    assert event_attrs["fortifyroot.safety.location"] == "PROMPT"


@pytest.mark.vcr
@pytest.mark.fr
def test_langchain_safety_allow_credit_card_not_masked(
    _set_allow_action, instrument_legacy, span_exporter, llm
):
    """ALLOW action: credit card detected but text passes through unchanged."""
    llm.invoke([HumanMessage(
        content="Verify payment for card 4111 1111 1111 1111 was received."
    )])

    spans = span_exporter.get_finished_spans()
    assert len(spans) >= 1
    span = _find_prompt_span(spans)

    prompt_content = span.attributes.get(
        f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"
    )
    assert prompt_content is not None
    assert "4111 1111 1111 1111" in prompt_content
    assert "[PCI.credit_card]" not in prompt_content

    # Findings still recorded
    safety_events = _collect_safety_events(spans)
    found_cc = any(
        e.attributes.get("fortifyroot.safety.rule_name") == "PCI.credit_card"
        and e.attributes.get("fortifyroot.safety.action") == "ALLOW"
        for e in safety_events
    )
    assert found_cc, "Expected PCI.credit_card finding with ALLOW action"


# ---------------------------------------------------------------------------
# T4-A4: LangGraph workflow with PII in nodes
# ---------------------------------------------------------------------------

@pytest.mark.vcr
@pytest.mark.fr
def test_langgraph_workflow_with_pii(instrument_legacy, span_exporter):
    """
    LangGraph StateGraph with PII in messages. Verify workflow span exists
    and safety masking applies at the LLM call node.
    """
    from langgraph.graph import StateGraph, START, END
    from typing import TypedDict, Annotated
    import operator

    class State(TypedDict):
        messages: Annotated[list, operator.add]

    llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=50)

    def llm_node(state: State) -> dict:
        """Node that calls LLM with PII-containing message."""
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    graph = StateGraph(State)
    graph.add_node("llm_call", llm_node)
    graph.add_edge(START, "llm_call")
    graph.add_edge("llm_call", END)
    app = graph.compile()

    app.invoke({
        "messages": [
            HumanMessage(
                content=(
                    "Reply to john.doe@example.com about Project-Phoenix status."
                )
            )
        ]
    })

    spans = span_exporter.get_finished_spans()
    assert len(spans) >= 1

    # Find the OpenAI chat span (where safety applies)
    chat_spans = [
        s for s in spans
        if s.attributes.get(GenAIAttributes.GEN_AI_PROMPT + ".0.content") is not None
    ]
    assert len(chat_spans) >= 1, "Should have at least one LLM call span"

    # Verify PII was masked in the prompt
    prompt_content = chat_spans[0].attributes.get(
        f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"
    )
    assert prompt_content is not None
    assert "john.doe@example.com" not in prompt_content
    assert "[PII.email]" in prompt_content

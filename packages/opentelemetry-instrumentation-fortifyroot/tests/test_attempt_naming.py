from types import SimpleNamespace

from opentelemetry.instrumentation.fortifyroot import attempt_naming
from opentelemetry.instrumentation.fortifyroot.attempt_naming import (
    FR_ATTEMPT_IS_RETRY_KEY,
    FR_ATTEMPT_NUMBER_KEY,
    FR_SPAN_ROLE_KEY,
    FR_SPAN_ROLE_LLM_ATTEMPT,
    clear_attempt_counters_for_test,
    first_llm_attempt,
    llm_attempt_attributes,
    next_llm_attempt,
)


class _FakeSpan:
    def __init__(self, trace_id: int, span_id: int, is_valid: bool = True):
        self._ctx = SimpleNamespace(
            trace_id=trace_id,
            span_id=span_id,
            is_valid=is_valid,
        )

    def get_span_context(self):
        return self._ctx


def setup_function():
    clear_attempt_counters_for_test()


def teardown_function():
    clear_attempt_counters_for_test()


def test_next_llm_attempt_counts_per_parent():
    parent = _FakeSpan(trace_id=0xAA, span_id=0x01)

    assert next_llm_attempt(parent, "fortifyroot.openai") == (
        "fortifyroot.openai.attempt_1",
        1,
        False,
    )
    assert next_llm_attempt(parent, "fortifyroot.openai") == (
        "fortifyroot.openai.attempt_2",
        2,
        True,
    )
    assert next_llm_attempt(parent, "fortifyroot.openai") == (
        "fortifyroot.openai.attempt_3",
        3,
        True,
    )


def test_next_llm_attempt_isolated_by_parent_context():
    parent_a = _FakeSpan(trace_id=0xAA, span_id=0x01)
    parent_b = _FakeSpan(trace_id=0xAA, span_id=0x02)

    assert next_llm_attempt(parent_a, "fortifyroot.openai")[1:] == (1, False)
    assert next_llm_attempt(parent_b, "fortifyroot.openai")[1:] == (1, False)
    assert next_llm_attempt(parent_a, "fortifyroot.openai")[1:] == (2, True)


def test_invalid_parent_falls_back_to_first_attempt_without_counter():
    parent = _FakeSpan(trace_id=0, span_id=0, is_valid=False)

    assert next_llm_attempt(parent, "fortifyroot.openai") == (
        "fortifyroot.openai.attempt_1",
        1,
        False,
    )
    assert next_llm_attempt(parent, "fortifyroot.openai") == (
        "fortifyroot.openai.attempt_1",
        1,
        False,
    )


def test_ttl_eviction_resets_parent_counter(monkeypatch):
    parent = _FakeSpan(trace_id=0xAA, span_id=0x01)

    monkeypatch.setattr(attempt_naming, "_now", lambda: 100.0)
    assert next_llm_attempt(parent, "fortifyroot.openai")[1:] == (1, False)
    assert next_llm_attempt(parent, "fortifyroot.openai")[1:] == (2, True)

    monkeypatch.setattr(
        attempt_naming,
        "_now",
        lambda: 100.0 + attempt_naming._COUNTER_TTL_SEC + 1.0,
    )
    assert next_llm_attempt(parent, "fortifyroot.openai") == (
        "fortifyroot.openai.attempt_1",
        1,
        False,
    )


def test_cap_eviction_drops_oldest_parent_counter(monkeypatch, caplog):
    parent_a = _FakeSpan(trace_id=0xAA, span_id=0x01)
    parent_b = _FakeSpan(trace_id=0xAA, span_id=0x02)
    parent_c = _FakeSpan(trace_id=0xAA, span_id=0x03)

    monkeypatch.setattr(attempt_naming, "_COUNTER_MAX", 2)
    monkeypatch.setattr(attempt_naming, "_COUNTER_EVICT_BATCH", 1)
    now = 100.0
    monkeypatch.setattr(attempt_naming, "_now", lambda: now)

    assert next_llm_attempt(parent_a, "fortifyroot.openai")[1:] == (1, False)
    now += 1.0
    assert next_llm_attempt(parent_a, "fortifyroot.openai")[1:] == (2, True)
    now += 1.0
    assert next_llm_attempt(parent_b, "fortifyroot.openai")[1:] == (1, False)
    now += 1.0

    with caplog.at_level("WARNING", logger=attempt_naming.logger.name):
        assert next_llm_attempt(parent_c, "fortifyroot.openai")[1:] == (1, False)

    assert "cap-evicted 1 parent attempt counters" in caplog.text

    now += 1.0
    assert next_llm_attempt(parent_a, "fortifyroot.openai") == (
        "fortifyroot.openai.attempt_1",
        1,
        False,
    )


def test_first_llm_attempt_is_conservative():
    assert first_llm_attempt("fortifyroot.langchain") == (
        "fortifyroot.langchain.attempt_1",
        1,
        False,
    )


def test_llm_attempt_attributes():
    attrs = llm_attempt_attributes(2, True)

    assert attrs[FR_SPAN_ROLE_KEY] == FR_SPAN_ROLE_LLM_ATTEMPT
    assert attrs[FR_ATTEMPT_NUMBER_KEY] == 2
    assert attrs[FR_ATTEMPT_IS_RETRY_KEY] is True

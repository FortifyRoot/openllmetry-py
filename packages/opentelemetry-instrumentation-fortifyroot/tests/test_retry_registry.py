"""Tests for the §4.7.1 framework-attempt registry.

Required tests per retry-loop design notes §4.7.1:
  - re-entrancy (one thread can hold multiple tokens, suppression
    only stops when ALL are unregistered)
  - stale-entry eviction (TTL-based, runs in-band on read path)
  - cap-eviction (when total entries exceed _REGISTRY_MAX)
  - parent-end cleanup (clear_for_thread drops orphan tokens)
  - thread-ID reuse (registry survives a TID being recycled by
    the OS for a subsequent thread, by virtue of token uniqueness)
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from opentelemetry.instrumentation.fortifyroot import (
    clear_framework_attempts_for_thread,
    is_framework_owned,
    register_framework_attempt,
    unregister_framework_attempt,
)
from opentelemetry.instrumentation.fortifyroot import retry_registry


@pytest.fixture(autouse=True)
def reset_registry():
    """Each test starts with an empty registry."""
    retry_registry._reset_for_test()
    yield
    retry_registry._reset_for_test()


def test_register_unregister_basic():
    """Registering then unregistering a single token leaves the
    registry empty and is_framework_owned() False."""
    assert not is_framework_owned()
    token = register_framework_attempt()
    assert isinstance(token, str) and len(token) > 0
    assert is_framework_owned()
    unregister_framework_attempt(token)
    assert not is_framework_owned()


def test_register_returns_unique_tokens():
    """Each register call mints a fresh token (uniqueness invariant)."""
    tokens = {register_framework_attempt() for _ in range(50)}
    assert len(tokens) == 50, "all tokens must be unique"


def test_reentrancy_one_thread_multiple_tokens():
    """Re-entrancy: one thread can register multiple tokens (e.g.
    nested framework attempts). Suppression remains True until ALL
    tokens are unregistered. This is the canonical case the
    refcount-via-token design solves vs the rejected set[int]
    design (round-3 Blocker disposition)."""
    t1 = register_framework_attempt()
    t2 = register_framework_attempt()
    t3 = register_framework_attempt()

    assert is_framework_owned()

    unregister_framework_attempt(t1)
    assert is_framework_owned(), "still owned: t2 + t3 outstanding"

    unregister_framework_attempt(t2)
    assert is_framework_owned(), "still owned: t3 outstanding"

    unregister_framework_attempt(t3)
    assert not is_framework_owned(), "fully released after all tokens removed"


def test_unregister_idempotent_on_unknown_token():
    """Unregistering an unknown / already-removed token is a no-op
    (defensive contract — handles double-unregister from sync+async
    callbacks both firing)."""
    t = register_framework_attempt()
    unregister_framework_attempt(t)
    # Should not raise, should not affect state.
    unregister_framework_attempt(t)
    unregister_framework_attempt("nonexistent-token")
    unregister_framework_attempt(None)
    assert not is_framework_owned()


def test_per_thread_isolation():
    """Tokens registered on thread A do NOT make thread B "owned"."""
    main_tid = threading.get_ident()
    register_framework_attempt()
    assert is_framework_owned(main_tid)

    foreign_tid_seen: dict[str, bool] = {}

    def worker():
        # Different thread → fresh TID → no tokens registered.
        foreign_tid_seen["owned"] = is_framework_owned()

    t = threading.Thread(target=worker)
    t.start()
    t.join()

    assert foreign_tid_seen["owned"] is False, (
        "thread B sees its own (empty) bucket, not thread A's"
    )


def test_stale_entry_eviction_via_ttl(monkeypatch):
    """Tokens older than _REGISTRY_STALE_TTL_SEC are evicted on the
    next is_framework_owned() call. This is the load-bearing
    in-band eviction path; without it a leaked
    token (framework crashed) would suppress emission forever."""
    # Shrink the TTL so the test runs in real time.
    monkeypatch.setattr(retry_registry, "_REGISTRY_STALE_TTL_SEC", 0.1)

    register_framework_attempt()
    assert is_framework_owned()

    time.sleep(0.15)  # past the TTL

    # is_framework_owned() runs eviction in-band.
    assert not is_framework_owned(), "stale token must be evicted on read path"


def test_stale_eviction_with_no_intervening_registration(monkeypatch):
    """The boundary case from §4.7.1: register, sleep past TTL, then
    check is_framework_owned() WITHOUT another registration in
    between. This is the case that the in-band eviction design
    explicitly fixes — eviction happens on every read, not only on
    write paths."""
    monkeypatch.setattr(retry_registry, "_REGISTRY_STALE_TTL_SEC", 0.1)

    register_framework_attempt()
    time.sleep(0.15)

    # No further register() calls. Eviction MUST happen in
    # is_framework_owned itself.
    assert not is_framework_owned()
    # Registry should be fully empty after eviction.
    n_tids, n_tokens = retry_registry._registry_size_for_test()
    assert n_tids == 0
    assert n_tokens == 0


def test_cap_eviction_at_max(monkeypatch):
    """When total entries exceed _REGISTRY_MAX, _REGISTRY_EVICT_BATCH
    oldest entries are dropped. Verifies the explicit math:
    register MAX+1 tokens, expect MAX+1 - EVICT_BATCH remaining."""
    # Use the real cap (4096) but evict in smaller batches to keep
    # the test fast. Force eviction by exceeding the cap.
    monkeypatch.setattr(retry_registry, "_REGISTRY_MAX", 16)
    monkeypatch.setattr(retry_registry, "_REGISTRY_EVICT_BATCH", 4)

    for _ in range(17):
        register_framework_attempt()
    # On the 17th register, cap is exceeded → 4 oldest dropped.
    _, n_tokens = retry_registry._registry_size_for_test()
    assert n_tokens == 17 - 4, (
        f"expected 13 tokens after cap-eviction, got {n_tokens}"
    )


def test_clear_for_thread_drops_all_tokens():
    """clear_for_thread() drops ALL tokens for the current TID.
    Used as a parent-span-end orphan-cleanup primitive: when the
    framework wrapper's parent span ends, any tokens still
    registered are by definition orphans."""
    t1 = register_framework_attempt()
    t2 = register_framework_attempt()
    assert is_framework_owned()

    dropped = clear_framework_attempts_for_thread()
    assert dropped == 2
    assert not is_framework_owned()

    # Subsequent unregister of an already-cleared token is a safe no-op.
    unregister_framework_attempt(t1)
    unregister_framework_attempt(t2)


def test_thread_id_reuse_does_not_leak_state():
    """When an OS thread terminates and its TID is recycled by the
    next thread, the new thread sees a clean bucket (because the
    terminated thread's parent-end cleanup or TTL eviction
    cleared its tokens before reuse).

    The registry tolerates TID reuse simply because tokens are
    UUIDs — a recycled TID with no tokens has an empty bucket
    (or is missing from the registry entirely), so
    is_framework_owned returns False naturally.
    """
    # Drive thread 1 to register + clear, simulating a clean shutdown.
    state: dict[str, int] = {}

    def thread_one():
        state["tid"] = threading.get_ident()
        register_framework_attempt()
        clear_framework_attempts_for_thread()

    t = threading.Thread(target=thread_one)
    t.start()
    t.join()

    # Now drive thread 2; if the OS happens to give it the same TID
    # (we can't deterministically force this, but we can verify the
    # invariant holds either way), it must not see thread 1's state.
    seen: dict[str, bool] = {}

    def thread_two():
        # If TID was reused, the bucket should still be empty
        # because thread 1 cleared it. If TID is fresh, ditto.
        seen["owned_at_start"] = is_framework_owned()
        register_framework_attempt()
        seen["owned_after_register"] = is_framework_owned()
        clear_framework_attempts_for_thread()

    t2 = threading.Thread(target=thread_two)
    t2.start()
    t2.join()

    assert seen["owned_at_start"] is False
    assert seen["owned_after_register"] is True


def test_unregister_works_from_different_thread_than_register():
    """Regression guard for cross-thread unregister behavior.

    A
    framework's start callback and terminal callback can run on
    different OS threads (e.g. asyncio dispatching success/failure
    callbacks to a worker thread). The original implementation used
    ``threading.get_ident()`` in BOTH register and unregister, so a
    cross-thread unregister silently failed and the originating
    thread stayed "framework-owned" until TTL.

    Fix: token → originating-TID reverse index, used by unregister
    to find the right bucket regardless of which thread runs it.
    """
    main_tid = threading.get_ident()

    captured: dict[str, str] = {}

    def thread_a_register():
        captured["token"] = register_framework_attempt()
        captured["registered_tid"] = str(threading.get_ident())

    ta = threading.Thread(target=thread_a_register)
    ta.start()
    ta.join()

    # Sanity: thread A registered. main thread is not framework-owned;
    # the originating thread A's TID *is* owned.
    assert not is_framework_owned(main_tid)
    registered_tid = int(captured["registered_tid"])
    assert is_framework_owned(registered_tid)

    # Now unregister from a DIFFERENT thread (could be main, could be
    # a third thread — anywhere except the original registrant).
    unregister_framework_attempt(captured["token"])

    # The originating TID's bucket must now be empty — i.e. NOT
    # falsely reported as owned. Without the fix, this assertion
    # fails until the 60s TTL kicks in.
    assert not is_framework_owned(registered_tid), (
        "cross-thread unregister MUST clean up the originating TID's bucket; "
        "otherwise direct-SDK suppression on the original thread persists "
        "incorrectly until TTL eviction"
    )

    # Registry should be fully empty.
    n_tids, n_tokens = retry_registry._registry_size_for_test()
    assert n_tids == 0 and n_tokens == 0


def test_concurrent_register_unregister_thread_safety():
    """Many threads registering/unregistering concurrently must not
    crash, lose tokens, or report inconsistent state. Smoke-tests the
    locking around _REGISTRY."""
    N_THREADS = 8
    N_OPS_PER_THREAD = 50

    errors: list[Exception] = []

    def worker():
        try:
            tokens = []
            for _ in range(N_OPS_PER_THREAD):
                tokens.append(register_framework_attempt())
            assert is_framework_owned()
            for tok in tokens:
                unregister_framework_attempt(tok)
            assert not is_framework_owned()
        except Exception as e:  # pragma: no cover
            errors.append(e)

    with ThreadPoolExecutor(max_workers=N_THREADS) as pool:
        futures = [pool.submit(worker) for _ in range(N_THREADS)]
        for f in futures:
            f.result()

    assert not errors, f"concurrent ops raised: {errors}"
    n_tids, n_tokens = retry_registry._registry_size_for_test()
    assert n_tokens == 0, f"all tokens should be unregistered, got {n_tokens}"

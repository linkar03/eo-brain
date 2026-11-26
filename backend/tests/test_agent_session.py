"""Unit tests for eo_brain.agent.session (in-memory store with TTL)."""

import time

import pytest

from eo_brain.agent import session as session_mod
from eo_brain.agent.session import SessionStore


@pytest.fixture
def clock(monkeypatch):
    # Session timestamps default to the real time.time (bound at class
    # definition), so the fake clock starts from the real "now".
    now = {"t": time.time()}
    monkeypatch.setattr(session_mod.time, "time", lambda: now["t"])
    return now


def test_get_or_create_returns_same_session(clock):
    store = SessionStore()
    a = store.get_or_create("s1")
    assert store.get_or_create("s1") is a
    assert store.get_or_create("s2") is not a


def test_context_is_sticky(clock):
    store = SessionStore()
    s = store.get_or_create("s1")
    s.set_context(last_location="Firenze", last_algorithm="ndvi")
    s.set_context(last_algorithm="ndwi")
    again = store.get_or_create("s1")
    assert again.get_context("last_location") == "Firenze"
    assert again.get_context("last_algorithm") == "ndwi"
    assert again.get_context("missing", "default") == "default"


def test_expired_session_is_replaced(clock):
    store = SessionStore()
    s = store.get_or_create("s1")
    s.set_context(last_location="Firenze")
    clock["t"] += session_mod._SESSION_TTL_SEC + 1
    fresh = store.get_or_create("s1")
    assert fresh is not s
    assert fresh.get_context("last_location") is None


def test_activity_refreshes_ttl(clock):
    store = SessionStore()
    s = store.get_or_create("s1")
    clock["t"] += session_mod._SESSION_TTL_SEC - 10
    s.append("user", "hi")
    clock["t"] += 20
    assert store.get_or_create("s1") is s


def test_other_expired_sessions_are_evicted(clock):
    store = SessionStore()
    store.get_or_create("old")
    clock["t"] += session_mod._SESSION_TTL_SEC + 1
    store.get_or_create("new")
    assert "old" not in store._sessions


def test_history_is_capped(clock):
    s = SessionStore().get_or_create("s1")
    for i in range(session_mod._MAX_HISTORY_TURNS + 5):
        s.append("user", f"m{i}")
    assert len(s.history) == session_mod._MAX_HISTORY_TURNS
    assert s.history[-1].text == f"m{session_mod._MAX_HISTORY_TURNS + 4}"


def test_history_format(clock):
    s = SessionStore().get_or_create("s1")
    s.append("user", "ndvi firenze")
    s.append("model", "ok")
    assert s.to_gemini_history() == [
        {"role": "user", "parts": ["ndvi firenze"]},
        {"role": "model", "parts": ["ok"]},
    ]


def test_delete(clock):
    store = SessionStore()
    a = store.get_or_create("s1")
    store.delete("s1")
    store.delete("never-existed")
    assert store.get_or_create("s1") is not a

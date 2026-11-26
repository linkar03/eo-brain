"""
eo_brain/agent/session.py - Conversation session store.

In-memory implementation with TTL eviction. Swap to Redis by
implementing the same interface (get/append/clear).
"""

from __future__ import annotations

import threading
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Sessions expire after 30min of inactivity
_SESSION_TTL_SEC = 30 * 60

# Hard cap on history length to avoid blowing context window
_MAX_HISTORY_TURNS = 40


@dataclass
class Turn:
    role: str  # "user" | "model"
    text: str
    timestamp: float = field(default_factory=time.time)
    # Metadata the agent can carry forward (e.g. last bbox, last algorithm)
    metadata: Optional[dict] = None


@dataclass
class Session:
    session_id: str
    history: List[Turn] = field(default_factory=list)
    context: Dict = field(default_factory=dict)  # sticky context across turns
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)

    def append(self, role: str, text: str, metadata: dict = None):
        self.history.append(Turn(role=role, text=text, metadata=metadata))
        self.last_active = time.time()

        # Trim oldest turns if we exceed the cap (keep system-relevant context)
        if len(self.history) > _MAX_HISTORY_TURNS:
            self.history = self.history[-_MAX_HISTORY_TURNS:]

    def set_context(self, **kwargs):
        """Persist sticky state (last_bbox, last_algorithm, etc.)."""
        self.context.update(kwargs)

    def get_context(self, key: str, default=None):
        return self.context.get(key, default)

    def to_gemini_history(self) -> List[dict]:
        """Format for Gemini's chat history parameter."""
        return [
            {"role": t.role, "parts": [t.text]}
            for t in self.history
        ]

    @property
    def is_expired(self) -> bool:
        return (time.time() - self.last_active) > _SESSION_TTL_SEC


class SessionStore:
    """Thread-safe in-memory session store with lazy eviction."""

    def __init__(self):
        self._sessions: Dict[str, Session] = {}
        self._lock = threading.Lock()

    def get_or_create(self, session_id: str) -> Session:
        with self._lock:
            self._evict_expired()

            if session_id in self._sessions:
                session = self._sessions[session_id]
                if not session.is_expired:
                    return session
                # Expired, nuke it
                del self._sessions[session_id]

            session = Session(session_id=session_id)
            self._sessions[session_id] = session
            logger.debug(f"New session: {session_id}")
            return session

    def delete(self, session_id: str):
        with self._lock:
            self._sessions.pop(session_id, None)

    def _evict_expired(self):
        expired = [
            sid for sid, s in self._sessions.items() if s.is_expired
        ]
        for sid in expired:
            del self._sessions[sid]
        if expired:
            logger.debug(f"Evicted {len(expired)} expired sessions")


# Module-level singleton
session_store = SessionStore()
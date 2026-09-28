# ============================================================
# FILE: gismol/nlp/chat_manager.py
# ============================================================
"""Multi-session dialogue management on top of :class:`ChatModule`."""

from __future__ import annotations

import uuid
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .chat_module import ChatModule, ChatResponse

__all__ = ["ChatSession", "ChatManager", "Turn"]


class Turn:
    """One user/assistant exchange."""

    def __init__(self, user: str, response: ChatResponse, index: int = 0):
        self.user = user
        self.response = response
        self.index = index

    def to_dict(self):
        return {"index": self.index, "user": self.user,
                "assistant": self.response.text,
                "intent": self.response.intent.name}

    def __repr__(self):
        return f"Turn({self.index}, intent={self.response.intent.name!r})"


class ChatSession:
    """A stateful conversation bound to a :class:`ChatModule`."""

    def __init__(self, session_id: Optional[str] = None, module: Optional[ChatModule] = None,
                 metadata: Optional[Mapping[str, Any]] = None):
        self.id = session_id or str(uuid.uuid4())
        self.module = module or ChatModule()
        self.turns: List[Turn] = []
        self.metadata: Dict[str, Any] = dict(metadata or {})

    def say(self, text: str) -> ChatResponse:
        response = self.module.process(text)
        self.turns.append(Turn(text, response, len(self.turns)))
        return response

    @property
    def coh(self):
        return self.module.coh

    @property
    def last(self) -> Optional[Turn]:
        return self.turns[-1] if self.turns else None

    def history(self, last: Optional[int] = None) -> List[Turn]:
        return self.turns[-last:] if last else list(self.turns)

    def transcript(self) -> str:
        lines = []
        for turn in self.turns:
            lines.append(f"user> {turn.user}")
            lines.append(f"coh > {turn.response.text}")
        return "\n".join(lines)

    def reset(self):
        self.module.reset()
        self.turns.clear()

    def __repr__(self):
        return f"ChatSession({self.id[:8]}, turns={len(self.turns)})"


class ChatManager:
    """Registry of chat sessions."""

    def __init__(self, module_factory=None):
        self.sessions: Dict[str, ChatSession] = {}
        self.module_factory = module_factory or ChatModule

    def new_session(self, session_id: Optional[str] = None, **kwargs) -> ChatSession:
        session = ChatSession(session_id, self.module_factory(), kwargs or None)
        self.sessions[session.id] = session
        return session

    def get(self, session_id: str) -> ChatSession:
        if session_id not in self.sessions:
            raise KeyError(f"unknown session {session_id!r}")
        return self.sessions[session_id]

    def process(self, session_id: str, text: str) -> ChatResponse:
        session = self.sessions.get(session_id)
        if session is None:
            session = self.new_session(session_id)
        return session.say(text)

    def close(self, session_id: str):
        self.sessions.pop(session_id, None)

    def list_sessions(self) -> List[str]:
        return list(self.sessions)

    def __len__(self):
        return len(self.sessions)

    def __repr__(self):
        return f"ChatManager(sessions={len(self.sessions)})"
    
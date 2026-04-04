"""Session and conversation history management."""
from typing import Dict, List
from datetime import datetime


class ConversationStore:
    """In-memory store for session-based conversation history."""

    def __init__(self, max_history: int = 6):
        """Initialize conversation store.
        
        Args:
            max_history: Maximum number of messages to keep per session.
        """
        self.max_history = max_history
        self.sessions: Dict[str, List[dict]] = {}

    def get_history(self, session_id: str) -> List[dict]:
        """Get conversation history for a session.
        
        Args:
            session_id: Session identifier.
        
        Returns:
            List of message dicts with 'role', 'content', 'timestamp'.
        """
        return self.sessions.get(session_id, [])

    def add_message(self, session_id: str, role: str, content: str) -> None:
        """Add a message to session history.
        
        Args:
            session_id: Session identifier.
            role: Message role ('user' or 'assistant').
            content: Message content.
        """
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat()
        }
        self.sessions[session_id].append(message)
        
        # Trim to max history size
        if len(self.sessions[session_id]) > self.max_history:
            self.sessions[session_id] = self.sessions[session_id][-self.max_history:]

    def clear_session(self, session_id: str) -> None:
        """Clear history for a session.
        
        Args:
            session_id: Session identifier.
        """
        if session_id in self.sessions:
            del self.sessions[session_id]

    def get_last_n_messages(self, session_id: str, n: int = 6) -> List[dict]:
        """Get the last N messages from session history.
        
        Args:
            session_id: Session identifier.
            n: Number of messages to return.
        
        Returns:
            List of last N messages.
        """
        history = self.get_history(session_id)
        return history[-n:] if history else []


# Global conversation store instance
_global_store = ConversationStore(max_history=6)


def get_conversation_store() -> ConversationStore:
    """Get global conversation store instance."""
    return _global_store

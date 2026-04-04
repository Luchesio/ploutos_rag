"""
conversation_store.py
─────────────────────
MongoDB-backed conversation memory for Femi.

Responsibilities:
  • Save every (question, answer) turn under a session_id
  • Retrieve the last N turns for context injection
  • Record a flag when the greeting has been sent for a session
    (so it's only shown once per new session)

Collection schema  — db.conversations
  {
    session_id : str,          # client-supplied or auto-generated UUID
    turns      : [             # ordered list, newest appended last
      {
        role     : "user" | "assistant",
        content  : str,
        timestamp: datetime
      }
    ],
    greeted    : bool,         # True once the welcome message has been sent
    created_at : datetime,
    updated_at : datetime
  }
"""

from datetime import datetime, timezone
from typing import Optional
import logging

from motor.motor_asyncio import AsyncIOMotorClient
from app.core.config import settings

logger = logging.getLogger(__name__)

_client: Optional[AsyncIOMotorClient] = None


def _get_client() -> AsyncIOMotorClient:
    """Return a singleton Motor client (created lazily)."""
    global _client
    if _client is None:
        _client = AsyncIOMotorClient(settings.MONGODB_URI)
        logger.info("MongoDB client initialised")
    return _client


class ConversationStore:
    """Async MongoDB helper for per-session conversation history."""

    def __init__(self):
        client = _get_client()
        db = client[settings.MONGODB_DB_NAME]
        self.col = db[settings.MONGODB_COLLECTION]

    # ------------------------------------------------------------------ #
    # Greeting flag
    # ------------------------------------------------------------------ #

    async def has_been_greeted(self, session_id: str) -> bool:
        """Return True if the welcome message was already sent this session."""
        doc = await self.col.find_one(
            {"session_id": session_id},
            {"greeted": 1}
        )
        return bool(doc and doc.get("greeted", False))

    async def mark_greeted(self, session_id: str) -> None:
        """Set greeted=True for the session (upsert-safe)."""
        now = datetime.now(timezone.utc)
        await self.col.update_one(
            {"session_id": session_id},
            {
                "$set": {"greeted": True, "updated_at": now},
                "$setOnInsert": {"created_at": now, "turns": []},
            },
            upsert=True,
        )

    # ------------------------------------------------------------------ #
    # Turn management
    # ------------------------------------------------------------------ #

    async def save_turn(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
    ) -> None:
        """Append a user+assistant turn pair to the session document."""
        now = datetime.now(timezone.utc)
        turns = [
            {"role": "user", "content": user_message, "timestamp": now},
            {"role": "assistant", "content": assistant_message, "timestamp": now},
        ]
        await self.col.update_one(
            {"session_id": session_id},
            {
                "$push": {"turns": {"$each": turns}},
                "$set": {"updated_at": now},
                "$setOnInsert": {"created_at": now, "greeted": False},
            },
            upsert=True,
        )
        logger.debug(f"Saved turn for session '{session_id}'")

    async def get_recent_turns(
        self,
        session_id: str,
        last_n: int = 5,
    ) -> list[dict]:
        """
        Return the last `last_n` user+assistant PAIRS (i.e. up to last_n*2 turn
        objects) for a session, oldest first, ready for context injection.

        Each item: {"role": "user"|"assistant", "content": str}
        """
        doc = await self.col.find_one(
            {"session_id": session_id},
            {"turns": 1}
        )
        if not doc or not doc.get("turns"):
            return []

        # Each pair = 2 turn objects; slice the tail
        turns = doc["turns"][-(last_n * 2):]
        return [{"role": t["role"], "content": t["content"]} for t in turns]

    # ------------------------------------------------------------------ #
    # Utility
    # ------------------------------------------------------------------ #

    async def delete_session(self, session_id: str) -> None:
        """Wipe a session (useful for testing or explicit resets)."""
        await self.col.delete_one({"session_id": session_id})
        logger.info(f"Session '{session_id}' deleted")
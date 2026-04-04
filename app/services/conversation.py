"""
Conversation Memory Service
Stores chat history in MongoDB so Femi remembers the last 5 turns.
"""
import motor.motor_asyncio
from datetime import datetime
from typing import List, Dict, Optional
import uuid
import logging

from app.core.config import settings

logger = logging.getLogger(__name__)


class ConversationService:
    def __init__(self):
        self.client = motor.motor_asyncio.AsyncIOMotorClient(settings.MONGO_URI)
        self.db = self.client[settings.MONGO_DB_NAME]
        self.collection = self.db[settings.CONVERSATION_COLLECTION]
        logger.info(f"✅ ConversationService connected to MongoDB → {settings.MONGO_DB_NAME}.{settings.CONVERSATION_COLLECTION}")

    async def ensure_conversation(self, conversation_id: Optional[str] = None) -> str:
        """Return a valid conversation_id. Creates a new one if none provided."""
        if not conversation_id:
            conversation_id = str(uuid.uuid4())

        # Ensure the document exists
        await self.collection.update_one(
            {"conversation_id": conversation_id},
            {"$setOnInsert": {"messages": [], "created_at": datetime.utcnow()}},
            upsert=True
        )
        return conversation_id

    async def get_history(self, conversation_id: str, limit: int = 5) -> List[Dict]:
        """Return last N turns (≈ last 10 messages) for context."""
        doc = await self.collection.find_one({"conversation_id": conversation_id})
        if not doc or "messages" not in doc:
            return []
        messages = doc["messages"]
        # Return enough messages for ~5 full turns
        return messages[-(limit * 2):]

    async def add_message(self, conversation_id: str, role: str, content: str):
        """Save one message (user or assistant)."""
        if not conversation_id:
            return

        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow()
        }

        await self.collection.update_one(
            {"conversation_id": conversation_id},
            {"$push": {"messages": message}},
            upsert=True
        )
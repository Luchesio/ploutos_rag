"""
vector_store.py — Qdrant Cloud vector store for Femi RAG Service.

Replaces ChromaDB (which requires hnswlib / libgomp.so.1 — unavailable on
Vercel/Lambda) with Qdrant Cloud, a hosted vector database with a free tier.

Interface is identical to the original ChromaDB implementation so no other
files need to change.

Setup:
  1. Create a free cluster at https://cloud.qdrant.io
  2. Copy your cluster URL and API key into .env and Vercel env vars:
       QDRANT_URL=https://xxxx.us-east4-0.gcp.cloud.qdrant.io:6333
       QDRANT_API_KEY=your-api-key-here
"""

import logging
import hashlib
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)

from app.core.config import settings
from app.services.chunker import PolicyChunk

logger = logging.getLogger(__name__)


def _chunk_id_to_int(chunk_id: str) -> int:
    """
    Convert a string chunk_id (e.g. 'chunk_0_POL001') to a stable
    unsigned 64-bit integer for Qdrant point IDs.
    Uses the first 16 hex chars of MD5 — collision-safe at our scale.
    """
    return int(hashlib.md5(chunk_id.encode()).hexdigest()[:16], 16) % (2**63)


class VectorStoreService:
    """
    Qdrant Cloud-backed vector store.

    Maintains the same public interface as the original ChromaDB
    implementation so the rest of the codebase is unaffected.
    """

    def __init__(self):
        self.client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
        )
        self.collection_name = settings.COLLECTION_NAME
        self._ensure_collection()
        count = self.count()
        logger.info(
            f"VectorStore ready — collection '{self.collection_name}' "
            f"has {count} existing documents"
        )

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _ensure_collection(self) -> None:
        """Create the Qdrant collection if it doesn't already exist."""
        existing = {c.name for c in self.client.get_collections().collections}
        if self.collection_name not in existing:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=settings.EMBEDDING_DIMENSION,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(f"Created Qdrant collection '{self.collection_name}'")

    # ------------------------------------------------------------------ #
    # Public interface (mirrors original ChromaDB implementation)
    # ------------------------------------------------------------------ #

    def is_empty(self) -> bool:
        return self.count() == 0

    def count(self) -> int:
        result = self.client.count(collection_name=self.collection_name)
        return result.count

    def add_chunks(
        self,
        chunks: list[PolicyChunk],
        embeddings: list[list[float]],
    ) -> None:
        """Upsert policy chunks and their embeddings into Qdrant."""
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunks ({len(chunks)}) and embeddings ({len(embeddings)}) count mismatch"
            )

        points = [
            PointStruct(
                id=_chunk_id_to_int(chunk.chunk_id),
                vector=embedding,
                payload={
                    **chunk.metadata,
                    # Store document text in payload so we can return it in results
                    "_document": chunk.text,
                    "_chunk_id": chunk.chunk_id,
                },
            )
            for chunk, embedding in zip(chunks, embeddings)
        ]

        # Batch upserts to stay within Qdrant's request size limits
        batch_size = 100
        for i in range(0, len(points), batch_size):
            self.client.upsert(
                collection_name=self.collection_name,
                points=points[i : i + batch_size],
            )

        logger.info(
            f"Upserted {len(chunks)} chunks into Qdrant collection '{self.collection_name}'"
        )

    def query(
        self,
        query_embedding: list[float],
        top_k: int,
        where: Optional[dict] = None,
    ) -> dict:
        """
        Retrieve the top-k most similar chunks for a query embedding.

        Returns a dict mirroring ChromaDB's response shape so rag.py's
        _parse_results() works without any changes:
          {
            "documents": [[doc_text, ...]],
            "metadatas": [[metadata_dict, ...]],
            "distances": [[score, ...]],
          }

        Args:
            query_embedding: The embedded query vector
            top_k:           Number of results to return
            where:           Optional equality filter e.g. {"policy_type": "Motor"}
        """
        qdrant_filter = None
        if where:
            qdrant_filter = Filter(
                must=[
                    FieldCondition(key=field, match=MatchValue(value=value))
                    for field, value in where.items()
                ]
            )

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=min(top_k, max(self.count(), 1)),
            query_filter=qdrant_filter,
            with_payload=True,
        )

        documents: list[str] = []
        metadatas: list[dict] = []
        distances: list[float] = []

        for hit in results:
            payload = hit.payload or {}
            documents.append(payload.get("_document", ""))
            # Strip internal underscore-prefixed fields from metadata
            meta = {k: v for k, v in payload.items() if not k.startswith("_")}
            metadatas.append(meta)
            distances.append(hit.score)

        # Wrap in outer lists to match ChromaDB's nested-list response shape
        return {
            "documents": [documents],
            "metadatas": [metadatas],
            "distances": [distances],
        }

    def delete_collection(self) -> None:
        """Wipe and recreate the collection (used by force re-ingestion)."""
        self.client.delete_collection(self.collection_name)
        self._ensure_collection()
        logger.warning(f"Collection '{self.collection_name}' deleted and recreated.")
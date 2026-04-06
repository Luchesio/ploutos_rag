"""
vector_store.py — Qdrant Cloud vector store for Femi RAG Service.

Uses qdrant-client >= 1.8 API (query_points replaces the deprecated search).
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
    """
    return int(hashlib.md5(chunk_id.encode()).hexdigest()[:16], 16) % (2**63)


class VectorStoreService:
    """
    Qdrant Cloud-backed vector store.
    Maintains the same public interface as the original ChromaDB implementation.
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
                    "_document": chunk.text,
                    "_chunk_id": chunk.chunk_id,
                },
            )
            for chunk, embedding in zip(chunks, embeddings)
        ]

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
        """
        qdrant_filter = None
        if where:
            qdrant_filter = Filter(
                must=[
                    FieldCondition(key=field, match=MatchValue(value=value))
                    for field, value in where.items()
                ]
            )

        # query_points is the current API in qdrant-client >= 1.8
        # (replaces the deprecated .search() method)
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=min(top_k, max(self.count(), 1)),
            query_filter=qdrant_filter,
            with_payload=True,
        )

        documents: list[str] = []
        metadatas: list[dict] = []
        distances: list[float] = []

        for hit in response.points:
            payload = hit.payload or {}
            documents.append(payload.get("_document", ""))
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
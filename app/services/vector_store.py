"""
vector_store.py — Qdrant Cloud vector store for Femi RAG Service.

Retrieval strategy: score_threshold instead of TOP_K
─────────────────────────────────────────────────────
Rather than returning an arbitrary fixed number of chunks (TOP_K), we ask
Qdrant to return EVERY chunk whose cosine similarity score meets or exceeds
SIMILARITY_THRESHOLD. This means Femi sees all genuinely relevant policy
records for a query — not just the top 5 or top 20.

  score_threshold = 1 - SIMILARITY_THRESHOLD
  (because Qdrant score is the inverse of ChromaDB-style distance)

limit is set to self.count() (the full collection size) so no relevant
result is ever capped out.

The response shape returned is still ChromaDB-compatible (distances are
returned as 1 - score) so rag.py requires no changes.
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
    """Convert a string chunk_id to a stable unsigned 64-bit integer."""
    return int(hashlib.md5(chunk_id.encode()).hexdigest()[:16], 16) % (2**63)


class VectorStoreService:

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
    # Public interface
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
            f"Upserted {len(chunks)} chunks into Qdrant '{self.collection_name}'"
        )

    def query(
        self,
        query_embedding: list[float],
        top_k: int,           # kept for interface compatibility — not used as a hard cap
        where: Optional[dict] = None,
    ) -> dict:
        """
        Return ALL chunks that score above SIMILARITY_THRESHOLD.

        score_threshold = 1 - SIMILARITY_THRESHOLD converts the ChromaDB-style
        distance threshold into a Qdrant score threshold.

        limit is set to the full collection size so no valid result is capped.
        Returned distances are (1 - score) to stay ChromaDB-compatible.
        """
        total = max(self.count(), 1)

        # Qdrant score is the inverse of ChromaDB distance:
        #   distance = 1 - score  →  score_threshold = 1 - SIMILARITY_THRESHOLD
        score_threshold = 1.0 - settings.SIMILARITY_THRESHOLD

        qdrant_filter = None
        if where:
            qdrant_filter = Filter(
                must=[
                    FieldCondition(key=field, match=MatchValue(value=value))
                    for field, value in where.items()
                ]
            )

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=total,                    # never cap — return all above threshold
            score_threshold=score_threshold,
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
            distances.append(1.0 - hit.score)   # convert back to distance for rag.py

        logger.info(
            f"Query returned {len(documents)} chunks above "
            f"score_threshold={score_threshold:.2f} "
            f"(SIMILARITY_THRESHOLD={settings.SIMILARITY_THRESHOLD})"
        )

        return {
            "documents": [documents],
            "metadatas": [metadatas],
            "distances": [distances],
        }

    def delete_collection(self) -> None:
        self.client.delete_collection(self.collection_name)
        self._ensure_collection()
        logger.warning(f"Collection '{self.collection_name}' deleted and recreated.")
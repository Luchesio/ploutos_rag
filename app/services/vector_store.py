import chromadb
import logging
from typing import Optional

from app.core.config import settings
from app.services.chunker import PolicyChunk

logger = logging.getLogger(__name__)

# Module-level singleton — shared across all VectorStoreService instances
# within the same process/Lambda invocation. EphemeralClient is pure Python
# (no Rust bindings) and works correctly on Vercel and other serverless runtimes.
_chroma_client: Optional[chromadb.EphemeralClient] = None


def _get_chroma_client() -> chromadb.EphemeralClient:
    """
    Return a module-level singleton EphemeralClient.

    Singleton is important on Vercel: each Lambda invocation may be reused
    (warm start), so we avoid re-creating the in-memory client — and the
    collection — on every request. Without this, every request would see an
    empty collection because a brand-new EphemeralClient starts blank.
    """
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.EphemeralClient()
        logger.info("ChromaDB EphemeralClient created (pure-Python, serverless-safe)")
    return _chroma_client


class VectorStoreService:
    """
    Manages the ChromaDB vector store for storing and retrieving
    embedded policy chunks.

    Uses EphemeralClient (in-memory, pure Python) instead of PersistentClient
    so the service works on serverless runtimes (Vercel/Lambda) where:
      • The Rust-based persistent backend is unavailable
      • The filesystem is read-only outside /tmp anyway

    A module-level singleton client ensures the collection survives across
    warm invocations within the same Lambda container.

    For Docker / long-running deployments persistence is handled by the
    ingestion skip-check at startup (data is re-embedded only once per
    container lifetime on cold start, then reused for all warm requests).
    """

    def __init__(self):
        self.client = _get_chroma_client()
        self.collection = self.client.get_or_create_collection(
            name=settings.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},  # cosine similarity for Gemini embeddings
        )
        logger.info(
            f"VectorStore ready — collection '{settings.COLLECTION_NAME}' "
            f"has {self.collection.count()} existing documents"
        )

    def is_empty(self) -> bool:
        return self.collection.count() == 0

    def add_chunks(self, chunks: list[PolicyChunk], embeddings: list[list[float]]) -> None:
        """Upsert policy chunks with their embeddings into ChromaDB."""
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunks ({len(chunks)}) and embeddings ({len(embeddings)}) count mismatch"
            )

        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.text for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]

        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        logger.info(f"Upserted {len(chunks)} chunks into vector store")

    def query(
        self,
        query_embedding: list[float],
        top_k: int,
        where: Optional[dict] = None,
    ) -> dict:
        """
        Retrieve the top-k most similar chunks for a query embedding.

        Args:
            query_embedding: The embedded query vector
            top_k: Number of results to return
            where: Optional ChromaDB metadata filter (e.g. {"policy_type": "Motor"})
        """
        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": min(top_k, self.collection.count()),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = self.collection.query(**kwargs)
        return results

    def count(self) -> int:
        return self.collection.count()

    def delete_collection(self) -> None:
        """Reset the entire collection (for re-ingestion)."""
        self.client.delete_collection(settings.COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=settings.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.warning("Collection deleted and recreated.")
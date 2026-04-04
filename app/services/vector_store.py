import chromadb
from chromadb.config import Settings as ChromaSettings
import logging
from typing import Optional

from app.core.config import settings
from app.services.chunker import PolicyChunk

logger = logging.getLogger(__name__)


class VectorStoreService:
    """
    Manages the ChromaDB vector store for storing and retrieving
    embedded policy chunks.

    Uses persistent storage so the vector store survives restarts
    and doesn't require re-ingestion every time.
    """

    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=settings.CHROMA_PERSIST_DIR,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
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
            raise ValueError(f"Chunks ({len(chunks)}) and embeddings ({len(embeddings)}) count mismatch")

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
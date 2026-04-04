import logging
from app.core.config import settings
from app.services.chunker import DocumentSpecificSplitter
from app.services.embedder import EmbeddingService
from app.services.vector_store import VectorStoreService

logger = logging.getLogger(__name__)


class IngestionService:
    """
    Orchestrates the full ingestion pipeline:
      1. Load & split Excel into policy chunks (document-specific splitting)
      2. Embed each chunk using gemini-embedding-001
      3. Store embeddings + metadata in ChromaDB

    Ingestion is skipped if the vector store already has data,
    making startup fast after the first run.
    """

    def __init__(self):
        self.splitter = DocumentSpecificSplitter(data_path=str(settings.DATA_PATH))
        self.embedder = EmbeddingService()
        self.vector_store = VectorStoreService()

    async def ingest(self, force: bool = False) -> dict:
        """
        Run ingestion pipeline. Skips if data already exists unless force=True.

        Returns a summary dict with chunk count and collection name.
        """
        if not force and not self.vector_store.is_empty():
            count = self.vector_store.count()
            logger.info(f"Vector store already has {count} chunks — skipping ingestion")
            return {"message": "Already ingested", "total_chunks": count, "collection": settings.COLLECTION_NAME}

        if force:
            logger.info("Force re-ingestion — clearing existing collection")
            self.vector_store.delete_collection()

        # Step 1: Document-specific splitting
        logger.info("Step 1/3 — Splitting document into policy chunks...")
        chunks = self.splitter.load_and_split()
        logger.info(f"Produced {len(chunks)} policy chunks")

        # Step 2: Embed all chunks
        logger.info("Step 2/3 — Embedding chunks with gemini-embedding-001...")
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedder.embed_documents_batch(texts)

        # Step 3: Store in vector store
        logger.info("Step 3/3 — Storing in ChromaDB...")
        self.vector_store.add_chunks(chunks, embeddings)

        total = self.vector_store.count()
        logger.info(f"Ingestion complete — {total} chunks stored in '{settings.COLLECTION_NAME}'")
        return {
            "message": "Ingestion successful",
            "total_chunks": total,
            "collection": settings.COLLECTION_NAME,
        }
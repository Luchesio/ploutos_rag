from google import genai
from google.genai import types
import logging
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Handles text embedding using Google's gemini-embedding-001 model.

    Uses task_type='RETRIEVAL_DOCUMENT' for indexing and
    'RETRIEVAL_QUERY' at query time — this is crucial for
    asymmetric retrieval quality with Gemini embeddings.
    """

    def __init__(self):
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)
        self.model = settings.EMBEDDING_MODEL
        logger.info(f"EmbeddingService initialised with model: {self.model}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def embed_document(self, text: str) -> list[float]:
        """Embed a document chunk for indexing."""
        result = self.client.models.embed_content(
            model=self.model,
            contents=text,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
        )
        return result.embeddings[0].values

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def embed_query(self, query: str) -> list[float]:
        """Embed a user query for retrieval."""
        result = self.client.models.embed_content(
            model=self.model,
            contents=query,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
        )
        return result.embeddings[0].values

    def embed_documents_batch(self, texts: list[str], batch_size: int = 20) -> list[list[float]]:
        """
        Embed a list of documents in batches to respect API rate limits.
        Gemini embedding API has per-minute limits, so we batch carefully.
        """
        embeddings = []
        total = len(texts)

        for i in range(0, total, batch_size):
            batch = texts[i: i + batch_size]
            logger.info(
                f"Embedding batch {i // batch_size + 1}/"
                f"{(total + batch_size - 1) // batch_size} ({len(batch)} docs)"
            )
            for text in batch:
                emb = self.embed_document(text)
                embeddings.append(emb)

        logger.info(f"Embedded {len(embeddings)} documents total")
        return embeddings
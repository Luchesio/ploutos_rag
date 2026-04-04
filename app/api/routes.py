import time
import logging
from fastapi import APIRouter, Depends, Request

from app.models.schemas import QueryRequest, QueryResponse, IngestResponse
from app.services.rag import RAGService
from app.services.ingestion import IngestionService
from app.services.conversation_store import ConversationStore
from app.core.security import get_current_user, check_rate_limit
from app.core.sanitiser import sanitise_input
from app.core.exceptions import (
    FemiException, DatabaseError, IngestionError, InputValidationError
)
from app.core.metrics import metrics

router = APIRouter()
logger = logging.getLogger(__name__)

_rag_service: RAGService | None = None
_ingestion_service: IngestionService | None = None


def get_rag_service() -> RAGService:
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service


def get_ingestion_service() -> IngestionService:
    global _ingestion_service
    if _ingestion_service is None:
        _ingestion_service = IngestionService()
    return _ingestion_service


# ------------------------------------------------------------------ #
# Query
# ------------------------------------------------------------------ #

@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Ask Femi a question",
)
async def query_femi(
    request: Request,
    body: QueryRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Ask Femi a question — about insurance policies or anything general.

    Pass the same `session_id` across turns to maintain conversation context.
    Omit it to start a fresh session (a UUID is auto-generated).
    """
    # Rate limit per user email
    check_rate_limit(current_user["email"])

    # Sanitise input server-side (strips control chars, enforces max length)
    clean_question = sanitise_input(body.question)

    rag = get_rag_service()
    return await rag.query(
        question=clean_question,
        session_id=body.session_id,
    )


# ------------------------------------------------------------------ #
# Session management
# ------------------------------------------------------------------ #

@router.delete(
    "/session/{session_id}",
    summary="Clear a conversation session",
    dependencies=[Depends(get_current_user)],
)
async def clear_session(session_id: str):
    """
    Delete all conversation history for the given session_id.
    Useful for a 'New Chat' button on the frontend.
    """
    try:
        store = ConversationStore()
        await store.delete_session(session_id)
        return {"message": f"Session '{session_id}' cleared successfully."}
    except Exception as e:
        logger.exception(f"Session clear failed: {e}")
        raise DatabaseError(detail=str(e))


# ------------------------------------------------------------------ #
# Ingestion
# ------------------------------------------------------------------ #

@router.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Re-ingest document data",
    dependencies=[Depends(get_current_user)],
)
async def reingest(force: bool = False):
    """
    Trigger re-ingestion of the source Excel document.
    Set force=true to wipe and rebuild the vector store from scratch.
    """
    try:
        ingestion = get_ingestion_service()
        result = await ingestion.ingest(force=force)
        return IngestResponse(**result)
    except FemiException:
        raise
    except Exception as e:
        logger.exception(f"Ingestion failed: {e}")
        raise IngestionError(detail=str(e))


# ------------------------------------------------------------------ #
# Observability
# ------------------------------------------------------------------ #

@router.get("/stats", summary="System configuration stats")
async def get_stats(_: dict = Depends(get_current_user)):
    """Returns current vector store and config stats."""
    from app.services.vector_store import VectorStoreService
    from app.core.config import settings

    vs = VectorStoreService()
    return {
        "assistant":                 settings.BOT_NAME,
        "owner":                     settings.BOT_OWNER,
        "collection":                settings.COLLECTION_NAME,
        "total_chunks":              vs.count(),
        "embedding_model":           settings.EMBEDDING_MODEL,
        "llm_model":                 settings.LLM_MODEL,
        "top_k":                     settings.TOP_K,
        "conversation_history_turns": settings.CONVERSATION_HISTORY_TURNS,
        "embedding_cache_size":      settings.EMBEDDING_CACHE_SIZE,
    }


@router.get("/metrics", summary="Runtime metrics and observability")
async def get_metrics(_: dict = Depends(get_current_user)):
    """
    Returns live runtime metrics:
    request counts, error rates, intent distribution,
    latency percentiles, and embedding cache hit rate.
    """
    return metrics.snapshot()
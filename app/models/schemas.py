import uuid
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        description="User question to query the RAG system",
    )
    session_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description=(
            "Conversation session identifier. "
            "Pass the same ID across turns to maintain context. "
            "Omit to start a fresh session (auto-generated UUID)."
        ),
    )



class PolicyChunk(BaseModel):
    policy_id: str
    customer_name: str
    policy_type: str
    status: str
    content: str
    relevance_score: float


class QueryResponse(BaseModel):
    question: str
    answer: str
    session_id: str
    sources: list[PolicyChunk]
    total_chunks_searched: int


class IngestResponse(BaseModel):
    message: str
    total_chunks: int
    collection: str
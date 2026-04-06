from pydantic_settings import BaseSettings
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    # ------------------------------------------------------------------ #
    # Application — hardcoded
    # ------------------------------------------------------------------ #
    APP_NAME: str        = "Ploutos Page Limited – Insurance RAG System"
    BOT_NAME: str        = "Femi"
    BOT_OWNER: str       = "Ploutos Page Limited"

    # ------------------------------------------------------------------ #
    # Gemini — must be in .env / Vercel env vars
    # ------------------------------------------------------------------ #
    GEMINI_API_KEY: str
    LLM_MODEL: str       = "gemini-2.5-flash"

    # ------------------------------------------------------------------ #
    # Embedding — hardcoded
    # ------------------------------------------------------------------ #
    EMBEDDING_MODEL:     str = "models/gemini-embedding-001"
    EMBEDDING_DIMENSION: int = 3072

    # ------------------------------------------------------------------ #
    # Data
    # ------------------------------------------------------------------ #
    DATA_PATH: Path = Path("data/sample_data.xlsx")

    # ------------------------------------------------------------------ #
    # Qdrant Cloud — must be in .env / Vercel env vars
    # ------------------------------------------------------------------ #
    QDRANT_URL:      str
    QDRANT_API_KEY:  str
    COLLECTION_NAME: str = "insurance_policies"

    # ------------------------------------------------------------------ #
    # Retrieval
    # ------------------------------------------------------------------ #
    # TOP_K is retained so rag.py requires no changes — vector_store.query()
    # ignores it and uses score_threshold + full collection size instead.
    # Set to 75 (full dataset size) as a safe pass-through value.
    TOP_K: int = 75

    # SIMILARITY_THRESHOLD drives the actual retrieval cutoff.
    # Qdrant score_threshold = 1 - SIMILARITY_THRESHOLD.
    # 0.5 → accepts any chunk where Qdrant scores above 0.5.
    # Lower to be more strict; raise to be more permissive.
    SIMILARITY_THRESHOLD: float = 0.67

    # ------------------------------------------------------------------ #
    # MongoDB — must be in .env / Vercel env vars
    # ------------------------------------------------------------------ #
    MONGODB_URI:        str
    MONGODB_DB_NAME:    str = "femi_rag"
    MONGODB_COLLECTION: str = "conversations"

    # ------------------------------------------------------------------ #
    # MongoDB users collection — hardcoded
    # ------------------------------------------------------------------ #
    MONGODB_USERS_COLLECTION: str = "users"

    # ------------------------------------------------------------------ #
    # JWT — secret must be in .env, algorithm and expiry are hardcoded
    # ------------------------------------------------------------------ #
    JWT_SECRET_KEY:            str = ""
    JWT_ALGORITHM:             str = "HS256"
    ACCESS_TOKEN_EXPIRE_HOURS: int = 24

    # ------------------------------------------------------------------ #
    # Conversation history — hardcoded
    # ------------------------------------------------------------------ #
    CONVERSATION_HISTORY_TURNS: int = 5

    # ------------------------------------------------------------------ #
    # Security
    # ------------------------------------------------------------------ #
    API_KEY:                str | None = None
    RATE_LIMIT_PER_MINUTE:  int        = 60
    MAX_QUESTION_LENGTH:    int        = 2000

    # ------------------------------------------------------------------ #
    # Performance
    # ------------------------------------------------------------------ #
    EMBEDDING_CACHE_SIZE: int = 512

    # ------------------------------------------------------------------ #
    # Observability
    # ------------------------------------------------------------------ #
    LOG_LEVEL:  str = "INFO"
    LOG_FORMAT: str = "human"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
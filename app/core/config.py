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
    # Gemini — must be in .env
    # ------------------------------------------------------------------ #
    GEMINI_API_KEY: str
    LLM_MODEL: str       = "gemini-2.5-flash"

    # ------------------------------------------------------------------ #
    # Embedding — hardcoded
    # ------------------------------------------------------------------ #
    EMBEDDING_MODEL:     str = "models/gemini-embedding-001"
    EMBEDDING_DIMENSION: int = 3072

    # ------------------------------------------------------------------ #
    # Data & persistence — hardcoded
    # ------------------------------------------------------------------ #
    DATA_PATH:          Path = Path("data/sample_data.xlsx")
    CHROMA_PERSIST_DIR: str  = "./chroma_db"
    COLLECTION_NAME:    str  = "insurance_policies"

    # ------------------------------------------------------------------ #
    # Retrieval — hardcoded
    # ------------------------------------------------------------------ #
    TOP_K:                int   = 5
    SIMILARITY_THRESHOLD: float = 0.3

    # ------------------------------------------------------------------ #
    # MongoDB — must be in .env
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
    JWT_SECRET_KEY:            str = ""          # required — must be set in .env
    JWT_ALGORITHM:             str = "HS256"     # hardcoded
    ACCESS_TOKEN_EXPIRE_HOURS: int = 24          # hardcoded

    # ------------------------------------------------------------------ #
    # Conversation history — hardcoded
    # ------------------------------------------------------------------ #
    CONVERSATION_HISTORY_TURNS: int = 5

    # ------------------------------------------------------------------ #
    # Security — API_KEY must be set in .env for production
    # ------------------------------------------------------------------ #
    # Leave API_KEY empty / unset to disable authentication (dev only).
    API_KEY:                str | None = None
    RATE_LIMIT_PER_MINUTE:  int        = 60
    MAX_QUESTION_LENGTH:    int        = 2000

    # ------------------------------------------------------------------ #
    # Performance
    # ------------------------------------------------------------------ #
    EMBEDDING_CACHE_SIZE: int = 512   # LRU slots for query embeddings

    # ------------------------------------------------------------------ #
    # Observability
    # ------------------------------------------------------------------ #
    LOG_LEVEL:  str = "INFO"   # DEBUG | INFO | WARNING | ERROR
    LOG_FORMAT: str = "human"  # "human" for dev, "json" for prod

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
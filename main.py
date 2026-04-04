from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from contextlib import asynccontextmanager
import logging

from app.api.routes import router
from app.api.auth_routes import auth_router
from app.core.config import settings
from app.core.logging_config import setup_logging, request_id_filter
from app.core.exceptions import (
    FemiException,
    femi_exception_handler,
    validation_exception_handler,
    generic_exception_handler,
)
from app.middleware.request_context import RequestContextMiddleware
from app.services.ingestion import IngestionService

# Initialise structured logging before anything else
setup_logging()

# Attach the request-ID filter to the root logger so every log record
# emitted during a request carries its ID automatically
logging.getLogger().addFilter(request_id_filter)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Lifespan
# ------------------------------------------------------------------ #

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting {settings.BOT_NAME} ({settings.APP_NAME})")
    try:
        # Ensure MongoDB user indexes exist
        from app.services.auth_service import AuthService
        await AuthService().ensure_indexes()
        # Ingest policy documents
        ingestion = IngestionService()
        await ingestion.ingest()
        logger.info(f"{settings.BOT_NAME} is ready.")
    except Exception as e:
        logger.error(f"Startup ingestion failed: {e}", exc_info=True)
        # Don't abort startup — vector store may already be populated
    yield
    logger.info(f"Shutting down {settings.BOT_NAME}.")


# ------------------------------------------------------------------ #
# App
# ------------------------------------------------------------------ #

app = FastAPI(
    title=settings.APP_NAME,
    description=(
        f"{settings.BOT_NAME} is an AI assistant owned by {settings.BOT_OWNER}. "
        "Powered by Gemini and ChromaDB."
    ),
    version="1.0.0",
    lifespan=lifespan,
    # Disable default 422 detail leak — our handler replaces it
    docs_url="/docs",
    redoc_url="/redoc",
)


# ------------------------------------------------------------------ #
# Middleware  (order matters — outermost registered last)
# ------------------------------------------------------------------ #

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request ID + timing + access logging
app.add_middleware(RequestContextMiddleware)


# ------------------------------------------------------------------ #
# Exception handlers
# ------------------------------------------------------------------ #

app.add_exception_handler(FemiException, femi_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, generic_exception_handler)


# ------------------------------------------------------------------ #
# Routes
# ------------------------------------------------------------------ #

app.include_router(auth_router, prefix="/auth")
app.include_router(router, prefix="/api/v1")


# ------------------------------------------------------------------ #
# Health check  (public — no auth required)
# ------------------------------------------------------------------ #

@app.get("/health", tags=["Observability"])
async def health_check():
    """
    Lightweight liveness probe.
    Returns 200 as long as the application process is running.
    """
    from app.core.metrics import metrics as m
    snap = m.snapshot()
    return {
        "status":    "ok",
        "app":       settings.APP_NAME,
        "assistant": settings.BOT_NAME,
        "uptime_s":  snap["uptime_seconds"],
        "requests":  snap["requests"]["total"],
    }


# ------------------------------------------------------------------ #
# Custom OpenAPI schema — clean HTTPBearer in Swagger UI
# ------------------------------------------------------------------ #

def custom_openapi():
    """
    Override the OpenAPI schema to register HTTPBearer as the single
    security scheme. This gives Swagger a clean 'Authorize' dialog with
    just one field: Value (paste your token — no 'Bearer' prefix needed).
    """
    if app.openapi_schema:
        return app.openapi_schema

    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

    schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type":         "http",
            "scheme":       "bearer",
            "bearerFormat": "JWT",
            "description":  (
                "Paste your JWT access token here. "
                "Obtain it from POST /auth/signin. "
                "Do NOT include the word 'Bearer' — just paste the token."
            ),
        }
    }

    # Apply the scheme globally to all protected operations
    for path in schema.get("paths", {}).values():
        for operation in path.values():
            operation["security"] = [{"BearerAuth": []}]

    app.openapi_schema = schema
    return app.openapi_schema


app.openapi = custom_openapi
"""
exceptions.py — Custom exception hierarchy for Femi RAG Service.

All domain errors map to a specific HTTP status code and a safe, structured
error response. Internal details (stack traces, LLM errors) are logged server-
side but never leaked to the client.
"""

from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import logging

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Base
# ------------------------------------------------------------------ #

class FemiException(Exception):
    """Base class for all application exceptions."""
    status_code: int = 500
    error_code: str = "INTERNAL_ERROR"
    message: str = "An unexpected error occurred."

    def __init__(self, message: str | None = None, detail: str | None = None):
        self.message = message or self.__class__.message
        self.detail = detail  # internal detail — logged, not exposed
        super().__init__(self.message)


# ------------------------------------------------------------------ #
# Authentication & authorisation
# ------------------------------------------------------------------ #

class AuthenticationError(FemiException):
    status_code = 401
    error_code = "AUTHENTICATION_FAILED"
    message = "Authentication failed. Please sign in to continue."


class RateLimitError(FemiException):
    status_code = 429
    error_code = "RATE_LIMIT_EXCEEDED"
    message = "Too many requests. Please slow down."


class UserAlreadyExistsError(FemiException):
    status_code = 409
    error_code = "USER_ALREADY_EXISTS"
    message = "An account with this email or username already exists."


class UserNotFoundError(FemiException):
    status_code = 404
    error_code = "USER_NOT_FOUND"
    message = "User account not found."


# ------------------------------------------------------------------ #
# Input validation
# ------------------------------------------------------------------ #

class InputValidationError(FemiException):
    status_code = 422
    error_code = "INPUT_VALIDATION_FAILED"
    message = "The request contains invalid input."


# ------------------------------------------------------------------ #
# RAG pipeline
# ------------------------------------------------------------------ #

class EmbeddingError(FemiException):
    status_code = 502
    error_code = "EMBEDDING_FAILED"
    message = "Failed to generate embedding for the query."


class LLMError(FemiException):
    status_code = 502
    error_code = "LLM_FAILED"
    message = "The language model failed to generate a response."


class RetrievalError(FemiException):
    status_code = 502
    error_code = "RETRIEVAL_FAILED"
    message = "Failed to retrieve documents from the vector store."


class IngestionError(FemiException):
    status_code = 500
    error_code = "INGESTION_FAILED"
    message = "Document ingestion failed."


# ------------------------------------------------------------------ #
# Database
# ------------------------------------------------------------------ #

class DatabaseError(FemiException):
    status_code = 503
    error_code = "DATABASE_ERROR"
    message = "A database error occurred. Please try again shortly."


# ------------------------------------------------------------------ #
# FastAPI exception handlers
# ------------------------------------------------------------------ #

def _error_body(error_code: str, message: str, request_id: str | None = None) -> dict:
    body = {"error": error_code, "message": message}
    if request_id:
        body["request_id"] = request_id
    return body


async def femi_exception_handler(request: Request, exc: FemiException) -> JSONResponse:
    """Handle all FemiException subclasses with structured JSON responses."""
    request_id = getattr(request.state, "request_id", None)

    # Log at appropriate level — 5xx as ERROR, 4xx as WARNING
    if exc.status_code >= 500:
        logger.error(
            f"[{request_id}] {exc.error_code}: {exc.message} | detail={exc.detail}",
            exc_info=True,
        )
    else:
        logger.warning(
            f"[{request_id}] {exc.error_code}: {exc.message}"
        )

    return JSONResponse(
        status_code=exc.status_code,
        content=_error_body(exc.error_code, exc.message, request_id),
    )


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Handle Pydantic validation errors with clean messages."""
    request_id = getattr(request.state, "request_id", None)
    errors = exc.errors()
    logger.warning(f"[{request_id}] Validation error: {errors}")
    return JSONResponse(
        status_code=422,
        content={
            "error": "VALIDATION_ERROR",
            "message": "Request validation failed.",
            "fields": [
                {
                    "field": " → ".join(str(loc) for loc in e["loc"]),
                    "issue": e["msg"],
                }
                for e in errors
            ],
            **({"request_id": request_id} if request_id else {}),
        },
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all handler for unhandled exceptions — never leaks internals."""
    request_id = getattr(request.state, "request_id", None)
    logger.exception(f"[{request_id}] Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=_error_body(
            "INTERNAL_ERROR",
            "An unexpected error occurred. Please try again.",
            request_id,
        ),
    )
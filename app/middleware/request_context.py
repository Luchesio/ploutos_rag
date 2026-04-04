"""
request_context.py — ASGI middleware for request lifecycle observability.

Responsibilities:
  1. Generate a unique request_id (UUID4) for every incoming request and
     attach it to request.state so handlers and loggers can reference it.
  2. Inject the request_id into all log records emitted during that request
     via a logging.Filter.
  3. Log a structured START and END record for every request with method,
     path, status code, and wall-clock duration.
  4. Record timing and status into the metrics collector.
  5. Echo the request_id back to the client as X-Request-ID response header
     so it can be correlated with server logs.
"""

import time
import uuid
import logging
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.core.metrics import metrics

logger = logging.getLogger(__name__)


class RequestIDFilter(logging.Filter):
    """
    Logging filter that injects request_id into every LogRecord emitted
    while a request is being processed.

    Because ASGI is async, multiple requests may be in-flight concurrently.
    We store the current request_id in a ContextVar so each async task sees
    only its own ID.
    """

    def __init__(self):
        super().__init__()
        self._current_id: str | None = None

    def set(self, request_id: str | None) -> None:
        self._current_id = request_id

    def filter(self, record: logging.LogRecord) -> bool:
        if self._current_id and not hasattr(record, "request_id"):
            record.request_id = self._current_id
        return True


# Module-level filter instance attached to the root logger in setup_logging()
request_id_filter = RequestIDFilter()


class RequestContextMiddleware(BaseHTTPMiddleware):
    """
    Starlette middleware that wraps every request with:
      - A unique request_id
      - Structured request/response logging
      - Duration measurement fed into metrics
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        request_id_filter.set(request_id)

        start = time.perf_counter()

        # Determine a clean endpoint label for metrics (strip path params)
        endpoint = self._endpoint_label(request)

        logger.info(
            f"→ {request.method} {request.url.path}",
            extra={"request_id": request_id},
        )

        try:
            response = await call_next(request)
        except Exception as exc:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.error(
                f"✗ {request.method} {request.url.path} — unhandled exception "
                f"after {duration_ms:.1f}ms",
                extra={"request_id": request_id},
                exc_info=True,
            )
            metrics.record_request(endpoint, 500, duration_ms)
            request_id_filter.set(None)
            raise

        duration_ms = (time.perf_counter() - start) * 1000
        status = response.status_code

        logger.info(
            f"← {request.method} {request.url.path} "
            f"[{status}] {duration_ms:.1f}ms",
            extra={"request_id": request_id},
        )

        metrics.record_request(endpoint, status, duration_ms)
        request_id_filter.set(None)

        # Echo the request_id back to the caller
        response.headers["X-Request-ID"] = request_id
        return response

    @staticmethod
    def _endpoint_label(request: Request) -> str:
        """
        Return a stable label for the endpoint, replacing path parameter
        values with their names so metrics aren't exploded by unique IDs.
        e.g. /api/v1/session/abc-123  →  /api/v1/session/{session_id}
        """
        if request.scope.get("route"):
            return request.scope["route"].path
        return request.url.path
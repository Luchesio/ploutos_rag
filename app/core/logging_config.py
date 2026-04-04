"""
logging_config.py — Structured JSON logging for Femi RAG Service.

Produces machine-readable JSON log lines in production and human-readable
coloured output in development, controlled by LOG_FORMAT in config.

Every log record carries:
  timestamp, level, logger, message, request_id (if in context)
"""

import logging
import json
import sys
from datetime import datetime, timezone
from app.core.config import settings


class JSONFormatter(logging.Formatter):
    """
    Formats log records as single-line JSON objects.
    Includes request_id from the record's extra dict when present.
    """

    LEVEL_MAP = {
        logging.DEBUG: "DEBUG",
        logging.INFO: "INFO",
        logging.WARNING: "WARNING",
        logging.ERROR: "ERROR",
        logging.CRITICAL: "CRITICAL",
    }

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": self.LEVEL_MAP.get(record.levelno, record.levelname),
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Attach request_id if it was injected via LoggerAdapter or extra={}
        if hasattr(record, "request_id"):
            log_entry["request_id"] = record.request_id

        # Attach session_id if present
        if hasattr(record, "session_id"):
            log_entry["session_id"] = record.session_id

        # Attach exception info
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, ensure_ascii=False)


class HumanFormatter(logging.Formatter):
    """
    Coloured human-readable formatter for development.
    Format: [LEVEL]  timestamp  logger  message
    """

    COLOURS = {
        "DEBUG":    "\033[36m",   # cyan
        "INFO":     "\033[32m",   # green
        "WARNING":  "\033[33m",   # yellow
        "ERROR":    "\033[31m",   # red
        "CRITICAL": "\033[35m",   # magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        level = record.levelname
        colour = self.COLOURS.get(level, "")
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        rid = f" [{record.request_id}]" if hasattr(record, "request_id") else ""
        return (
            f"{colour}[{level:<8}]{self.RESET} "
            f"{ts}{rid}  "
            f"\033[90m{record.name}\033[0m  "
            f"{record.getMessage()}"
            + (f"\n{self.formatException(record.exc_info)}" if record.exc_info else "")
        )


class RequestIdFilter(logging.Filter):
    """
    Logging filter that injects a request_id field into every log record.
    If no request_id is set on the record already, defaults to "-".
    Attach to the root logger so all handlers benefit automatically.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "request_id"):
            record.request_id = "-"
        return True


# Singleton filter instance — attached to root logger in main.py via:
#   logging.getLogger().addFilter(request_id_filter)
request_id_filter = RequestIdFilter()


def setup_logging() -> None:
    """
    Configure root logger and silence noisy third-party loggers.
    Call once at application startup.
    """
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    use_json = settings.LOG_FORMAT.lower() == "json"

    formatter = JSONFormatter() if use_json else HumanFormatter()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(log_level)

    # Remove existing handlers (e.g. basicConfig default)
    root.handlers.clear()
    root.addHandler(handler)

    # Silence excessively chatty third-party loggers
    for noisy in (
        "httpx",
        "httpcore",
        "chromadb",
        "google_genai",
        "urllib3",
        "multipart",
        "watchfiles",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.getLogger(__name__).info(
        f"Logging initialised — level={settings.LOG_LEVEL} format={'JSON' if use_json else 'HUMAN'}"
    )
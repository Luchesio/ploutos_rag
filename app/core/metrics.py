"""
metrics.py — Lightweight in-memory observability for Femi RAG Service.

Tracks:
  • Request counts (total, by endpoint, by status code)
  • Error counts (by error code)
  • Intent distribution (DOCUMENT / GENERAL / CLARIFY)
  • Query latency (rolling last-1000 samples, per-endpoint)
  • Embedding cache hit/miss rate

All state is in-process. For multi-process or persistent metrics,
replace the in-memory store with a Prometheus push gateway or
an OTLP exporter — the interface stays the same.

Usage:
    from app.core.metrics import metrics
    metrics.record_request("query", 200, duration_ms=142.3)
    metrics.record_intent("DOCUMENT")
"""

import time
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque


# ------------------------------------------------------------------ #
# Data structures
# ------------------------------------------------------------------ #

@dataclass
class LatencyStats:
    """Rolling latency statistics over the last N samples."""
    samples: Deque[float] = field(default_factory=lambda: deque(maxlen=1000))

    def record(self, ms: float) -> None:
        self.samples.append(ms)

    def to_dict(self) -> dict:
        if not self.samples:
            return {"count": 0, "mean_ms": 0, "min_ms": 0, "max_ms": 0, "p95_ms": 0}
        sorted_s = sorted(self.samples)
        n = len(sorted_s)
        p95_idx = int(n * 0.95)
        return {
            "count": n,
            "mean_ms": round(sum(sorted_s) / n, 2),
            "min_ms":  round(sorted_s[0], 2),
            "max_ms":  round(sorted_s[-1], 2),
            "p95_ms":  round(sorted_s[min(p95_idx, n - 1)], 2),
        }


class MetricsCollector:
    """Thread-safe in-memory metrics store."""

    def __init__(self):
        self._lock = threading.Lock()
        self._start_time = time.time()

        # Counters
        self._requests_total:  int = 0
        self._errors_total:    int = 0
        self._by_endpoint:     dict[str, int]       = defaultdict(int)
        self._by_status:       dict[int, int]        = defaultdict(int)
        self._by_error_code:   dict[str, int]        = defaultdict(int)
        self._intent_counts:   dict[str, int]        = defaultdict(int)

        # Cache stats
        self._cache_hits:   int = 0
        self._cache_misses: int = 0

        # Latency
        self._latency_overall:  LatencyStats                  = LatencyStats()
        self._latency_endpoint: dict[str, LatencyStats]       = defaultdict(LatencyStats)

    # ------------------------------------------------------------------ #
    # Recording methods
    # ------------------------------------------------------------------ #

    def record_request(
        self,
        endpoint: str,
        status_code: int,
        duration_ms: float,
    ) -> None:
        with self._lock:
            self._requests_total += 1
            self._by_endpoint[endpoint] += 1
            self._by_status[status_code] += 1
            self._latency_overall.record(duration_ms)
            self._latency_endpoint[endpoint].record(duration_ms)
            if status_code >= 500:
                self._errors_total += 1

    def record_error(self, error_code: str) -> None:
        with self._lock:
            self._by_error_code[error_code] += 1

    def record_intent(self, intent: str) -> None:
        """Track how often each intent (DOCUMENT/GENERAL/CLARIFY) is classified."""
        with self._lock:
            self._intent_counts[intent.upper()] += 1

    def record_cache_hit(self) -> None:
        with self._lock:
            self._cache_hits += 1

    def record_cache_miss(self) -> None:
        with self._lock:
            self._cache_misses += 1

    # ------------------------------------------------------------------ #
    # Snapshot
    # ------------------------------------------------------------------ #

    def snapshot(self) -> dict:
        """Return a JSON-serialisable snapshot of all current metrics."""
        with self._lock:
            uptime_s = round(time.time() - self._start_time, 1)

            total_cache = self._cache_hits + self._cache_misses
            cache_hit_rate = (
                round(self._cache_hits / total_cache * 100, 1)
                if total_cache > 0 else 0
            )

            return {
                "uptime_seconds": uptime_s,
                "requests": {
                    "total":       self._requests_total,
                    "errors_5xx":  self._errors_total,
                    "by_endpoint": dict(self._by_endpoint),
                    "by_status":   {str(k): v for k, v in self._by_status.items()},
                },
                "errors": {
                    "by_code": dict(self._by_error_code),
                },
                "intents": dict(self._intent_counts),
                "latency": {
                    "overall":     self._latency_overall.to_dict(),
                    "by_endpoint": {
                        ep: stats.to_dict()
                        for ep, stats in self._latency_endpoint.items()
                    },
                },
                "embedding_cache": {
                    "hits":      self._cache_hits,
                    "misses":    self._cache_misses,
                    "hit_rate_%": cache_hit_rate,
                },
            }


# ------------------------------------------------------------------ #
# Singleton
# ------------------------------------------------------------------ #

metrics = MetricsCollector()
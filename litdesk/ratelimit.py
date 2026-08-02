"""Minimal sleep-based rate limiter. Single-threaded, sequential callers only."""

from __future__ import annotations

import time


class RateLimiter:
    def __init__(self, requests_per_sec: float):
        self.min_interval = 1.0 / requests_per_sec if requests_per_sec > 0 else 0.0
        self._last_call: float | None = None

    def wait(self) -> None:
        if self.min_interval <= 0 or self._last_call is None:
            self._last_call = time.monotonic()
            return
        elapsed = time.monotonic() - self._last_call
        remaining = self.min_interval - elapsed
        if remaining > 0:
            time.sleep(remaining)
        self._last_call = time.monotonic()

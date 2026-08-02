"""Shared GET-with-cache-and-backoff helper. Each source client instantiates
its own `CachedSession` (own cache subdirectory, own rate limit) so a schema
change in one API stays a contained edit, per spec section 2."""

from __future__ import annotations

import logging
import time

import requests

from litdesk.cache import FileCache
from litdesk.ratelimit import RateLimiter

logger = logging.getLogger("litdesk.http")


class HTTPError(Exception):
    pass


class CachedSession:
    def __init__(
        self,
        cache_dir: str,
        requests_per_sec: float,
        ttl_hours: float = 24.0,
        max_retries: int = 3,
        timeout: float = 30.0,
        user_agent: str = "litdesk/0.1 (personal research tool)",
    ):
        self.cache = FileCache(cache_dir, ttl_hours=ttl_hours)
        self.limiter = RateLimiter(requests_per_sec)
        self.max_retries = max_retries
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})

    def get_json(self, url: str, params: dict | None = None, use_cache: bool = True) -> dict:
        if use_cache:
            cached = self.cache.get(url, params)
            if cached is not None:
                return cached

        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            self.limiter.wait()
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
                if resp.status_code >= 500 or resp.status_code == 429:
                    raise HTTPError(f"{resp.status_code} from {url}")
                resp.raise_for_status()
                body = resp.json()
            except (requests.RequestException, HTTPError, ValueError) as exc:
                last_exc = exc
                if attempt < self.max_retries - 1:
                    backoff = 2**attempt
                    logger.warning("GET %s failed (attempt %d/%d): %s — retrying in %ss",
                                    url, attempt + 1, self.max_retries, exc, backoff)
                    time.sleep(backoff)
                continue
            if use_cache:
                self.cache.set(url, params, body)
            return body

        raise HTTPError(f"Giving up on {url} after {self.max_retries} attempts") from last_exc

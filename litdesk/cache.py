"""On-disk HTTP response cache, keyed by URL + params, TTL-based.

Spec gotcha (section 7): "Never call these APIs from inside a loop over
papers without a rate limiter and an on-disk cache. Cache by URL, TTL 24h.
This makes re-runs during development free." This also makes the pipeline
runnable offline against cached data (acceptance criterion, section 8).
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any


class FileCache:
    def __init__(self, cache_dir: str | Path, ttl_hours: float = 24.0):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_hours * 3600

    def _key(self, url: str, params: dict | None) -> str:
        payload = json.dumps({"url": url, "params": params or {}}, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _path(self, url: str, params: dict | None) -> Path:
        return self.cache_dir / f"{self._key(url, params)}.json"

    def get(self, url: str, params: dict | None = None) -> Any | None:
        path = self._path(url, params)
        if not path.exists():
            return None
        try:
            with open(path) as fh:
                envelope = json.load(fh)
        except (json.JSONDecodeError, OSError):
            return None
        age = time.time() - envelope["cached_at"]
        if age > self.ttl_seconds:
            return None
        return envelope["body"]

    def set(self, url: str, params: dict | None, body: Any) -> None:
        path = self._path(url, params)
        envelope = {
            "cached_at": time.time(),
            "url": url,
            "params": params or {},
            "body": body,
        }
        tmp_path = path.with_suffix(".tmp")
        with open(tmp_path, "w") as fh:
            json.dump(envelope, fh)
        tmp_path.replace(path)

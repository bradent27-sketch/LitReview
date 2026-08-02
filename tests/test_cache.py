import time

from litdesk.cache import FileCache


def test_set_then_get_roundtrip(tmp_path):
    cache = FileCache(tmp_path, ttl_hours=24)
    assert cache.get("https://example.com/x", {"a": 1}) is None
    cache.set("https://example.com/x", {"a": 1}, {"hello": "world"})
    assert cache.get("https://example.com/x", {"a": 1}) == {"hello": "world"}


def test_different_params_are_different_keys(tmp_path):
    cache = FileCache(tmp_path, ttl_hours=24)
    cache.set("https://example.com/x", {"a": 1}, {"v": 1})
    cache.set("https://example.com/x", {"a": 2}, {"v": 2})
    assert cache.get("https://example.com/x", {"a": 1}) == {"v": 1}
    assert cache.get("https://example.com/x", {"a": 2}) == {"v": 2}


def test_expired_entry_is_not_returned(tmp_path):
    cache = FileCache(tmp_path, ttl_hours=24)
    cache.set("https://example.com/x", None, {"v": 1})
    # Simulate an entry written 25 hours ago by rewriting cached_at directly.
    path = cache._path("https://example.com/x", None)
    import json

    with open(path) as fh:
        envelope = json.load(fh)
    envelope["cached_at"] = time.time() - 25 * 3600
    with open(path, "w") as fh:
        json.dump(envelope, fh)

    assert cache.get("https://example.com/x", None) is None

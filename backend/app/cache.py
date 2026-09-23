"""Two-tier exact-match prediction cache (Phase 3, system level).

The UI pre-fills every field with the training-set defaults and most users
only tweak one or two of them, so identical requests are common. A prediction
is a pure function of (model backend, model version, canonical input), which
makes exact-match caching safe:

    key = "hp:v<SERVICE_VERSION>:<backend>:" + sha256(sorted-key JSON of the input)

Tiers
-----
L1  in-process LRU (``CACHE_L1_SIZE`` entries, per worker). A hit costs a
    dict lookup - no syscalls, no network.
L2  Redis (``REDIS_URL``), shared by every worker and container and surviving
    restarts/deploys. L2 hits are promoted into L1.

Why two tiers: after the Week 5 optimisations one inference costs ~12 us, but a
Redis round trip costs ~0.3 ms plus CPU on both ends. Benchmarks showed a
Redis-only cache made hot keys *slower* than recomputing them at high
concurrency; L1 serves hot keys for free while Redis still provides a shared,
warm cache for new workers. Writes to Redis happen in the background so a
cache miss never waits on the SET.

Redis is an *optimisation*, never a dependency: if it is unset or unreachable
the service keeps answering. After a failure L2 is bypassed for
``_RETRY_AFTER`` seconds (a tiny circuit breaker) so a dead Redis does not add
a timeout to every request.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass
from typing import Any

from . import config

log = logging.getLogger("house_price.cache")

_RETRY_AFTER = 2.0  # seconds
# Generous enough not to trip when the host CPU is saturated (event-loop lag
# shows up as socket latency), short enough that a dead Redis is noticed fast.
_TIMEOUT = 0.25  # seconds


@dataclass
class CacheStats:
    l1_enabled: bool = False
    l2_enabled: bool = False
    l1_hits: int = 0
    l2_hits: int = 0
    misses: int = 0
    errors: int = 0

    @property
    def hit_rate(self) -> float:
        hits = self.l1_hits + self.l2_hits
        total = hits + self.misses
        return hits / total if total else 0.0


class LRU:
    """Tiny TTL-aware LRU; single-threaded use from the event loop."""

    def __init__(self, capacity: int, ttl: float) -> None:
        self.capacity, self.ttl = capacity, ttl
        self._data: OrderedDict[str, tuple[float, dict]] = OrderedDict()

    def get(self, key: str) -> dict | None:
        item = self._data.get(key)
        if item is None:
            return None
        expires, value = item
        if expires < time.monotonic():
            del self._data[key]
            return None
        self._data.move_to_end(key)
        return value

    def put(self, key: str, value: dict) -> None:
        self._data[key] = (time.monotonic() + self.ttl, value)
        self._data.move_to_end(key)
        while len(self._data) > self.capacity:
            self._data.popitem(last=False)

    def __len__(self) -> int:
        return len(self._data)


class PredictionCache:
    def __init__(self, url: str, ttl: int, l1_size: int) -> None:
        self.url = url
        self.ttl = ttl
        self.l1 = LRU(l1_size, ttl) if l1_size > 0 else None
        self.stats = CacheStats(l1_enabled=self.l1 is not None, l2_enabled=bool(url))
        self._client = None
        self._down_until = 0.0
        self._pending: set[asyncio.Task] = set()

    @property
    def enabled(self) -> bool:
        return self.l1 is not None or self._client is not None

    async def connect(self) -> None:
        if not self.url:
            return
        import redis.asyncio as redis

        self._client = redis.from_url(
            self.url,
            socket_timeout=_TIMEOUT,
            socket_connect_timeout=_TIMEOUT * 4,
            decode_responses=True,
        )
        try:
            await self._client.ping()
            log.info("prediction cache connected to %s", self.url)
        except Exception as exc:  # pragma: no cover - depends on environment
            self._trip(exc)

    async def close(self) -> None:
        if self._pending:
            await asyncio.gather(*self._pending, return_exceptions=True)
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    @staticmethod
    def key(payload: dict[str, Any], backend: str) -> str:
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        digest = hashlib.sha256(canonical.encode()).hexdigest()
        return f"hp:v{config.SERVICE_VERSION}:{backend}:{digest}"

    def _l2_available(self) -> bool:
        return self._client is not None and time.monotonic() >= self._down_until

    def _trip(self, exc: Exception) -> None:
        self.stats.errors += 1
        self._down_until = time.monotonic() + _RETRY_AFTER
        log.warning("redis cache unavailable (%s); bypassing for %.0fs", exc, _RETRY_AFTER)

    async def get(self, key: str) -> dict[str, float] | None:
        if self.l1 is not None:
            value = self.l1.get(key)
            if value is not None:
                self.stats.l1_hits += 1
                return value
        if self._l2_available():
            try:
                raw = await self._client.get(key)
            except Exception as exc:
                self._trip(exc)
                raw = None
            if raw is not None:
                value = json.loads(raw)
                self.stats.l2_hits += 1
                if self.l1 is not None:
                    self.l1.put(key, value)
                return value
        if self.enabled:
            self.stats.misses += 1
        return None

    async def _l2_set(self, key: str, value: dict[str, float]) -> None:
        try:
            await self._client.set(key, json.dumps(value), ex=self.ttl)
        except Exception as exc:
            self._trip(exc)

    async def set(self, key: str, value: dict[str, float]) -> None:
        if self.l1 is not None:
            self.l1.put(key, value)
        if self._l2_available():
            # Fire-and-forget: the response does not wait for Redis.
            task = asyncio.create_task(self._l2_set(key, value))
            self._pending.add(task)
            task.add_done_callback(self._pending.discard)

    def snapshot(self) -> dict[str, Any]:
        data = asdict(self.stats)
        data["hit_rate"] = round(self.stats.hit_rate, 4)
        data["l1_entries"] = len(self.l1) if self.l1 is not None else 0
        data["l2_connected"] = self._l2_available()
        return data


cache = PredictionCache(config.REDIS_URL, config.CACHE_TTL_SECONDS, config.CACHE_L1_SIZE)

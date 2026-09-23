"""Tests for the Phase 3 optimisations: backends, batching and caching."""

from __future__ import annotations

import asyncio
import math

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app import model as model_module
from backend.app.batching import MicroBatcher
from backend.app.cache import LRU, PredictionCache, cache
from backend.app.model import OnnxPipelineBackend, SklearnBackend, StudentBackend


@pytest.fixture(scope="module")
def backends() -> dict:
    return {"sklearn": SklearnBackend(), "onnx": OnnxPipelineBackend(), "student": StudentBackend()}


@pytest.fixture(scope="module")
def payloads(valid_payload: dict) -> list[dict]:
    rng = np.random.default_rng(0)
    hoods = ["CollgCr", "NAmes", "OldTown", "NridgHt", "Edwards", "Somerst"]
    out = []
    for _ in range(40):
        p = dict(valid_payload)
        p["OverallQual"] = int(rng.integers(1, 11))
        p["GrLivArea"] = float(rng.integers(600, 4000))
        p["YearBuilt"] = int(rng.integers(1900, 2010))
        p["GarageCars"] = int(rng.integers(0, 4))
        p["Neighborhood"] = str(rng.choice(hoods))
        out.append(p)
    return out


# --- model-level -----------------------------------------------------------


def test_onnx_pipeline_matches_sklearn(backends: dict, payloads: list[dict]) -> None:
    ref = backends["sklearn"].predict_log(payloads)
    got = backends["onnx"].predict_log(payloads)
    assert np.abs(ref - got).max() < 1e-4


def test_student_tracks_teacher(backends: dict, payloads: list[dict]) -> None:
    ref = backends["sklearn"].predict_log(payloads)
    got = backends["student"].predict_log(payloads)
    # Distillation fidelity is ~0.015 RMSE in log price on held-out requests.
    assert np.sqrt(np.mean((ref - got) ** 2)) < 0.05


def test_student_falls_back_for_non_form_columns(backends: dict, valid_payload: dict) -> None:
    extended = dict(valid_payload) | {"MSZoning": "RM"}
    student = backends["student"].predict_log([extended])
    onnx = backends["onnx"].predict_log([extended])
    assert student[0] == pytest.approx(onnx[0], abs=1e-6)


def test_batch_prediction_equals_single_predictions(backends: dict, payloads: list[dict]) -> None:
    for backend in backends.values():
        batch = backend.predict_log(payloads)
        single = np.array([backend.predict_log([p])[0] for p in payloads])
        assert np.allclose(batch, single, atol=1e-6)


def test_every_backend_can_be_selected(valid_payload: dict) -> None:
    try:
        for name in ("sklearn", "onnx", "student"):
            model_module.load_model(backend=name)
            assert model_module.backend_name() == name
            price, log_price = model_module.predict(valid_payload)
            assert 20_000 < price < 2_000_000
            assert math.isclose(math.expm1(log_price), price, rel_tol=1e-6)
    finally:
        model_module.load_model(force=True)


def test_unknown_backend_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown MODEL_BACKEND"):
        model_module.load_model(backend="tensorrt")
    model_module.load_model(force=True)


# --- dynamic batching --------------------------------------------------------


def test_micro_batcher_groups_concurrent_requests() -> None:
    calls: list[int] = []

    def fake_predict(rows):
        calls.append(len(rows))
        return [(float(r["x"]), float(r["x"])) for r in rows]

    async def scenario():
        batcher = MicroBatcher(fake_predict, max_size=16, max_wait_ms=20)
        await batcher.start()
        results = await asyncio.gather(*(batcher.submit({"x": i}) for i in range(40)))
        await batcher.stop()
        return results, batcher

    results, batcher = asyncio.run(scenario())
    assert [r[0] for r in results] == list(range(40))  # each caller gets its own row
    assert sum(calls) == 40
    assert max(calls) == 16  # capped at max_size
    assert len(calls) < 40  # requests were actually grouped
    assert batcher.snapshot()["avg_batch_size"] > 1


def test_micro_batcher_propagates_errors() -> None:
    def broken(rows):
        raise ValueError("boom")

    async def scenario():
        batcher = MicroBatcher(broken, max_size=8, max_wait_ms=5)
        await batcher.start()
        try:
            with pytest.raises(ValueError, match="boom"):
                await batcher.submit({"x": 1})
        finally:
            await batcher.stop()

    asyncio.run(scenario())


# --- caching -----------------------------------------------------------------


class FakeRedis:
    def __init__(self, fail: bool = False) -> None:
        self.store: dict[str, str] = {}
        self.fail = fail

    async def get(self, key):
        if self.fail:
            raise ConnectionError("redis down")
        return self.store.get(key)

    async def set(self, key, value, ex=None):
        if self.fail:
            raise ConnectionError("redis down")
        self.store[key] = value

    async def aclose(self):
        return None


def test_cache_key_is_order_independent(valid_payload: dict) -> None:
    reordered = dict(reversed(list(valid_payload.items())))
    assert PredictionCache.key(valid_payload, "student") == PredictionCache.key(
        reordered, "student"
    )
    assert PredictionCache.key(valid_payload, "student") != PredictionCache.key(
        valid_payload, "onnx"
    )


@pytest.fixture()
def isolated_cache():
    """Give each test a fresh L1 and no Redis; restore afterwards."""
    saved = (cache.l1, cache._client, cache._down_until)
    cache.l1, cache._client, cache._down_until = LRU(100, 60), None, 0.0
    yield cache
    cache.l1, cache._client, cache._down_until = saved


def test_lru_evicts_oldest_and_expires() -> None:
    lru = LRU(capacity=2, ttl=60)
    lru.put("a", {"v": 1})
    lru.put("b", {"v": 2})
    lru.get("a")  # a is now most recent
    lru.put("c", {"v": 3})
    assert lru.get("b") is None and lru.get("a") == {"v": 1}
    expired = LRU(capacity=2, ttl=-1)
    expired.put("x", {"v": 1})
    assert expired.get("x") is None


def test_api_serves_repeat_requests_from_l1(
    client: TestClient, valid_payload: dict, isolated_cache
) -> None:
    first = client.post("/predict", json=valid_payload).json()
    second = client.post("/predict", json=valid_payload).json()
    assert first["cached"] is False
    assert second["cached"] is True
    assert second["sale_price"] == first["sale_price"]
    assert isolated_cache.stats.l1_hits >= 1


def test_api_uses_redis_as_shared_l2(
    client: TestClient, valid_payload: dict, isolated_cache
) -> None:
    fake = FakeRedis()
    isolated_cache._client = fake
    first = client.post("/predict", json=valid_payload).json()
    assert first["cached"] is False
    assert len(fake.store) == 1  # written through to Redis
    isolated_cache.l1 = LRU(100, 60)  # e.g. a different worker: empty L1
    second = client.post("/predict", json=valid_payload).json()
    assert second["cached"] is True
    assert isolated_cache.stats.l2_hits == 1


def test_api_survives_a_dead_cache(client: TestClient, valid_payload: dict, isolated_cache) -> None:
    isolated_cache.l1 = None
    isolated_cache._client = FakeRedis(fail=True)
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200
    assert response.json()["cached"] is False
    assert isolated_cache.stats.errors >= 1


# --- API surface ---------------------------------------------------------------


def test_batch_endpoint(client: TestClient, valid_payload: dict) -> None:
    houses = [dict(valid_payload, OverallQual=q) for q in (3, 6, 9)]
    response = client.post("/predict/batch", json=houses)
    assert response.status_code == 200
    prices = [item["sale_price"] for item in response.json()]
    assert prices == sorted(prices)


def test_batch_endpoint_rejects_empty_list(client: TestClient) -> None:
    assert client.post("/predict/batch", json=[]).status_code == 422


def test_stats_and_health_report_phase3_state(client: TestClient, valid_payload: dict) -> None:
    client.post("/predict", json=valid_payload)
    stats = client.get("/stats").json()
    assert stats["model_backend"] == "student"
    assert stats["batching"]["enabled"] is True
    assert stats["batching"]["items"] >= 1
    assert client.get("/health").json()["model_backend"] == "student"


def test_greedy_batching_groups_requests_that_queue_behind_a_busy_model() -> None:
    import time as _time

    calls: list[int] = []

    def slow_predict(rows):
        _time.sleep(0.01)  # model busy -> later requests pile up in the queue
        calls.append(len(rows))
        return [(0.0, 0.0) for _ in rows]

    async def scenario():
        batcher = MicroBatcher(slow_predict, max_size=64, max_wait_ms=0)
        await batcher.start()
        await asyncio.gather(*(batcher.submit({"x": i}) for i in range(30)))
        await batcher.stop()

    asyncio.run(scenario())
    assert sum(calls) == 30
    assert len(calls) < 30  # no timer, yet requests were still grouped

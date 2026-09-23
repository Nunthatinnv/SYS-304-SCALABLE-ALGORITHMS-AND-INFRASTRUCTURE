"""FastAPI application exposing the House Prices baseline model."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Body, FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware

from . import config
from . import model as model_module
from .batching import MicroBatcher
from .cache import PredictionCache, cache
from .config import CORS_ORIGINS, SERVICE_NAME, SERVICE_VERSION
from .reference import get_reference
from .schemas import (
    USER_FIELDS,
    FieldSpec,
    HealthResponse,
    HouseFeatures,
    PredictResponse,
    SchemaResponse,
)

MAX_BATCH_REQUEST = 256

FIELD_LABELS: dict[str, str] = {
    "OverallQual": "Overall quality (1-10)",
    "GrLivArea": "Above-grade living area (sq ft)",
    "TotalBsmtSF": "Total basement area (sq ft)",
    "1stFlrSF": "First floor area (sq ft)",
    "YearBuilt": "Year built",
    "GarageCars": "Garage capacity (cars)",
    "FullBath": "Full bathrooms",
    "TotRmsAbvGrd": "Rooms above grade",
    "LotArea": "Lot area (sq ft)",
    "Neighborhood": "Neighborhood",
}


batcher = MicroBatcher(
    model_module.predict_many,
    max_size=config.BATCH_MAX_SIZE,
    max_wait_ms=config.BATCH_MAX_WAIT_MS,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Per worker process: load the model, connect Redis, start the batcher."""
    model_module.load_model()
    await cache.connect()
    if config.BATCHING_ENABLED:
        await batcher.start()
    yield
    await batcher.stop()
    await cache.close()


app = FastAPI(
    title="House Price Prediction API",
    description=(
        "Serves the SYS-304 Milestone 1 XGBoost baseline for the Ames, Iowa "
        "House Prices regression problem."
    ),
    version=SERVICE_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    """Report service liveness and whether the model is in memory."""
    return HealthResponse(
        status="ok",
        service=SERVICE_NAME,
        version=SERVICE_VERSION,
        model_loaded=model_module.is_loaded(),
        model_backend=model_module.backend_name(),
    )


@app.get("/stats", tags=["ops"])
def stats() -> dict[str, Any]:
    """Per-worker optimisation counters (cache hit rate, batch sizes)."""
    return {
        "pid": os.getpid(),
        "model_backend": model_module.backend_name(),
        "cache": cache.snapshot(),
        "batching": batcher.snapshot(),
    }


@app.get("/schema", response_model=SchemaResponse, tags=["model"])
def form_schema() -> SchemaResponse:
    """Describe the input form so the frontend does not hard-code fields."""
    reference = get_reference()
    fields: list[FieldSpec] = []

    for name in USER_FIELDS:
        label = FIELD_LABELS.get(name, name)
        if name in reference.categorical_columns:
            fields.append(
                FieldSpec(
                    name=name,
                    label=label,
                    type="select",
                    default=reference.defaults[name],
                    options=reference.categories_for(name),
                )
            )
        else:
            bounds = reference.numeric_ranges[name]
            fields.append(
                FieldSpec(
                    name=name,
                    label=label,
                    type="number",
                    default=reference.defaults[name],
                    minimum=bounds["min"],
                    maximum=bounds["max"],
                )
            )

    return SchemaResponse(fields=fields, total_model_columns=len(reference.columns))


async def _infer(payload: dict[str, Any]) -> tuple[float, float]:
    if batcher.running:
        return await batcher.submit(payload)
    return await run_in_threadpool(model_module.predict, payload)


def _response(sale_price: float, log_price: float, cached: bool) -> PredictResponse:
    return PredictResponse(sale_price=round(sale_price, 2), log_price=log_price, cached=cached)


@app.post("/predict", response_model=PredictResponse, tags=["model"])
async def predict(features: HouseFeatures) -> PredictResponse:
    """Predict the sale price for one house.

    Path: Redis exact-match cache -> dynamic batcher -> ONNX model.
    """
    payload = features.to_raw_columns()
    key = PredictionCache.key(payload, model_module.backend_name() or "none")
    hit = await cache.get(key)
    if hit is not None:
        return _response(hit["sale_price"], hit["log_price"], cached=True)

    try:
        model_module.validate(payload)
        sale_price, log_price = await _infer(payload)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    await cache.set(key, {"sale_price": sale_price, "log_price": log_price})
    return _response(sale_price, log_price, cached=False)


@app.post("/predict/batch", response_model=list[PredictResponse], tags=["model"])
async def predict_batch(
    houses: list[HouseFeatures] = Body(..., min_length=1, max_length=MAX_BATCH_REQUEST),
) -> list[PredictResponse]:
    """Client-side batching: score up to 256 houses in one model call."""
    payloads = [house.to_raw_columns() for house in houses]
    try:
        results = await run_in_threadpool(model_module.predict_many, payloads)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return [_response(price, log_price, cached=False) for price, log_price in results]

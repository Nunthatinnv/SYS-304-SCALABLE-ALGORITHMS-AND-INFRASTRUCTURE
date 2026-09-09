"""FastAPI application exposing the House Prices baseline model."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from . import model as model_module
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup rather than per request."""
    model_module.load_model()
    yield


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
    )


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


@app.post("/predict", response_model=PredictResponse, tags=["model"])
def predict(features: HouseFeatures) -> PredictResponse:
    """Predict the sale price for one house."""
    try:
        sale_price, log_price = model_module.predict(features.to_raw_columns())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return PredictResponse(sale_price=round(sale_price, 2), log_price=log_price)

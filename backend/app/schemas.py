"""Request and response schemas for the prediction API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .reference import get_reference

#: The subset of raw columns a user fills in. Every other column falls back to
#: its training-set default.
USER_FIELDS: list[str] = [
    "OverallQual",
    "GrLivArea",
    "TotalBsmtSF",
    "1stFlrSF",
    "YearBuilt",
    "GarageCars",
    "FullBath",
    "TotRmsAbvGrd",
    "LotArea",
    "Neighborhood",
]


class HouseFeatures(BaseModel):
    """User-supplied features for a single house.

    Field names mirror the raw Kaggle column names. ``1stFlrSF`` is not a valid
    Python identifier, so it is exposed through an alias.
    """

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    OverallQual: int = Field(..., ge=1, le=10, description="Overall material and finish quality")
    GrLivArea: float = Field(..., gt=0, le=20000, description="Above-grade living area (sq ft)")
    TotalBsmtSF: float = Field(..., ge=0, le=20000, description="Total basement area (sq ft)")
    FirstFlrSF: float = Field(
        ..., alias="1stFlrSF", gt=0, le=20000, description="First floor area (sq ft)"
    )
    YearBuilt: int = Field(..., ge=1800, le=2100, description="Original construction year")
    GarageCars: int = Field(..., ge=0, le=10, description="Garage capacity in cars")
    FullBath: int = Field(..., ge=0, le=10, description="Full bathrooms above grade")
    TotRmsAbvGrd: int = Field(..., ge=1, le=30, description="Total rooms above grade")
    LotArea: float = Field(..., gt=0, le=1000000, description="Lot size (sq ft)")
    Neighborhood: str = Field(..., description="Physical location within Ames city limits")

    @field_validator("Neighborhood")
    @classmethod
    def _known_neighborhood(cls, value: str) -> str:
        allowed = get_reference().categories_for("Neighborhood")
        if value not in allowed:
            raise ValueError(f"unknown Neighborhood '{value}'")
        return value

    def to_raw_columns(self) -> dict[str, Any]:
        """Return the user input keyed by raw Kaggle column names."""
        return self.model_dump(by_alias=True)


class PredictResponse(BaseModel):
    """Prediction returned by ``POST /predict``."""

    sale_price: float = Field(..., description="Predicted sale price in US dollars")
    log_price: float = Field(..., description="Raw model output, log1p(SalePrice)")
    currency: str = Field(default="USD")
    cached: bool = Field(default=False, description="Served from the Redis cache")


class FieldSpec(BaseModel):
    """Description of one form field, consumed by the frontend."""

    name: str
    label: str
    type: str
    default: Any
    minimum: float | None = None
    maximum: float | None = None
    options: list[str] | None = None


class SchemaResponse(BaseModel):
    """Everything the frontend needs to render its form."""

    fields: list[FieldSpec]
    total_model_columns: int


class HealthResponse(BaseModel):
    """Liveness/readiness payload."""

    status: str
    service: str
    version: str
    model_loaded: bool
    model_backend: str | None = None

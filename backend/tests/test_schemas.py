"""Unit tests for request validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from backend.app.schemas import HouseFeatures


def test_valid_payload_parses(valid_payload: dict) -> None:
    features = HouseFeatures(**valid_payload)
    assert features.OverallQual == 7
    assert features.FirstFlrSF == 856


def test_to_raw_columns_uses_kaggle_names(valid_payload: dict) -> None:
    raw = HouseFeatures(**valid_payload).to_raw_columns()
    assert "1stFlrSF" in raw
    assert "FirstFlrSF" not in raw
    assert set(raw) == set(valid_payload)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("OverallQual", 0),
        ("OverallQual", 11),
        ("GrLivArea", 0),
        ("GrLivArea", -500),
        ("YearBuilt", 1200),
        ("GarageCars", -1),
        ("TotRmsAbvGrd", 0),
        ("LotArea", -1),
    ],
)
def test_out_of_range_values_are_rejected(valid_payload: dict, field: str, bad_value) -> None:
    payload = dict(valid_payload)
    payload[field] = bad_value
    with pytest.raises(ValidationError):
        HouseFeatures(**payload)


def test_unknown_neighborhood_is_rejected(valid_payload: dict) -> None:
    payload = dict(valid_payload) | {"Neighborhood": "Atlantis"}
    with pytest.raises(ValidationError):
        HouseFeatures(**payload)


def test_missing_field_is_rejected(valid_payload: dict) -> None:
    payload = dict(valid_payload)
    del payload["GrLivArea"]
    with pytest.raises(ValidationError):
        HouseFeatures(**payload)


def test_extra_field_is_rejected(valid_payload: dict) -> None:
    payload = dict(valid_payload) | {"PoolOfLava": 1}
    with pytest.raises(ValidationError):
        HouseFeatures(**payload)

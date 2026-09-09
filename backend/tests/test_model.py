"""Unit tests for row construction and inference."""

from __future__ import annotations

import math

import pytest

from backend.app import model as model_module
from backend.app.reference import get_reference


def test_build_row_widens_input_to_full_column_set(valid_payload: dict) -> None:
    frame = model_module.build_row(valid_payload)
    reference = get_reference()
    assert frame.shape == (1, len(reference.columns))
    assert list(frame.columns) == reference.columns


def test_build_row_keeps_user_values_and_fills_the_rest(valid_payload: dict) -> None:
    frame = model_module.build_row(valid_payload)
    assert frame.loc[0, "OverallQual"] == valid_payload["OverallQual"]
    assert frame.loc[0, "1stFlrSF"] == valid_payload["1stFlrSF"]
    # A column the user never supplies falls back to its training default.
    assert frame.loc[0, "MSZoning"] == get_reference().defaults["MSZoning"]


def test_build_row_rejects_unknown_columns() -> None:
    with pytest.raises(ValueError, match="unknown columns"):
        model_module.build_row({"NotAColumn": 1})


def test_model_loads() -> None:
    assert model_module.load_model() is not None
    assert model_module.is_loaded() is True


def test_prediction_is_a_plausible_price(valid_payload: dict) -> None:
    sale_price, log_price = model_module.predict(valid_payload)
    assert math.isfinite(sale_price)
    assert 20_000 < sale_price < 2_000_000
    assert math.isclose(math.expm1(log_price), sale_price, rel_tol=1e-6)


def test_prediction_is_deterministic(valid_payload: dict) -> None:
    assert model_module.predict(valid_payload) == model_module.predict(valid_payload)


def test_better_quality_predicts_a_higher_price(valid_payload: dict) -> None:
    low = model_module.predict(dict(valid_payload) | {"OverallQual": 3})[0]
    high = model_module.predict(dict(valid_payload) | {"OverallQual": 9})[0]
    assert high > low

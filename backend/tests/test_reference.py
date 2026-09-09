"""Unit tests for the reference metadata loaded from defaults.json."""

from __future__ import annotations

from backend.app.reference import get_reference
from backend.app.schemas import USER_FIELDS


def test_reference_has_all_training_columns() -> None:
    reference = get_reference()
    assert len(reference.columns) == 79
    assert "Id" not in reference.columns
    assert "SalePrice" not in reference.columns


def test_numeric_and_categorical_partition_the_columns() -> None:
    reference = get_reference()
    numeric = set(reference.numeric_columns)
    categorical = set(reference.categorical_columns)
    assert numeric.isdisjoint(categorical)
    assert numeric | categorical == set(reference.columns)
    assert len(numeric) == 36
    assert len(categorical) == 43


def test_every_column_has_a_default() -> None:
    reference = get_reference()
    base = reference.base_row()
    assert set(base) == set(reference.columns)
    assert all(value is not None for value in base.values())


def test_base_row_is_a_fresh_copy() -> None:
    reference = get_reference()
    first = reference.base_row()
    first["OverallQual"] = -999
    assert reference.base_row()["OverallQual"] != -999


def test_user_fields_are_real_columns() -> None:
    reference = get_reference()
    assert set(USER_FIELDS).issubset(set(reference.columns))


def test_categorical_defaults_are_valid_categories() -> None:
    reference = get_reference()
    for column in reference.categorical_columns:
        assert reference.defaults[column] in reference.categories_for(column)

"""Reference metadata for the model's input space.

The Milestone 1 pipeline was fitted on a DataFrame with 79 raw columns. At
serving time the user only fills in a handful of high-signal fields, so this
module supplies the column order and a sensible default for every remaining
column (median for numeric columns, mode for categorical ones), computed once
from ``data/train.csv`` and frozen into ``defaults.json``.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from .config import DEFAULTS_PATH


class Reference:
    """Immutable view over ``defaults.json``."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.columns: list[str] = list(payload["columns"])
        self.numeric_columns: list[str] = list(payload["numeric_columns"])
        self.categorical_columns: list[str] = list(payload["categorical_columns"])
        self.defaults: dict[str, Any] = dict(payload["defaults"])
        self.categories: dict[str, list[str]] = {
            k: list(v) for k, v in payload["categories"].items()
        }
        self.numeric_ranges: dict[str, dict[str, float]] = {
            k: dict(v) for k, v in payload["numeric_ranges"].items()
        }

    def base_row(self) -> dict[str, Any]:
        """Return a fresh full-width row filled with default values."""
        return {column: self.defaults[column] for column in self.columns}

    def categories_for(self, column: str) -> list[str]:
        """Return the allowed values for a categorical column."""
        return self.categories.get(column, [])


@lru_cache(maxsize=1)
def get_reference() -> Reference:
    """Load and cache the reference metadata."""
    with DEFAULTS_PATH.open(encoding="utf-8") as handle:
        return Reference(json.load(handle))

"""Model loading and inference.

The Milestone 1 artefact is a full scikit-learn ``Pipeline`` (ColumnTransformer
plus ``XGBRegressor``) that was trained on ``log1p(SalePrice)``. This module
loads it once, widens a partial user input to the full 79-column row the
pipeline expects, and inverts the log transform on the way out.
"""

from __future__ import annotations

import threading
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .config import MODEL_PATH
from .reference import get_reference

_model: Any | None = None
_lock = threading.Lock()


def load_model(force: bool = False) -> Any:
    """Load the pipeline from disk, caching it for subsequent calls."""
    global _model
    with _lock:
        if _model is None or force:
            if not MODEL_PATH.exists():
                raise FileNotFoundError(f"model artefact not found at {MODEL_PATH}")
            _model = joblib.load(MODEL_PATH)
        return _model


def is_loaded() -> bool:
    """Return True when a model is already in memory."""
    return _model is not None


def build_row(user_input: dict[str, Any]) -> pd.DataFrame:
    """Widen a partial input to the full column set the pipeline was fitted on.

    Unknown keys are rejected so that a typo cannot silently fall back to a
    default value.
    """
    reference = get_reference()
    unknown = set(user_input) - set(reference.columns)
    if unknown:
        raise ValueError(f"unknown columns: {sorted(unknown)}")

    row = reference.base_row()
    row.update(user_input)
    return pd.DataFrame([row], columns=reference.columns)


def predict(user_input: dict[str, Any]) -> tuple[float, float]:
    """Predict a sale price.

    Returns a ``(sale_price, log_price)`` pair, where ``log_price`` is the raw
    model output and ``sale_price`` is ``expm1(log_price)`` in dollars.
    """
    frame = build_row(user_input)
    log_price = float(np.asarray(load_model().predict(frame)).ravel()[0])
    return float(np.expm1(log_price)), log_price

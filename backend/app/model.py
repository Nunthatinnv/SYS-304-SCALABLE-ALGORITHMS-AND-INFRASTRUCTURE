"""Model loading and inference.

Phase 3 adds pluggable inference backends, selected with ``MODEL_BACKEND``:

``sklearn`` (Phase 2 / naive)
    The Milestone 1 ``Pipeline`` pickle (ColumnTransformer + XGBRegressor)
    called on a 79-column pandas DataFrame.
``onnx`` (model-format optimisation)
    The same pipeline exported to one ONNX graph (``export_onnx.py``) and run
    by onnxruntime. Numerically equivalent (max |diff| ~1e-5 in log price);
    no pandas / scikit-learn / xgboost on the request path.
``student`` (distillation + ONNX, default)
    A 150-tree student distilled from the teacher over the 10 fields the API
    accepts (``distill_student.py``), fed a 10-wide float32 vector. Requests
    that carry any other column fall back to the ``onnx`` backend, so the
    service never silently ignores an input.

Every backend exposes ``predict_log(rows) -> np.ndarray`` for a *batch* of
rows, which is what the dynamic batcher (``batching.py``) calls.
"""

from __future__ import annotations

import json
import threading
from typing import Any, Protocol

import numpy as np

from . import config
from .reference import get_reference


class Backend(Protocol):
    name: str

    def predict_log(self, rows: list[dict[str, Any]]) -> np.ndarray: ...


def _ort_session(path):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # One thread per session: parallelism comes from Gunicorn workers, and
    # letting every worker spawn N intra-op threads oversubscribes the CPU.
    options.intra_op_num_threads = config.ORT_THREADS
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def _require(path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"model artefact not found at {path}")


class SklearnBackend:
    """Phase 2 baseline: pickle + pandas DataFrame."""

    name = "sklearn"

    def __init__(self) -> None:
        import joblib

        _require(config.MODEL_PATH)
        self.pipeline = joblib.load(config.MODEL_PATH)

    def predict_log(self, rows: list[dict[str, Any]]) -> np.ndarray:
        import pandas as pd

        reference = get_reference()
        full = [reference.base_row() | row for row in rows]
        frame = pd.DataFrame(full, columns=reference.columns)
        return np.asarray(self.pipeline.predict(frame), dtype=np.float64).ravel()


class OnnxPipelineBackend:
    """Full 79-column pipeline exported to ONNX."""

    name = "onnx"

    def __init__(self) -> None:
        _require(config.ONNX_PATH)
        self.session = _ort_session(config.ONNX_PATH)
        meta = json.loads(config.ONNX_META_PATH.read_text())
        numeric = set(meta["numeric"])
        reference = get_reference()
        self.columns: list[tuple[str, str, bool]] = [
            (col, name, col in numeric)
            for col, name in zip(meta["columns"], meta["input_names"], strict=True)
        ]
        # Pre-built 1-row default tensors; per request we only tile them and
        # overwrite the handful of user-supplied columns.
        self.defaults = {
            col: self._tensor([reference.defaults[col]], is_num) for col, _, is_num in self.columns
        }

    @staticmethod
    def _tensor(values: list[Any], is_numeric: bool) -> np.ndarray:
        if is_numeric:
            return np.asarray(values, dtype=np.float32).reshape(-1, 1)
        return np.asarray(["" if v is None else str(v) for v in values], dtype=object).reshape(
            -1, 1
        )

    def predict_log(self, rows: list[dict[str, Any]]) -> np.ndarray:
        n = len(rows)
        supplied = set().union(*rows) if rows else set()
        feeds = {}
        for col, name, is_num in self.columns:
            if col in supplied:
                default = self.defaults[col][0, 0]
                feeds[name] = self._tensor([row.get(col, default) for row in rows], is_num)
            else:
                feeds[name] = np.repeat(self.defaults[col], n, axis=0)
        return self.session.run(None, feeds)[0].ravel().astype(np.float64)


class StudentBackend:
    """Distilled 10-feature student (ONNX), with a full-pipeline fallback."""

    name = "student"

    def __init__(self) -> None:
        _require(config.STUDENT_PATH)
        self.session = _ort_session(config.STUDENT_PATH)
        meta = json.loads(config.STUDENT_META_PATH.read_text())
        self.features: list[str] = meta["features"]
        self.feature_set = frozenset(self.features)
        self.codes: dict[str, int] = meta["neighborhood_codes"]
        self._fallback: OnnxPipelineBackend | None = None
        self._fallback_lock = threading.Lock()

    @property
    def fallback(self) -> OnnxPipelineBackend:
        with self._fallback_lock:
            if self._fallback is None:
                self._fallback = OnnxPipelineBackend()
            return self._fallback

    def encode(self, rows: list[dict[str, Any]]) -> np.ndarray:
        out = np.empty((len(rows), len(self.features)), dtype=np.float32)
        for i, row in enumerate(rows):
            for j, feature in enumerate(self.features):
                value = row[feature]
                out[i, j] = self.codes[value] if feature == "Neighborhood" else value
        return out

    def predict_log(self, rows: list[dict[str, Any]]) -> np.ndarray:
        if all(row.keys() == self.feature_set for row in rows):
            return (
                self.session.run(None, {"features": self.encode(rows)})[0]
                .ravel()
                .astype(np.float64)
            )
        return self.fallback.predict_log(rows)


BACKENDS: dict[str, type] = {
    "sklearn": SklearnBackend,
    "onnx": OnnxPipelineBackend,
    "student": StudentBackend,
}

_model: Backend | None = None
_lock = threading.Lock()


def load_model(force: bool = False, backend: str | None = None) -> Backend:
    """Load the configured backend once and cache it for subsequent calls."""
    global _model
    with _lock:
        wanted = backend or config.MODEL_BACKEND
        if _model is None or force or _model.name != wanted:
            if wanted not in BACKENDS:
                raise ValueError(f"unknown MODEL_BACKEND '{wanted}'")
            _model = BACKENDS[wanted]()
        return _model


def is_loaded() -> bool:
    """Return True when a model is already in memory."""
    return _model is not None


def backend_name() -> str | None:
    return _model.name if _model is not None else None


def validate(user_input: dict[str, Any]) -> None:
    """Reject unknown keys so that a typo cannot silently use a default."""
    unknown = set(user_input) - set(get_reference().columns)
    if unknown:
        raise ValueError(f"unknown columns: {sorted(unknown)}")


def build_row(user_input: dict[str, Any]):
    """Widen a partial input to the full 79-column DataFrame (naive path)."""
    import pandas as pd

    validate(user_input)
    reference = get_reference()
    row = reference.base_row()
    row.update(user_input)
    return pd.DataFrame([row], columns=reference.columns)


def predict_many(rows: list[dict[str, Any]]) -> list[tuple[float, float]]:
    """Predict a batch; returns ``(sale_price, log_price)`` per row."""
    for row in rows:
        validate(row)
    logs = load_model().predict_log(rows)
    return [(float(np.expm1(v)), float(v)) for v in logs]


def predict(user_input: dict[str, Any]) -> tuple[float, float]:
    """Predict one sale price as ``(sale_price, log_price)``."""
    return predict_many([user_input])[0]

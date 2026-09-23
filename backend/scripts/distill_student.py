"""Distil the Milestone 1 teacher into a small student model (Phase 3).

Key observation: the API only lets a user set 10 fields; the other 69 columns
are *always* filled with the frozen defaults from ``defaults.json``. The
teacher's effective input space at serving time is therefore 10-dimensional,
and a student that only sees those 10 fields can reproduce it with far less
work per request (no 79-column row, no ColumnTransformer, no 285-wide one-hot
vector, fewer and shallower trees on a 10-wide float vector).

Procedure
---------
1. Re-create the Milestone 1 train/validation split (``random_state=42``).
2. Generate synthetic *serving requests* from the training split only:
   real rows, jittered rows (+/-20 %, random neighbourhood swaps) and
   independently resampled marginals, so the student covers form combinations
   that never occur together in the data.
3. Label them with the teacher (full pipeline, 69 columns at their defaults).
4. Fit a small ``XGBRegressor`` on those soft labels (``Neighborhood`` is an
   ordinal code - trees do not care about the ordering).
5. Report fidelity to the teacher and accuracy on the real validation rows,
   then export the student to ONNX.

Run from the repository root::

    python backend/scripts/distill_student.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
from onnxmltools import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TEACHER = REPO_ROOT / "models" / "xgb_baseline_pipeline.pkl"
DEFAULTS = REPO_ROOT / "backend" / "app" / "defaults.json"
TRAIN_CSV = REPO_ROOT / "data" / "train.csv"
OUT_NATIVE = REPO_ROOT / "models" / "student_xgb.json"
OUT_ONNX = REPO_ROOT / "models" / "student_xgb.onnx"
OUT_META = REPO_ROOT / "models" / "student_meta.json"

#: Must match ``backend.app.schemas.USER_FIELDS`` (order = student feature order).
FEATURES = [
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
INTEGER_FIELDS = ["OverallQual", "YearBuilt", "GarageCars", "FullBath", "TotRmsAbvGrd"]
STUDENT_PARAMS = dict(
    n_estimators=150, max_depth=4, learning_rate=0.1, subsample=0.9, random_state=42
)
N_TRAIN, N_HOLDOUT = 60_000, 10_000
SEED = 42


def synthesize(base: pd.DataFrame, neighborhoods: list[str], n: int, rng) -> pd.DataFrame:
    """Sample ``n`` plausible serving requests from the training rows."""
    numeric = FEATURES[:-1]
    real = base.sample(n // 5, replace=True, random_state=int(rng.integers(1e9)))

    jitter = base.sample(2 * n // 5, replace=True, random_state=int(rng.integers(1e9))).copy()
    for col in numeric:
        jitter[col] = jitter[col] * rng.uniform(0.8, 1.2, len(jitter))
    swap = rng.random(len(jitter)) < 0.3
    jitter.loc[swap, "Neighborhood"] = rng.choice(neighborhoods, swap.sum())

    rest = n - len(real) - len(jitter)
    marginal = pd.DataFrame({c: rng.choice(base[c].to_numpy(), rest) for c in FEATURES})

    out = pd.concat([real, jitter, marginal], ignore_index=True)
    out[INTEGER_FIELDS] = out[INTEGER_FIELDS].round()
    out["OverallQual"] = out["OverallQual"].clip(1, 10)
    out["TotRmsAbvGrd"] = out["TotRmsAbvGrd"].clip(1, 30)
    return out


def widen(requests: pd.DataFrame, ref: dict) -> pd.DataFrame:
    """Build the 79-column teacher input exactly as the API does."""
    rows = pd.DataFrame([ref["defaults"]] * len(requests))[ref["columns"]]
    for col in FEATURES:
        rows[col] = requests[col].to_numpy()
    rows[ref["numeric_columns"]] = rows[ref["numeric_columns"]].astype(float)
    return rows


def encode(requests: pd.DataFrame, code: dict[str, int]) -> np.ndarray:
    """10 serving fields -> float32 student feature matrix."""
    frame = requests[FEATURES].copy()
    frame["Neighborhood"] = frame["Neighborhood"].map(code)
    return frame.to_numpy(np.float32)


def rmse(a, b) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def node_count(model: XGBRegressor) -> int:
    return int(len(model.get_booster().trees_to_dataframe()))


def main() -> None:
    ref = json.loads(DEFAULTS.read_text())
    neighborhoods = ref["categories"]["Neighborhood"]
    code = {name: i for i, name in enumerate(neighborhoods)}
    teacher = joblib.load(TEACHER)

    data = pd.read_csv(TRAIN_CSV)
    y = np.log1p(data["SalePrice"])
    X = data.drop(columns=["Id", "SalePrice"])
    X_tr, X_va, _, y_va = train_test_split(X, y, test_size=0.2, random_state=SEED)

    rng = np.random.default_rng(SEED)
    base = X_tr[FEATURES].reset_index(drop=True)
    train_req = synthesize(base, neighborhoods, N_TRAIN, rng)
    hold_req = synthesize(base, neighborhoods, N_HOLDOUT, rng)
    train_soft = teacher.predict(widen(train_req, ref))
    hold_soft = teacher.predict(widen(hold_req, ref))

    student = XGBRegressor(**STUDENT_PARAMS, n_jobs=-1)
    student.fit(encode(train_req, code), train_soft)
    student.save_model(OUT_NATIVE)

    # --- evaluation ------------------------------------------------------
    va_req = X_va[FEATURES].reset_index(drop=True)
    teacher_va_full = teacher.predict(X_va)  # all 79 real columns
    teacher_va_serve = teacher.predict(widen(va_req, ref))  # 10 fields + defaults
    student_va = student.predict(encode(va_req, code))
    metrics = {
        "fidelity_rmse_log_holdout": rmse(student.predict(encode(hold_req, code)), hold_soft),
        "fidelity_rmse_log_validation": rmse(student_va, teacher_va_serve),
        "val_rmse_log_teacher_all_79_columns": rmse(teacher_va_full, y_va),
        "val_rmse_log_teacher_serving_protocol": rmse(teacher_va_serve, y_va),
        "val_rmse_log_student": rmse(student_va, y_va),
        "val_rmse_usd_teacher_serving_protocol": rmse(np.expm1(teacher_va_serve), np.expm1(y_va)),
        "val_rmse_usd_student": rmse(np.expm1(student_va), np.expm1(y_va)),
        "teacher_trees": int(teacher.named_steps["regressor"].n_estimators),
        "teacher_nodes": node_count(teacher.named_steps["regressor"]),
        "student_trees": STUDENT_PARAMS["n_estimators"],
        "student_nodes": node_count(student),
    }

    # --- ONNX export -----------------------------------------------------
    onx = convert_xgboost(
        student,
        initial_types=[("features", FloatTensorType([None, len(FEATURES)]))],
        target_opset=15,
    )
    onnx.checker.check_model(onx)
    OUT_ONNX.write_bytes(onx.SerializeToString())
    session = ort.InferenceSession(str(OUT_ONNX), providers=["CPUExecutionProvider"])
    onnx_va = session.run(None, {"features": encode(va_req, code)})[0].ravel()
    metrics["onnx_vs_native_max_abs_log"] = float(np.abs(onnx_va - student_va).max())
    assert metrics["onnx_vs_native_max_abs_log"] < 1e-4

    OUT_META.write_text(
        json.dumps(
            {
                "features": FEATURES,
                "neighborhood_codes": code,
                "params": STUDENT_PARAMS,
                "synthetic_train_rows": N_TRAIN,
                "metrics": metrics,
            },
            indent=2,
        )
    )
    for key, value in metrics.items():
        print(f"{key:42s} {value:.4f}" if isinstance(value, float) else f"{key:42s} {value}")


if __name__ == "__main__":
    main()

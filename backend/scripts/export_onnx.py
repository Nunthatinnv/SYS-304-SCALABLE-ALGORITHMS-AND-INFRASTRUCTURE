"""Export the Milestone 1 pipeline to ONNX (Phase 3, model-format optimisation).

The whole scikit-learn ``Pipeline`` -- ColumnTransformer (imputers, scaler,
one-hot encoder) plus the ``XGBRegressor`` -- becomes a single ONNX graph that
``onnxruntime`` executes without pandas, scikit-learn or xgboost in the loop.

Three details make a naive conversion wrong, and both are handled here:

1. **String imputer.** skl2onnx can only convert a categorical ``SimpleImputer``
   whose ``missing_values`` is a string. We copy the fitted pipeline and set
   ``missing_values=""`` (the learned ``statistics_`` are unchanged). Callers
   therefore pass missing categoricals as ``""`` instead of ``NaN``.
2. **Scaler precision.** XGBoost's histogram split thresholds coincide with
   training values, so a scaled feature that is 1 ulp off flips a branch. The
   scaler is therefore computed in float64 (``div_cast``) with the exact
   float64 ``mean_`` / ``scale_`` constants, as in scikit-learn.
3. **Sparse zeros are "missing" to XGBoost.** The ColumnTransformer output is
   sparse (density < ``sparse_threshold``), and XGBoost treats *absent* sparse
   entries as missing values, not as 0.0. ONNX works on dense tensors, so we
   insert ``Where(x == 0, NaN, x)`` in front of the tree ensemble to reproduce
   the training-time semantics. Without this, predictions drift by up to
   ~0.8 in log-price.

Run from the repository root::

    python backend/scripts/export_onnx.py
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import joblib
import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
from onnx import TensorProto, helper, numpy_helper
from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost
from skl2onnx import convert_sklearn, update_registered_converter
from skl2onnx.common.data_types import FloatTensorType, StringTensorType
from skl2onnx.common.shape_calculator import calculate_linear_regressor_output_shapes
from xgboost import XGBRegressor

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SRC = REPO_ROOT / "models" / "xgb_baseline_pipeline.pkl"
DST = REPO_ROOT / "models" / "xgb_pipeline.onnx"
META = REPO_ROOT / "models" / "xgb_pipeline.onnx.json"
TRAIN_CSV = REPO_ROOT / "data" / "train.csv"
TEST_CSV = REPO_ROOT / "data" / "test.csv"

update_registered_converter(
    XGBRegressor,
    "XGBoostXGBRegressor",
    calculate_linear_regressor_output_shapes,
    convert_xgboost,
)


def _sparse_zero_as_missing(model: onnx.ModelProto) -> onnx.ModelProto:
    """Insert ``Where(x == 0, NaN, x)`` before the TreeEnsembleRegressor."""
    graph = model.graph
    idx, tree = next(
        (i, n) for i, n in enumerate(graph.node) if n.op_type == "TreeEnsembleRegressor"
    )
    dense_in = tree.input[0]
    graph.initializer.extend(
        [
            helper.make_tensor("zero_f", TensorProto.FLOAT, [], [0.0]),
            helper.make_tensor("nan_f", TensorProto.FLOAT, [], [float("nan")]),
        ]
    )
    is_zero = helper.make_node("Equal", [dense_in, "zero_f"], ["is_zero"], name="IsZero")
    masked = helper.make_node(
        "Where", ["is_zero", "nan_f", dense_in], ["sparse_masked"], name="ZeroAsMissing"
    )
    tree.input[0] = "sparse_masked"
    graph.node.insert(idx, masked)
    graph.node.insert(idx, is_zero)
    return model


def _exact_scaler_constants(model: onnx.ModelProto, scaler) -> onnx.ModelProto:
    """Overwrite the scaler's Sub/Div constants with the exact float64 values.

    With ``div_cast`` skl2onnx computes the scaling in float64, but it stores
    ``mean_`` / ``scale_`` after a round-trip through float32. Restoring the
    exact constants makes the transformed features bit-identical to sklearn's.
    """
    exact = {"Sub": scaler.mean_, "Div": scaler.scale_}
    inits = {init.name: init for init in model.graph.initializer}
    for node in model.graph.node:
        if node.op_type in exact:
            const = next(name for name in node.input if name in inits)
            inits[const].CopyFrom(
                numpy_helper.from_array(np.asarray(exact[node.op_type], np.float64), const)
            )
    return model


def main() -> None:
    pipeline = joblib.load(SRC)
    onnx_ready = copy.deepcopy(pipeline)
    pre = onnx_ready.named_steps["preprocessor"]
    pre.named_transformers_["cat"].named_steps["imputer"].missing_values = ""

    numeric = list(pre.transformers_[0][2])
    columns = list(pre.feature_names_in_)
    initial_types = [
        (c, FloatTensorType([None, 1]) if c in numeric else StringTensorType([None, 1]))
        for c in columns
    ]
    scaler = pre.named_transformers_["num"].named_steps["scaler"]
    model = convert_sklearn(
        onnx_ready,
        initial_types=initial_types,
        target_opset={"": 17, "ai.onnx.ml": 3},
        # Scale in float64 then cast, exactly like scikit-learn does. In pure
        # float32 some scaled values land 1 ulp away from an XGBoost split
        # threshold (hist cut points coincide with training values) and flip
        # the branch.
        options={id(scaler): {"div": "div_cast"}},
    )
    model = _exact_scaler_constants(model, scaler)
    model = _sparse_zero_as_missing(model)
    onnx.checker.check_model(model)
    DST.write_bytes(model.SerializeToString())

    # skl2onnx renames inputs that are not valid identifiers (e.g. 1stFlrSF ->
    # _1stFlrSF); record the mapping so the server can build its feed dict.
    session = ort.InferenceSession(str(DST), providers=["CPUExecutionProvider"])
    input_names = [i.name for i in session.get_inputs()]
    META.write_text(
        json.dumps({"columns": columns, "input_names": input_names, "numeric": numeric}, indent=2)
    )

    # Parity check against the original pipeline on the full training set.
    frame = pd.concat(
        [
            pd.read_csv(TRAIN_CSV).drop(columns=["Id", "SalePrice"]),
            pd.read_csv(TEST_CSV).drop(columns=["Id"]),
        ],
        ignore_index=True,
    )
    feeds = {
        name: (
            frame[c].to_numpy(np.float32).reshape(-1, 1)
            if c in numeric
            else frame[c].fillna("").astype(str).to_numpy().reshape(-1, 1)
        )
        for c, name in zip(columns, input_names, strict=True)
    }
    onnx_pred = session.run(None, feeds)[0].ravel()
    ref_pred = pipeline.predict(frame)
    diff = np.abs(onnx_pred - ref_pred)
    print(f"wrote {DST.relative_to(REPO_ROOT)} ({DST.stat().st_size / 1024:.0f} KiB)")
    print(f"parity vs sklearn on {len(frame)} rows: max |diff| = {diff.max():.2e} (log scale)")
    assert diff.max() < 1e-4, "ONNX export does not match the original pipeline"


if __name__ == "__main__":
    main()

"""Week 5 benchmark: naive model vs optimised models (latency, memory, accuracy).

Sends the same batch of requests to every model variant and reports:

* load time, artefact size, resident memory (RSS) after import + load and peak
* single-request latency (p50 / p95 / p99 / mean) over N sequential requests
* batched throughput (rows/s) at batch sizes 32 and 256
* accuracy on the real validation rows under the serving protocol (the 10 form
  fields + defaults), plus max deviation from the naive model

Variants (all use the production code in backend/app/model.py, except
``student_xgb`` which calls the distilled booster through xgboost directly to
isolate the effect of distillation from the effect of ONNX):

    naive_sklearn   Phase 2: pickle + pandas + scikit-learn + xgboost
    onnx_pipeline   same model, one ONNX graph in onnxruntime
    student_xgb     distilled 10-feature student, native xgboost
    student_onnx    distilled student in onnxruntime  (Phase 3 default)

Each variant runs in a fresh subprocess so memory numbers are not polluted by
libraries another variant imported. Usage (repository root)::

    python benchmarks/bench_model.py              # full run, writes results/
    python benchmarks/bench_model.py --requests 500 --repeats 1   # quick
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "benchmarks" / "results"
VARIANTS = ["naive_sklearn", "onnx_pipeline", "student_xgb", "student_onnx"]
FIELDS = [
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
ARTEFACTS = {
    "naive_sklearn": ["models/xgb_baseline_pipeline.pkl"],
    "onnx_pipeline": ["models/xgb_pipeline.onnx"],
    "student_xgb": ["models/student_xgb.json"],
    "student_onnx": ["models/student_xgb.onnx"],
}


# --------------------------------------------------------------------------
# Child process: measure one variant
# --------------------------------------------------------------------------


def _rss_mb() -> float:
    import psutil

    return psutil.Process().memory_info().rss / 2**20


def _peak_rss_mb() -> float:
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 2**20 if sys.platform == "darwin" else peak / 1024


def _load_requests() -> tuple[list[dict], list[float]]:
    """Validation rows (Milestone 1 split) reduced to the 10 API fields."""
    import csv
    import math

    with open(REPO_ROOT / "data" / "train.csv", newline="") as handle:
        rows = list(csv.DictReader(handle))
    # Re-create sklearn's train_test_split(test_size=0.2, random_state=42)
    # without importing sklearn here (it would inflate every variant's RSS).
    order = _sklearn_split_indices(len(rows))
    requests, truth = [], []
    for i in order:
        row = rows[i]
        req = {}
        for f in FIELDS:
            if f == "Neighborhood":
                req[f] = row[f]
            elif f in {"GrLivArea", "TotalBsmtSF", "1stFlrSF", "LotArea"}:
                req[f] = float(row[f])
            else:
                req[f] = int(row[f])
        requests.append(req)
        truth.append(math.log1p(float(row["SalePrice"])))
    return requests, truth


def _sklearn_split_indices(n: int) -> list[int]:
    """Indices of the validation split of train_test_split(n, 0.2, rs=42)."""
    import numpy as np

    rng = np.random.RandomState(42)
    n_test = int(np.ceil(0.2 * n))
    permutation = rng.permutation(n)
    return [int(i) for i in permutation[:n_test]]


def _make_predictor(variant: str):
    """Return ``fn(list[dict]) -> sequence[float]`` of log prices."""
    os.environ.setdefault("ORT_THREADS", "1")
    sys.path.insert(0, str(REPO_ROOT))
    if variant == "student_xgb":
        import numpy as np
        import xgboost as xgb

        meta = json.loads((REPO_ROOT / "models" / "student_meta.json").read_text())
        booster = xgb.Booster()
        booster.load_model(REPO_ROOT / "models" / "student_xgb.json")
        codes, feats = meta["neighborhood_codes"], meta["features"]

        def predict(rows):
            x = np.array(
                [[codes[r[f]] if f == "Neighborhood" else r[f] for f in feats] for r in rows],
                dtype=np.float32,
            )
            return booster.inplace_predict(x)

        return predict

    from backend.app import model as model_module

    backend = {
        "naive_sklearn": "sklearn",
        "onnx_pipeline": "onnx",
        "student_onnx": "student",
    }[variant]
    return model_module.load_model(backend=backend).predict_log


def _percentile(sorted_values: list[float], q: float) -> float:
    idx = min(len(sorted_values) - 1, max(0, round(q / 100 * (len(sorted_values) - 1))))
    return sorted_values[idx]


def measure(variant: str, n_requests: int, repeats: int) -> dict:
    rss_start = _rss_mb()
    t0 = time.perf_counter()
    predict = _make_predictor(variant)
    load_s = time.perf_counter() - t0
    rss_loaded = _rss_mb()

    requests, truth = _load_requests()
    workload = [requests[i % len(requests)] for i in range(n_requests)]

    for req in workload[:50]:  # warm-up
        predict([req])

    # Single-request latency: the naive API scores one house per call.
    best = None
    for _ in range(repeats):
        lat = []
        for req in workload:
            s = time.perf_counter()
            predict([req])
            lat.append((time.perf_counter() - s) * 1000)
        if best is None or statistics.median(lat) < statistics.median(best):
            best = lat
    lat = sorted(best)

    # Batched throughput (what dynamic batching exploits).
    throughput = {}
    for batch_size in (32, 256):
        batches = [workload[i : i + batch_size] for i in range(0, len(workload), batch_size)]
        batches = [b for b in batches if len(b) == batch_size] or [workload[:batch_size]]
        s = time.perf_counter()
        rows = 0
        for _ in range(repeats):
            for b in batches:
                predict(b)
                rows += len(b)
        throughput[batch_size] = rows / (time.perf_counter() - s)

    # Accuracy under the serving protocol.
    preds = [float(v) for v in predict(requests)]
    rmse = statistics.fmean((p - t) ** 2 for p, t in zip(preds, truth, strict=True)) ** 0.5

    return {
        "variant": variant,
        "artefact_kb": sum((REPO_ROOT / p).stat().st_size for p in ARTEFACTS[variant]) / 1024,
        "load_s": load_s,
        "rss_after_load_mb": rss_loaded,
        "rss_model_delta_mb": rss_loaded - rss_start,
        "rss_peak_mb": _peak_rss_mb(),
        "n_requests": n_requests,
        "lat_mean_ms": statistics.fmean(lat),
        "lat_p50_ms": _percentile(lat, 50),
        "lat_p95_ms": _percentile(lat, 95),
        "lat_p99_ms": _percentile(lat, 99),
        "single_req_per_s": 1000 / statistics.fmean(lat),
        "batch32_rows_per_s": throughput[32],
        "batch256_rows_per_s": throughput[256],
        "val_rmse_log": rmse,
        "predictions": preds,
    }


# --------------------------------------------------------------------------
# Parent process: run every variant, compare, write results
# --------------------------------------------------------------------------


def _markdown(results: list[dict]) -> str:
    base = results[0]
    head = (
        "| Variant | Artefact (KB) | RSS after load (MB) | p50 (ms) | p95 (ms) | p99 (ms) "
        "| Speed-up (p50) | Batch-256 rows/s | Val RMSE (log) | Max |Δ| vs naive |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    lines = []
    for r in results:
        lines.append(
            f"| {r['variant']} | {r['artefact_kb']:.0f} | {r['rss_after_load_mb']:.0f} "
            f"| {r['lat_p50_ms']:.3f} | {r['lat_p95_ms']:.3f} | {r['lat_p99_ms']:.3f} "
            f"| {base['lat_p50_ms'] / r['lat_p50_ms']:.1f}x | {r['batch256_rows_per_s']:,.0f} "
            f"| {r['val_rmse_log']:.4f} | {r['max_abs_diff_vs_naive']:.4f} |"
        )
    return head + "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--requests", type=int, default=2000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--variants", nargs="*", default=VARIANTS)
    parser.add_argument("--child", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.child:
        print(json.dumps(measure(args.child, args.requests, args.repeats)))
        return

    results = []
    for variant in args.variants:
        print(f"-> {variant}", flush=True)
        out = subprocess.run(
            [
                sys.executable,
                __file__,
                "--child",
                variant,
                "--requests",
                str(args.requests),
                "--repeats",
                str(args.repeats),
            ],
            check=True,
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        )
        results.append(json.loads(out.stdout.strip().splitlines()[-1]))

    naive = results[0]["predictions"]
    for r in results:
        r["max_abs_diff_vs_naive"] = max(
            abs(a - b) for a, b in zip(r["predictions"], naive, strict=True)
        )
        r.pop("predictions")

    RESULTS.mkdir(parents=True, exist_ok=True)
    import platform

    import psutil

    env = {
        "python": platform.python_version(),
        "machine": platform.machine(),
        "system": platform.system(),
        "cpus": os.cpu_count(),
        "memory_gb": round(psutil.virtual_memory().total / 2**30, 1),
        "requests": args.requests,
        "repeats": args.repeats,
    }
    (RESULTS / "model_bench.json").write_text(
        json.dumps({"env": env, "results": results}, indent=2)
    )
    table = _markdown(results)
    (RESULTS / "model_bench.md").write_text(table)
    print(table)


if __name__ == "__main__":
    main()

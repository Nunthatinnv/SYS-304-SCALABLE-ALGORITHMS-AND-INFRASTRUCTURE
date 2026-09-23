"""Runtime configuration for the House Price prediction service.

All settings are read from environment variables so the same image can run
locally, in docker-compose, and in CI without code changes.
"""

from __future__ import annotations

import os
from pathlib import Path

# Repository root, resolved relative to this file (backend/app/config.py).
_APP_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _APP_DIR.parent.parent

#: Path to the serialised scikit-learn pipeline produced in Milestone 1.
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(_REPO_ROOT / "models" / "xgb_baseline_pipeline.pkl")))

_MODELS_DIR = _REPO_ROOT / "models"

#: Inference backend: "sklearn" (Phase 2 naive), "onnx" (full pipeline in
#: ONNX) or "student" (distilled 10-feature student in ONNX, default).
MODEL_BACKEND = os.getenv("MODEL_BACKEND", "student").strip().lower()

#: Phase 3 artefacts produced by backend/scripts/export_onnx.py and
#: backend/scripts/distill_student.py.
ONNX_PATH = Path(os.getenv("ONNX_PATH", str(_MODELS_DIR / "xgb_pipeline.onnx")))
ONNX_META_PATH = Path(os.getenv("ONNX_META_PATH", str(_MODELS_DIR / "xgb_pipeline.onnx.json")))
STUDENT_PATH = Path(os.getenv("STUDENT_PATH", str(_MODELS_DIR / "student_xgb.onnx")))
STUDENT_META_PATH = Path(os.getenv("STUDENT_META_PATH", str(_MODELS_DIR / "student_meta.json")))

#: onnxruntime intra-op threads per worker process.
ORT_THREADS = int(os.getenv("ORT_THREADS", "1"))

#: Two-tier exact-match prediction cache: per-worker in-process LRU (L1,
#: CACHE_L1_SIZE entries, 0 disables) in front of Redis (L2, empty REDIS_URL
#: disables).
REDIS_URL = os.getenv("REDIS_URL", "").strip()
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "86400"))
CACHE_L1_SIZE = int(os.getenv("CACHE_L1_SIZE", "10000"))

#: Dynamic batching: group concurrent /predict calls into one model call.
#: BATCH_MAX_WAIT_MS=0 is "greedy" batching: never wait for company, just
#: take everything that queued up while the model thread was busy.
BATCHING_ENABLED = os.getenv("BATCHING_ENABLED", "true").strip().lower() in {"1", "true", "yes"}
BATCH_MAX_SIZE = int(os.getenv("BATCH_MAX_SIZE", "64"))
BATCH_MAX_WAIT_MS = float(os.getenv("BATCH_MAX_WAIT_MS", "0"))

#: Path to the reference metadata (column order, defaults, categories, ranges).
DEFAULTS_PATH = Path(os.getenv("DEFAULTS_PATH", str(_APP_DIR / "defaults.json")))

#: Comma-separated list of allowed CORS origins ("*" allows any origin).
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]

#: Service metadata surfaced on /health.
SERVICE_NAME = "house-price-api"
SERVICE_VERSION = "0.3.0"

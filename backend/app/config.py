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

#: Path to the reference metadata (column order, defaults, categories, ranges).
DEFAULTS_PATH = Path(os.getenv("DEFAULTS_PATH", str(_APP_DIR / "defaults.json")))

#: Comma-separated list of allowed CORS origins ("*" allows any origin).
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]

#: Service metadata surfaced on /health.
SERVICE_NAME = "house-price-api"
SERVICE_VERSION = "0.2.0"

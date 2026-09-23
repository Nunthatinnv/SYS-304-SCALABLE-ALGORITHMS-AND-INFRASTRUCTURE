# Build context is the repository root, because the image needs both
# backend/ and models/.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/models/xgb_baseline_pipeline.pkl \
    MODEL_BACKEND=student \
    ORT_THREADS=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Dependencies first so the layer is cached across code changes.
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r backend/requirements.txt

COPY backend/app backend/app
COPY backend/gunicorn.conf.py backend/gunicorn.conf.py
# Phase 2 pickle (MODEL_BACKEND=sklearn baseline) + Phase 3 ONNX artefacts.
COPY models/xgb_baseline_pipeline.pkl models/xgb_baseline_pipeline.pkl
COPY models/xgb_pipeline.onnx models/xgb_pipeline.onnx.json models/
COPY models/student_xgb.onnx models/student_meta.json models/

RUN useradd --create-home --uid 10001 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --start-period=40s --retries=5 \
    CMD curl -fsS http://localhost:8000/health || exit 1

# Phase 3: Gunicorn process manager with WEB_CONCURRENCY Uvicorn workers.
CMD ["gunicorn", "backend.app.main:app", "-c", "backend/gunicorn.conf.py"]

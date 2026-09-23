"""Gunicorn settings for the House Price API (Phase 3: concurrency & workers).

One Uvicorn worker *process* per CPU core. Inference is CPU-bound and holds the
GIL, so extra threads in one process cannot run the model in parallel; extra
processes can. Each worker loads its own copy of the (small, 150 KB) ONNX
model and runs its own async event loop + micro-batcher.
"""

import multiprocessing
import os

bind = f"0.0.0.0:{os.getenv('PORT', '8000')}"
worker_class = "uvicorn_worker.UvicornWorker"
workers = int(os.getenv("WEB_CONCURRENCY") or multiprocessing.cpu_count())
# ONNX Runtime sessions are not fork-safe, so the app (and the model) is
# loaded inside each worker, not preloaded in the master.
preload_app = False
timeout = int(os.getenv("WORKER_TIMEOUT", "30"))
graceful_timeout = 20
keepalive = 5
# Optional worker recycling to cap slow memory growth. Off by default: a
# recycled worker drops its open keep-alive connections, which showed up as
# client errors in the load test when only one worker was running.
max_requests = int(os.getenv("MAX_REQUESTS", "0"))
max_requests_jitter = max_requests // 10
accesslog = os.getenv("ACCESS_LOG") or None
errorlog = "-"
loglevel = os.getenv("LOG_LEVEL", "info")

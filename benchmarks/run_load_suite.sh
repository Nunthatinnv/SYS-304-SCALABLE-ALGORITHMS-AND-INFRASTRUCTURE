#!/usr/bin/env bash
#
# run_load_suite.sh — reproduce the Phase 2 vs Phase 3 load tests with Docker.
#
#   ./benchmarks/run_load_suite.sh            # all configurations
#   DUR=5 ./benchmarks/run_load_suite.sh      # shorter runs
#
# Every configuration is the same image with different environment knobs, so
# each optimisation can be switched on one at a time (ablation):
#
#   phase2                 Milestone 2 commit: 1 uvicorn worker, sklearn pickle
#   p3_model_only          student ONNX, 1 worker, no batching, no cache  (Week 5 only)
#   p3_infra_naive_model   sklearn pickle, N workers + batching           (Week 6 only)
#   p3_workers             student ONNX, N workers
#   p3_workers_batching    student ONNX, N workers + dynamic batching
#   p3_redis_only          ... + Redis cache only (no in-process L1)
#   p3_full                student ONNX, N workers + batching + L1 + Redis  (Phase 3)
#
# Requires: docker compose v2, python with requirements-dev.txt installed.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PHASE2_REF="${PHASE2_REF:-2129b1f}"   # "Milestone 2" commit
DUR="${DUR:-8}"
LEVELS="${LEVELS:-1 8 32 64}"
URL="http://localhost:${BACKEND_PORT:-8000}"

bench() { # label modes containers...
  local label=$1 modes=$2; shift 2
  python benchmarks/load_test.py --url "$URL" --label "$label" --concurrency $LEVELS \
    --mode $modes --duration "$DUR" --warmup 2 --containers "$@"
  curl -fsS "$URL/stats" > "benchmarks/results/stats_${label}.json" 2>/dev/null || true
}

up() { # env assignments...
  docker compose down --remove-orphans >/dev/null 2>&1 || true
  env "$@" docker compose up -d --build --wait backend redis
  docker compose exec -T redis redis-cli flushall >/dev/null
}

# --- Phase 2 baseline, built from the Milestone 2 commit --------------------
worktree="$(mktemp -d)/phase2"
git worktree add --detach "$worktree" "$PHASE2_REF" >/dev/null
docker compose down --remove-orphans >/dev/null 2>&1 || true
(cd "$worktree" && docker compose -p sys304-phase2 up -d --build --wait backend)
bench phase2 "cold warm" sys304-backend
(cd "$worktree" && docker compose -p sys304-phase2 down --remove-orphans)
git worktree remove --force "$worktree"

# --- Phase 3 ablation ---------------------------------------------------------
NOCACHE=(CACHE_L1_SIZE=0 REDIS_URL=)
up MODEL_BACKEND=student WEB_CONCURRENCY=1 BATCHING_ENABLED=false "${NOCACHE[@]}"
bench p3_model_only cold sys304-backend

up MODEL_BACKEND=sklearn BATCHING_ENABLED=true "${NOCACHE[@]}"
bench p3_infra_naive_model cold sys304-backend

up MODEL_BACKEND=student BATCHING_ENABLED=false "${NOCACHE[@]}"
bench p3_workers cold sys304-backend

up MODEL_BACKEND=student BATCHING_ENABLED=true "${NOCACHE[@]}"
bench p3_workers_batching "cold warm" sys304-backend

up MODEL_BACKEND=student BATCHING_ENABLED=true CACHE_L1_SIZE=0
bench p3_redis_only warm sys304-backend sys304-redis

up MODEL_BACKEND=student BATCHING_ENABLED=true
bench p3_full "cold warm" sys304-backend sys304-redis

docker compose down --remove-orphans
echo "Done. Results in benchmarks/results/; render charts with: python benchmarks/make_report.py"

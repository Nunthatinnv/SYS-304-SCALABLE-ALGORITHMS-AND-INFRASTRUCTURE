#!/usr/bin/env bash
#
# deploy.sh — launch (or tear down) the full House Price stack.
#
#   ./deploy.sh            build images, start backend + frontend, wait for health
#   ./deploy.sh --down     stop and remove the stack
#   ./deploy.sh --logs     follow backend logs
#
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-8080}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-180}"

log() { printf '\033[1;32m==>\033[0m %s\n' "$*"; }
err() { printf '\033[1;31m!!\033[0m %s\n' "$*" >&2; }

require_docker() {
  if ! command -v docker >/dev/null 2>&1; then
    err "docker is not installed or not on PATH."
    exit 1
  fi
  if ! docker compose version >/dev/null 2>&1; then
    err "'docker compose' is unavailable. Install Docker Compose v2."
    exit 1
  fi
  if ! docker info >/dev/null 2>&1; then
    err "The Docker daemon is not running. Start Docker Desktop and retry."
    exit 1
  fi
}

wait_for_health() {
  local url="http://localhost:${BACKEND_PORT}/health"
  local waited=0
  log "Waiting for the API at ${url} (timeout ${HEALTH_TIMEOUT}s)"
  until curl -fsS "$url" >/dev/null 2>&1; do
    if [ "$waited" -ge "$HEALTH_TIMEOUT" ]; then
      err "Backend did not become healthy in ${HEALTH_TIMEOUT}s."
      docker compose logs --tail 50 backend
      exit 1
    fi
    sleep 3
    waited=$((waited + 3))
  done
  log "API is healthy."
}

case "${1:-up}" in
  --down|down)
    require_docker
    log "Stopping the stack"
    docker compose down --remove-orphans
    log "Stopped."
    ;;
  --logs|logs)
    require_docker
    docker compose logs -f backend
    ;;
  up|--up)
    require_docker
    log "Building images"
    docker compose build
    log "Starting containers"
    BACKEND_PORT="$BACKEND_PORT" FRONTEND_PORT="$FRONTEND_PORT" docker compose up -d
    wait_for_health
    echo
    log "Stack is up."
    echo "    Frontend UI : http://localhost:${FRONTEND_PORT}"
    echo "    API docs    : http://localhost:${BACKEND_PORT}/docs"
    echo "    Health      : http://localhost:${BACKEND_PORT}/health"
    echo
    echo "    Follow logs : ./deploy.sh --logs"
    echo "    Tear down   : ./deploy.sh --down"
    ;;
  *)
    err "Unknown option: $1"
    echo "Usage: ./deploy.sh [up|--down|--logs]" >&2
    exit 2
    ;;
esac

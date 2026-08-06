#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
HOST=${DEMO_HOST:-0.0.0.0}
PORT=${DEMO_PORT:-8500}

cd "${REPO_ROOT}/demo"
exec uvicorn app:app --host "${HOST}" --port "${PORT}"

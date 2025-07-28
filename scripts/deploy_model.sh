#!/usr/bin/env bash
set -euo pipefail

echo "🔧  Preparing MLflow tracking server ..."

# Kill anything already on 5000
pkill -f "mlflow.server" 2>/dev/null || true

# Launch fresh server
mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlartifacts \
  --workers 1 &
MLFLOW_PID=$!

# Simple health-check loop
for i in {1..15}; do
  if curl -fs http://127.0.0.1:5000/health >/dev/null; then
    echo "✅  MLflow is up (PID $MLFLOW_PID)"
    exit 0
  fi
  sleep 2
done

echo "❌  MLflow failed to start" >&2
exit 1
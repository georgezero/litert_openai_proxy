#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MODEL_ID="${MODEL_ID:-gemma-4-e2b-it}"
LITERT_BACKEND="${LITERT_BACKEND:-cpu}"
VENV_PATH="${VENV_PATH:-$HOME/.venvs/litert-openai-proxy}"
APP_PATH="${APP_PATH:-$SCRIPT_DIR/litert_openai_proxy.py}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found in PATH" >&2
  exit 1
fi

MODEL_FILE="$HOME/.litert-lm/models/$MODEL_ID/model.litertlm"
if [ ! -f "$MODEL_FILE" ]; then
  echo "Model '$MODEL_ID' not found at $MODEL_FILE" >&2
  echo "Import it first:" >&2
  echo "  litert-lm import --from-huggingface-repo litert-community/gemma-4-E2B-it-litert-lm gemma-4-E2B-it.litertlm $MODEL_ID" >&2
  exit 1
fi

uv venv --allow-existing "$VENV_PATH" >/dev/null
# litert-lm-api provides the in-process Python inference API (real streaming)
uv pip install --python "$VENV_PATH/bin/python" --upgrade \
  fastapi uvicorn pydantic litert-lm-api >/dev/null

export LITERT_MODEL_ID="$MODEL_ID"
export LITERT_BACKEND="$LITERT_BACKEND"

cat <<MSG
Starting LiteRT native proxy (in-process model, real streaming)
  host:    $HOST
  port:    $PORT
  model:   $MODEL_ID
  backend: $LITERT_BACKEND

Optional auth:
  export OPENAI_API_KEY='your-secret'

Test (non-streaming):
  curl -s http://127.0.0.1:$PORT/v1/chat/completions \\
    -H 'Content-Type: application/json' \\
    -d '{"model":"$MODEL_ID","messages":[{"role":"user","content":"hello"}]}'

Test (streaming):
  curl -sN http://127.0.0.1:$PORT/v1/chat/completions \\
    -H 'Content-Type: application/json' \\
    -d '{"model":"$MODEL_ID","stream":true,"messages":[{"role":"user","content":"hello"}]}'
MSG

exec "$VENV_PATH/bin/uvicorn" --host "$HOST" --port "$PORT" litert_openai_proxy:app --app-dir "$(dirname "$APP_PATH")"

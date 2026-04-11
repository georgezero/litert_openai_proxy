#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MODEL_ID="${MODEL_ID:-gemma-4-e2b-it}"
VENV_PATH="${VENV_PATH:-$HOME/.venvs/litert-openai-proxy}"
APP_PATH="${APP_PATH:-$HOME/litert_openai_proxy.py}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found in PATH" >&2
  exit 1
fi
if ! command -v litert-lm >/dev/null 2>&1; then
  echo "litert-lm not found in PATH (install with: uv tool install litert-lm)" >&2
  exit 1
fi

if ! litert-lm list | awk 'NR>2 {print $1}' | grep -Fxq "$MODEL_ID"; then
  echo "Model '$MODEL_ID' is not imported. Import it first, for example:" >&2
  echo "  litert-lm import --from-huggingface-repo litert-community/gemma-4-E2B-it-litert-lm gemma-4-E2B-it.litertlm $MODEL_ID" >&2
  exit 1
fi

uv venv --allow-existing "$VENV_PATH" >/dev/null
uv pip install --python "$VENV_PATH/bin/python" --upgrade fastapi uvicorn pydantic >/dev/null

export LITERT_MODEL_ID="$MODEL_ID"

cat <<MSG
Starting LiteRT OpenAI proxy
  host:  $HOST
  port:  $PORT
  model: $MODEL_ID

Optional auth:
  export OPENAI_API_KEY='your-secret'

Test:
  curl -s http://127.0.0.1:$PORT/v1/chat/completions \\
    -H 'Content-Type: application/json' \\
    -d '{"model":"$MODEL_ID","messages":[{"role":"user","content":"hello world"}]}'
MSG

exec "$VENV_PATH/bin/uvicorn" --host "$HOST" --port "$PORT" litert_openai_proxy:app --app-dir "$(dirname "$APP_PATH")"

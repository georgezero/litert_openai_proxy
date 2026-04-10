#!/usr/bin/env bash
# install.sh — one-shot setup for litert-lm OpenAI proxy
#
# Run this from the checked-out repo directory:
#   cd ~/Downloads/src/llm/litert_openai_proxy
#   ./install.sh
#
# What it does:
#   1. Checks prerequisites (uv)
#   2. Installs litert-lm CLI via uv tool
#   3. Downloads and imports the Gemma 4 E2B IT model (~2.4 GB)
#   4. Creates the proxy venv and installs Python dependencies
#   5. Writes a systemd user service pointing at this repo directory
#   6. Enables and starts the service

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_REPO="litert-community/gemma-4-E2B-it-litert-lm"
MODEL_FILE="gemma-4-E2B-it.litertlm"
MODEL_ID="${MODEL_ID:-gemma-4-e2b-it}"
VENV_PATH="${VENV_PATH:-$HOME/.venvs/litert-openai-proxy}"
LOG_FILE="${LOG_FILE:-$HOME/.local/state/litert-proxy/proxy.log}"
SERVICE_NAME="litert-openai-proxy"
SERVICE_DIR="$HOME/.config/systemd/user"

log()  { printf '\033[1;32m[install]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[install]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[install]\033[0m %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 1. Prerequisites
# ---------------------------------------------------------------------------

log "Checking prerequisites..."
command -v uv >/dev/null 2>&1 || die "uv is required. Install from https://docs.astral.sh/uv/getting-started/installation/"

# ---------------------------------------------------------------------------
# 2. Install litert-lm CLI
# ---------------------------------------------------------------------------

log "Installing litert-lm CLI via uv tool..."
uv tool install --force litert-lm
command -v litert-lm >/dev/null 2>&1 || die "litert-lm not found in PATH after install. Try: source ~/.bashrc or add ~/.local/bin to PATH"
log "litert-lm version: $(litert-lm --version)"

# ---------------------------------------------------------------------------
# 3. Download and import model
# ---------------------------------------------------------------------------

MODEL_PATH="$HOME/.litert-lm/models/$MODEL_ID/model.litertlm"
if [ -f "$MODEL_PATH" ]; then
  warn "Model '$MODEL_ID' already exists at $MODEL_PATH — skipping import."
  warn "Re-import with: litert-lm import --from-huggingface-repo $MODEL_REPO $MODEL_FILE $MODEL_ID"
else
  log "Importing $MODEL_ID from $MODEL_REPO (~2.4 GB download)..."
  litert-lm import --from-huggingface-repo "$MODEL_REPO" "$MODEL_FILE" "$MODEL_ID"
  log "Model imported to $MODEL_PATH"
fi

# ---------------------------------------------------------------------------
# 4. Create proxy venv and install dependencies
# ---------------------------------------------------------------------------

log "Creating Python venv at $VENV_PATH..."
uv venv --allow-existing "$VENV_PATH"
log "Installing proxy dependencies (fastapi, uvicorn, pydantic, litert-lm-api)..."
uv pip install --python "$VENV_PATH/bin/python" --upgrade \
  fastapi uvicorn pydantic litert-lm-api

# ---------------------------------------------------------------------------
# 5. Write systemd user service
# ---------------------------------------------------------------------------

mkdir -p "$SERVICE_DIR"

log "Writing $SERVICE_DIR/$SERVICE_NAME.service..."
cat > "$SERVICE_DIR/$SERVICE_NAME.service" <<EOF
[Unit]
Description=LiteRT OpenAI-compatible Proxy
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
Environment=HOME=$HOME
Environment=PATH=$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin
Environment=OPENAI_API_KEY=
Environment=MODEL_ID=$MODEL_ID
Environment=HOST=0.0.0.0
Environment=PORT=8000
Environment=LITERT_BACKEND=cpu
Environment=VENV_PATH=$VENV_PATH
Environment=LOG_FILE=$LOG_FILE
StandardOutput=append:$LOG_FILE
StandardError=append:$LOG_FILE
WorkingDirectory=$REPO_DIR
ExecStart=$REPO_DIR/run-litert-openai-proxy.sh
Restart=always
RestartSec=2

[Install]
WantedBy=default.target
EOF

# ---------------------------------------------------------------------------
# 6. Enable and start service
# ---------------------------------------------------------------------------

log "Enabling and starting $SERVICE_NAME..."
systemctl --user daemon-reload
systemctl --user enable --now "$SERVICE_NAME"

sleep 2
if systemctl --user is-active --quiet "$SERVICE_NAME"; then
  log "Service is running."
else
  warn "Service may have failed to start. Check with:"
  warn "  systemctl --user status $SERVICE_NAME"
  warn "  journalctl --user -u $SERVICE_NAME -n 50"
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------

cat <<MSG

\033[1;32mInstallation complete!\033[0m

Proxy endpoint:  http://0.0.0.0:8000/v1
Health check:    python3 -c "import urllib.request; print(urllib.request.urlopen('http://localhost:8000/healthz').read().decode())"
Logs (live):     tail -f $LOG_FILE

Manage the service:
  systemctl --user status $SERVICE_NAME
  systemctl --user restart $SERVICE_NAME
  systemctl --user stop $SERVICE_NAME

To enable auth, edit the service and set OPENAI_API_KEY:
  systemctl --user edit $SERVICE_NAME
MSG

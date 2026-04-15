#!/usr/bin/env bash
# install-proxy-and-cloudflare-tunnel.sh
# End-to-end installer for LiteRT OpenAI proxy + Cloudflare tunnel.
#
# What it does:
#   1. Installs litert-lm + model
#   2. Installs proxy Python deps
#   3. Creates/updates systemd user service: litert-openai-proxy
#   4. Installs cloudflared if missing (APT repo fallback trixie->bookworm)
#   5. Creates/updates Cloudflare tunnel + DNS route
#   6. Creates/updates systemd user service for the tunnel
#
# Optional env vars:
#   MODEL_ID=gemma-4-e2b-it
#   VENV_PATH=$HOME/.venvs/litert-openai-proxy
#   LOG_FILE=$HOME/.local/state/litert-proxy/proxy.log
#   HOST=0.0.0.0
#   PORT=8000
#   LITERT_BACKEND=cpu
#   LITERT_MAX_NUM_TOKENS=16384
#   OPENAI_API_KEY=
#   CF_TUNNEL_NAME=pecan-litert-openai
#   CF_HOSTNAME=pecan.ggg.ad
#   CF_CONFIG_PATH=$HOME/.cloudflared/pecan-litert-openai.yml
#   SKIP_DNS=false
#   DRY_RUN=false

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_REPO="litert-community/gemma-4-E2B-it-litert-lm"
MODEL_FILE="gemma-4-E2B-it.litertlm"
MODEL_ID="${MODEL_ID:-gemma-4-e2b-it}"
VENV_PATH="${VENV_PATH:-$HOME/.venvs/litert-openai-proxy}"
LOG_FILE="${LOG_FILE:-$HOME/.local/state/litert-proxy/proxy.log}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
LITERT_BACKEND="${LITERT_BACKEND:-cpu}"
LITERT_MAX_NUM_TOKENS="${LITERT_MAX_NUM_TOKENS:-16384}"
OPENAI_API_KEY="${OPENAI_API_KEY:-}"

PROXY_SERVICE_NAME="litert-openai-proxy"
SERVICE_DIR="$HOME/.config/systemd/user"

CF_TUNNEL_NAME="${CF_TUNNEL_NAME:-pecan-litert-openai}"
CF_HOSTNAME="${CF_HOSTNAME:-pecan.ggg.ad}"
CF_CONFIG_PATH="${CF_CONFIG_PATH:-$HOME/.cloudflared/${CF_TUNNEL_NAME}.yml}"
CF_CREDENTIALS_DIR="$HOME/.cloudflared"
CF_SERVICE_NAME="cloudflared-${CF_TUNNEL_NAME}"
SKIP_DNS="${SKIP_DNS:-false}"
DRY_RUN="${DRY_RUN:-false}"

log()  { printf '\033[1;32m[install]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[install]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[install]\033[0m %s\n' "$*" >&2; exit 1; }

run_cmd() {
  if [ "$DRY_RUN" = "true" ]; then
    printf '[dry-run] %s\n' "$*"
  else
    "$@"
  fi
}

run_sh() {
  if [ "$DRY_RUN" = "true" ]; then
    printf '[dry-run] %s\n' "$*"
  else
    bash -lc "$*"
  fi
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Required command missing: $1"
}

ensure_cloudflared_installed() {
  if command -v cloudflared >/dev/null 2>&1; then
    log "cloudflared already installed: $(cloudflared --version | head -n1)"
    return
  fi

  log "cloudflared not found; installing via APT..."
  require_cmd sudo
  require_cmd curl

  local codename repo_suite
  codename="$(. /etc/os-release && echo "$VERSION_CODENAME")"
  repo_suite="$codename"
  if [ "$codename" = "trixie" ]; then
    # Cloudflare repo does not currently publish trixie; use bookworm.
    repo_suite="bookworm"
  fi

  run_sh "sudo install -m 0755 -d /usr/share/keyrings"
  run_sh "curl -fsSL https://pkg.cloudflare.com/cloudflare-main.gpg | sudo tee /usr/share/keyrings/cloudflare-archive-keyring.gpg >/dev/null"
  run_sh "echo 'deb [signed-by=/usr/share/keyrings/cloudflare-archive-keyring.gpg] https://pkg.cloudflare.com/cloudflared ${repo_suite} main' | sudo tee /etc/apt/sources.list.d/cloudflared.list >/dev/null"
  run_sh "sudo apt-get update"
  run_sh "sudo apt-get install -y cloudflared"

  command -v cloudflared >/dev/null 2>&1 || die "cloudflared install failed"
  log "Installed cloudflared: $(cloudflared --version | head -n1)"
}

install_proxy() {
  log "Checking prerequisites..."
  require_cmd uv

  log "Installing litert-lm CLI via uv tool..."
  run_cmd uv tool install --force litert-lm
  command -v litert-lm >/dev/null 2>&1 || die "litert-lm not found in PATH after install"
  log "litert-lm version: $(litert-lm --version)"

  local model_path
  model_path="$HOME/.litert-lm/models/$MODEL_ID/model.litertlm"
  if [ -f "$model_path" ]; then
    warn "Model '$MODEL_ID' already exists at $model_path — skipping import."
  else
    log "Importing $MODEL_ID from $MODEL_REPO (~2.4 GB download)..."
    run_cmd litert-lm import --from-huggingface-repo "$MODEL_REPO" "$MODEL_FILE" "$MODEL_ID"
    log "Model imported to $model_path"
  fi

  log "Creating Python venv at $VENV_PATH..."
  run_cmd uv venv --allow-existing "$VENV_PATH"
  log "Installing proxy dependencies..."
  run_cmd uv pip install --python "$VENV_PATH/bin/python" --upgrade fastapi uvicorn pydantic litert-lm-api

  run_cmd mkdir -p "$(dirname "$LOG_FILE")" "$SERVICE_DIR"

  log "Writing $SERVICE_DIR/$PROXY_SERVICE_NAME.service..."
  if [ "$DRY_RUN" = "false" ]; then
    cat > "$SERVICE_DIR/$PROXY_SERVICE_NAME.service" <<UNIT
[Unit]
Description=LiteRT OpenAI-compatible Proxy
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
Environment=HOME=$HOME
Environment=PATH=$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin
Environment=OPENAI_API_KEY=$OPENAI_API_KEY
Environment=MODEL_ID=$MODEL_ID
Environment=HOST=$HOST
Environment=PORT=$PORT
Environment=LITERT_BACKEND=$LITERT_BACKEND
Environment=LITERT_MAX_NUM_TOKENS=$LITERT_MAX_NUM_TOKENS
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
UNIT
  else
    printf '[dry-run] write %s/%s.service\n' "$SERVICE_DIR" "$PROXY_SERVICE_NAME"
  fi

  log "Enabling and starting $PROXY_SERVICE_NAME..."
  run_cmd systemctl --user daemon-reload
  run_cmd systemctl --user enable --now "$PROXY_SERVICE_NAME"
}

get_tunnel_id() {
  local name="$1"
  cloudflared tunnel info "$name" 2>/dev/null | awk '/^ID:/ {print $2; exit}'
}

ensure_tunnel_and_dns() {
  ensure_cloudflared_installed

  run_cmd mkdir -p "$CF_CREDENTIALS_DIR" "$SERVICE_DIR"

  if [ ! -f "$CF_CREDENTIALS_DIR/cert.pem" ]; then
    warn "Cloudflare cert not found at $CF_CREDENTIALS_DIR/cert.pem"
    warn "Run this once, then rerun this script:"
    warn "  cloudflared tunnel login"
    die "Missing Cloudflare origin cert"
  fi

  local tunnel_id
  tunnel_id="$(get_tunnel_id "$CF_TUNNEL_NAME" || true)"
  if [ -n "${tunnel_id:-}" ]; then
    log "Tunnel already exists: $CF_TUNNEL_NAME ($tunnel_id)"
  else
    log "Creating tunnel: $CF_TUNNEL_NAME"
    if [ "$DRY_RUN" = "false" ]; then
      local create_out
      create_out="$(cloudflared tunnel create "$CF_TUNNEL_NAME")"
      printf '%s\n' "$create_out"
      tunnel_id="$(printf '%s\n' "$create_out" | awk '/Created tunnel/ {print $NF}')"
      [ -n "$tunnel_id" ] || tunnel_id="$(get_tunnel_id "$CF_TUNNEL_NAME")"
    else
      printf '[dry-run] cloudflared tunnel create %s\n' "$CF_TUNNEL_NAME"
      tunnel_id="DRY_RUN_TUNNEL_ID"
    fi
  fi

  [ -n "${tunnel_id:-}" ] || die "Could not determine tunnel ID for $CF_TUNNEL_NAME"

  local cred_file
  cred_file="$CF_CREDENTIALS_DIR/${tunnel_id}.json"
  if [ "$DRY_RUN" = "false" ] && [ ! -f "$cred_file" ]; then
    die "Tunnel credentials missing: $cred_file"
  fi

  if [ "$SKIP_DNS" != "true" ]; then
    log "Routing DNS $CF_HOSTNAME -> tunnel $CF_TUNNEL_NAME (overwrite enabled)..."
    run_cmd cloudflared tunnel route dns -f "$CF_TUNNEL_NAME" "$CF_HOSTNAME"
  else
    warn "SKIP_DNS=true, not changing DNS route"
  fi

  log "Writing Cloudflare tunnel config: $CF_CONFIG_PATH"
  if [ "$DRY_RUN" = "false" ]; then
    cat > "$CF_CONFIG_PATH" <<YAML
tunnel: $tunnel_id
credentials-file: $cred_file
ingress:
  - hostname: $CF_HOSTNAME
    service: http://127.0.0.1:$PORT
  - service: http_status:404
YAML
  else
    printf '[dry-run] write %s\n' "$CF_CONFIG_PATH"
  fi

  log "Writing $SERVICE_DIR/$CF_SERVICE_NAME.service..."
  if [ "$DRY_RUN" = "false" ]; then
    cat > "$SERVICE_DIR/$CF_SERVICE_NAME.service" <<UNIT
[Unit]
Description=Cloudflare Tunnel for LiteRT OpenAI proxy ($CF_HOSTNAME)
After=network-online.target $PROXY_SERVICE_NAME.service
Wants=network-online.target $PROXY_SERVICE_NAME.service

[Service]
Type=simple
ExecStart=$(command -v cloudflared) tunnel --config $CF_CONFIG_PATH run
Restart=always
RestartSec=2

[Install]
WantedBy=default.target
UNIT
  else
    printf '[dry-run] write %s/%s.service\n' "$SERVICE_DIR" "$CF_SERVICE_NAME"
  fi

  log "Enabling and starting $CF_SERVICE_NAME..."
  run_cmd systemctl --user daemon-reload
  run_cmd systemctl --user enable --now "$CF_SERVICE_NAME"
}

print_summary() {
  cat <<MSG

\033[1;32mInstallation complete!\033[0m

Proxy endpoint:  http://$HOST:$PORT/v1
Public endpoint: https://$CF_HOSTNAME/v1

Check status:
  systemctl --user status $PROXY_SERVICE_NAME
  systemctl --user status $CF_SERVICE_NAME

Health checks:
  python3 -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:$PORT/healthz').read().decode())"
  curl -s https://$CF_HOSTNAME/healthz

Logs:
  tail -f $LOG_FILE
  journalctl --user -u $CF_SERVICE_NAME -f
MSG
}

main() {
  install_proxy
  ensure_tunnel_and_dns
  print_summary
}

main "$@"

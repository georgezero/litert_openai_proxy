#!/usr/bin/env bash
set -euo pipefail

MODEL_REPO="litert-community/gemma-4-E2B-it-litert-lm"
MODEL_FILE="gemma-4-E2B-it.litertlm"
MODEL_ID="gemma-4-e2b-it"
INSTALL_ONLY=0
FORCE_MODEL=0

usage() {
  cat <<USAGE
Usage: $0 [--install-only] [--force-model] [--help]

Installs litert-lm using the official uv tool flow.
By default, also downloads/imports Gemma 4 E2B IT.

Options:
  --install-only  Install litert-lm only; skip model download/import.
  --force-model   Re-import model even if MODEL_ID already exists locally.
  -h, --help      Show this help.
USAGE
}

log() {
  printf '[litert-lm-setup] %s\n' "$*"
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --install-only)
      INSTALL_ONLY=1
      ;;
    --force-model)
      FORCE_MODEL=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required but not found in PATH" >&2
  exit 1
fi

log "Installing/upgrading litert-lm with uv tool"
uv tool install --force litert-lm

if ! command -v litert-lm >/dev/null 2>&1; then
  echo "litert-lm not found in PATH after install" >&2
  exit 1
fi

log "Installed: $(litert-lm --version)"

if [ "$INSTALL_ONLY" -eq 1 ]; then
  log "Install-only mode requested; skipping model download/import"
  exit 0
fi

if litert-lm list | awk 'NR>2 {print $1}' | grep -Fxq "$MODEL_ID"; then
  if [ "$FORCE_MODEL" -eq 1 ]; then
    log "Deleting existing model id: $MODEL_ID"
    litert-lm delete "$MODEL_ID"
  else
    log "Model already present ($MODEL_ID); skipping import"
    exit 0
  fi
fi

log "Importing model $MODEL_ID from $MODEL_REPO/$MODEL_FILE"
litert-lm import --from-huggingface-repo "$MODEL_REPO" "$MODEL_FILE" "$MODEL_ID"

log "Done. Quick test command:"
log "litert-lm run $MODEL_ID --prompt=\"What is the capital of France?\""

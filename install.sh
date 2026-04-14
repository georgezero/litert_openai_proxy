#!/usr/bin/env bash
# Backward-compatible wrapper.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/install-proxy-and-cloudflare-tunnel.sh" "$@"

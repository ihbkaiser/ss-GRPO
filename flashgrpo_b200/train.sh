#!/usr/bin/env bash
set -euo pipefail

# Backward-compatible default launcher. Prefer the explicit per-model scripts.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/train_qwen25_7b.sh" "$@"


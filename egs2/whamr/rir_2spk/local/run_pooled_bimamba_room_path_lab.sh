#!/usr/bin/env bash
# Scratch room/path training followed automatically by WHAMR-only evaluation.
set -euo pipefail
[[ $# == 2 ]] || { echo 'Usage: script GPU_INDEX RESULT_RECIPE' >&2; exit 2; }
recipe=$(cd "$(dirname "$0")/.." && pwd)
exec bash "${recipe}/local/run_pooled_bimamba_attention_lab.sh" room_path "$@"

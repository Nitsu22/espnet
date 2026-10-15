#!/usr/bin/env bash
# Native 8 kHz physical RIR pairs, with two-second FIRs and PIM references.
set -eo pipefail
recipe=$(cd "$(dirname "$0")" && pwd)
exec bash "${recipe}/run_ls_direct_reverb_eval.sh" --sample_rate 8000 "$@"

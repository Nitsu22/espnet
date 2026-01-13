#!/usr/bin/env bash
set -e

real_python="${PYTHON_REAL:-python3}"

if [ "$#" -ge 2 ] && [ "$1" = "-m" ]; then
  module="$2"
  shift 2
  case "$module" in
    espnet2.bin.enh_train)
      module="espnet2.bin.enh_train_log"
      ;;
    espnet2.bin.enh_tse_train)
      module="espnet2.bin.enh_tse_train_log"
      ;;
  esac
  exec "$real_python" -m "$module" "$@"
else
  exec "$real_python" "$@"
fi

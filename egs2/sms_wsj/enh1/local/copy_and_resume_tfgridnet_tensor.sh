#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
run_dir=exp_tfgridnet_sms_wsj_1ch_paper
mkdir -p "${run_dir}/local_copy"
trap 'code=$?; if [ "$code" -ne 0 ]; then echo "$(date -Is) FAILED exit=${code}" >> "${run_dir}/local_copy/copy.status"; fi' EXIT
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
python3 -B local/copy_sms_wsj_to_tensor.py > "${run_dir}/local_copy/copy.log" 2>&1
test -f /var/tmp/nitsu/tfgridnet_sms_wsj_1ch/COPY_VERIFIED.json

# Preserve the original shared-storage attempt, including its partial stats.
archive_dir="${run_dir}/shared_read_attempt_before_local_copy"
mkdir "${archive_dir}"
for item in enh_stats_8k run.log run.status smoke_gpu.log; do
    if [ -e "${run_dir}/${item}" ]; then
        mv "${run_dir}/${item}" "${archive_dir}/${item}"
    fi
done
tmux new-session -d -s tfgridnet_sms_wsj_1ch_paper \
  'TFGRIDNET_DUMP_DIR=/var/tmp/nitsu/tfgridnet_sms_wsj_1ch/dump bash local/launch_tfgridnet_sms_wsj_1ch_paper_tensor.sh'
echo "$(date -Is) copy verified; local-data training launcher started" >> "${run_dir}/local_copy/copy.status"

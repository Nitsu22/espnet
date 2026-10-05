#!/usr/bin/env bash
# Run on tensor inside tmux. Transfer only; never start training.
set -euo pipefail
cd "$(dirname "$0")/.."
source_dump=/net/ox/data5/roland/research/sms_wsj_dump_4mic
target_dump=/gs/bs/tga-shinoda/nitsu/data/sms_wsj_4mic_koudaisai/dump_4mic
transfer_dir=exp_tfgridnet_koudaisai_1ch/transfer
mkdir -p "${transfer_dir}"
trap 'code=$?; printf "%s exit=%s\n" "$(date -Is)" "$code" > "${transfer_dir}/status"' EXIT
printf '%s preparing\n' "$(date -Is)" > "${transfer_dir}/status"
python3 local/prepare_koudaisai_dump.py prepare --source "${source_dump}" \
    --target "${target_dump}" --work "${transfer_dir}"
# Check capacity on the destination, including 50 GiB reserve.
required_bytes=$(python3 -c 'import json; print(json.load(open("exp_tfgridnet_koudaisai_1ch/transfer/manifest.json"))["total_bytes"] + 50 * 1024**3)')
ssh tsubame "mkdir -p '${target_dump}'; python3 -c 'import shutil; assert shutil.disk_usage(\"${target_dump}\").free > ${required_bytes}, \"Insufficient space\"'"
printf '%s copying\n' "$(date -Is)" > "${transfer_dir}/status"
rsync -rt --partial --info=progress2 --files-from="${transfer_dir}/audio_files.txt" \
    "${source_dump}/" "tsubame:${target_dump}/"
rsync -rt "${transfer_dir}/metadata/" "tsubame:${target_dump}/"
rsync -t "${transfer_dir}/manifest.json" "tsubame:${target_dump}/manifest.json"
printf '%s checksum-audit\n' "$(date -Is)" > "${transfer_dir}/status"
rsync -rcn --out-format='%i %n' --files-from="${transfer_dir}/audio_files.txt" \
    "${source_dump}/" "tsubame:${target_dump}/" > "${transfer_dir}/checksum_diff.txt"
test ! -s "${transfer_dir}/checksum_diff.txt"
ssh tsubame "touch '${target_dump}/rsync_checksum_verified'; cd /gs/bs/tga-shinoda/nitsu/research/tf-locoformer/espnet/egs2/sms_wsj/enh1 && python3 local/prepare_koudaisai_dump.py verify --target '${target_dump}'"

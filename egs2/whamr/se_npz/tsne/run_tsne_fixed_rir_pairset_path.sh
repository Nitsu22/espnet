#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "$0")" && pwd)
se_npz_dir=$(cd "${script_dir}/.." && pwd)

exp_dir="${se_npz_dir}/exp_swap/enh_train_se_resnet_triplet_noise_noseg_valid_256"
out_dir="${exp_dir}/tsne_path_json_seed1234_pair20_rir8_1ch"

if command -v conda >/dev/null 2>&1; then
  runner=(conda run -n tf-locoformer python)
else
  runner=(python)
fi

"${runner[@]}" "${script_dir}/tsne_fixed_rir_pairset_path.py" \
  --exp-dir "${exp_dir}" \
  --checkpoint "valid.loss.best.pth" \
  --data-dir "${se_npz_dir}/data/cv_mix_both_reverb_min_8k" \
  --rir-path-config "${script_dir}/path.json" \
  --num-pairs 20 \
  --pair-selection random \
  --pair-seed 1234 \
  --fixed-rir-count 8 \
  --rir-selection-seed 1234 \
  --sample-rate 8000 \
  --mix-type both \
  --ref-condition reverb \
  --anchor-single-channel \
  --perplexity 5 \
  --learning-rate auto \
  --random-seed 0 \
  --device cpu \
  --png-output-dir "${script_dir}/output" \
  --output-dir "${out_dir}" \
  "$@"

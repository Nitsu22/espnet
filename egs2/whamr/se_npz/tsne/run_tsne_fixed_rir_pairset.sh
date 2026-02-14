#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "$0")" && pwd)
se_npz_dir=$(cd "${script_dir}/.." && pwd)

exp_dir="${se_npz_dir}/exp_swap/enh_train_se_resnet_triplet_noise_noseg_valid_256"
out_dir="${exp_dir}/tsne_rir_pairset_fixed2_first5_1utt"

if command -v conda >/dev/null 2>&1; then
  runner=(conda run -n tf-locoformer python)
else
  runner=(python)
fi

"${runner[@]}" "${script_dir}/tsne_fixed_rir_pairset.py" \
  --exp-dir "${exp_dir}" \
  --checkpoint "valid.loss.best.pth" \
  --data-dir "${se_npz_dir}/data/cv_mix_both_reverb_min_8k" \
  --num-pairs 5 \
  --pair-selection head \
  --fixed-rir-count 2 \
  --sample-rate 8000 \
  --mix-type both \
  --ref-condition reverb \
  --anchor-single-channel \
  --perplexity 5 \
  --learning-rate auto \
  --random-seed 0 \
  --device cpu \
  --output-dir "${out_dir}" \
  "$@"

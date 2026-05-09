#!/usr/bin/env bash

set -e
set -u
set -o pipefail

min_or_max=min
sample_rate=8k
seed=1234
epoch=0
ngpu=1
max_utts=0
skip_utts=0

passthrough_opts=()
while [ $# -gt 0 ]; do
    case "$1" in
        --seed)
            seed="$2"; shift 2 ;;
        --seed=*)
            seed="${1#*=}"; shift ;;
        --epoch)
            epoch="$2"; shift 2 ;;
        --epoch=*)
            epoch="${1#*=}"; shift ;;
        --ngpu)
            ngpu="$2"; shift 2 ;;
        --ngpu=*)
            ngpu="${1#*=}"; shift ;;
        --max_utts)
            max_utts="$2"; shift 2 ;;
        --max_utts=*)
            max_utts="${1#*=}"; shift ;;
        --skip_utts)
            skip_utts="$2"; shift 2 ;;
        --skip_utts=*)
            skip_utts="${1#*=}"; shift ;;
        *)
            passthrough_opts+=("$1"); shift ;;
    esac
done

. ./path.sh

exp_dir=exp/enh_train_enh_tflocoformer_nocashe_se_aeafusion_trainable_lr3_lr4_resnet_256_4gpu_small_film
out_dir="${exp_dir}/spatial_embedding_distance_tt_seed${seed}_epoch${epoch}"

python local/analyze_spatial_embedding_distances.py \
    --train_config "${exp_dir}/config.yaml" \
    --model_file "${exp_dir}/valid.loss.best.pth" \
    --npz_data_dir ../se_npz/data/tt_mix_both_reverb_${min_or_max}_${sample_rate} \
    --output_dir "${out_dir}" \
    --seed "${seed}" \
    --epoch "${epoch}" \
    --fs 8000 \
    --ngpu "${ngpu}" \
    --max_utts "${max_utts}" \
    --skip_utts "${skip_utts}" \
    "${passthrough_opts[@]}"

# Examples:
# bash run_spatial_embedding_distance_small.sh --max_utts 100
# bash run_spatial_embedding_distance_small.sh --ngpu 0 --max_utts 10

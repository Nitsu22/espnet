#!/usr/bin/env bash
set -euo pipefail

# Run only inside a matching allocation; this script does not select GPUs.
stage=6
stop_stage=6
ngpu=1
batch_size=4
valid_batch_size=1
accum_grad=1
nj=4
inference_nj=1
gpu_inference=true
enh_config=conf/tuning/train_enh_tflocoformer_s_nf_8k.yaml
enh_exp=exp/enh_train_tflocoformer_s_nf_8k_1gpu_batch4
inference_model=valid.loss.best.pth
enh_args=
# Preserve fixed batch sizes despite shape files describing full utterances.
# Current NF-WHAMR utterances are shorter than 200000 samples (25 seconds).
enh_speech_fold_length=2000
help_message="Usage: bash run.sh [options]
NF-WHAMR TF-Locoformer-S baseline; reuse existing dump and statistics.
  --stage 6 --stop_stage 6       Training (default)
  --ngpu 1 --batch_size 4 --accum_grad 1
  --stage 7 --stop_stage 7       GPU inference
  --stage 8 --stop_stage 8 --ngpu 0  CPU scoring
  --enh_exp PATH                New experiment or checkpoint-resume directory
Changing batch settings requires a separate --enh_exp. Stages 1-5 are disabled."

. utils/parse_options.sh
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-${PWD}/.cache/numba}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${PWD}/.cache/matplotlib}"
if (( stage < 6 || stop_stage > 8 || stage > stop_stage || ngpu < 0 || batch_size < 1 || valid_batch_size < 1 || accum_grad < 1 || nj < 1 || inference_nj < 1 || enh_speech_fold_length < 1 )); then
    echo "Use Stages 6-8 and positive batch/job/fold settings" >&2
    exit 2
fi
if (( stage == 6 && ngpu < 1 )); then
    echo "Stage 6 requires an allocated GPU" >&2
    exit 2
fi
if (( stage == 6 )) && [[ ${enh_exp} == exp/enh_train_tflocoformer_s_nf_8k_1gpu_batch4 ]] \
    && (( ngpu != 1 || batch_size != 4 || accum_grad != 1 )); then
    echo "Choose a separate --enh_exp when changing GPU/batch/accumulation settings" >&2
    exit 2
fi
if [[ ${gpu_inference} == true ]] && (( stage <= 7 && stop_stage >= 7 && ngpu < 1 )); then
    echo "GPU inference requires an allocated GPU" >&2
    exit 2
fi
if [[ ! -L dump_clean || ! -d dump_clean/raw ]]; then
    echo "Missing linked enh4_clean/dump_clean; do not regenerate the corpus" >&2
    exit 2
fi
if (( stage == 6 )); then
    for split in train valid; do
        for field in speech_mix speech_ref1 speech_ref2; do
            shape=../enh4_clean/exp/enh_stats_8k/${split}/${field}_shape
            if [[ ! -s ${shape} ]]; then
                echo "Missing existing statistics: ${shape}" >&2
                exit 2
            fi
            if [[ ${split} == train ]] && ! awk -v limit="$((enh_speech_fold_length * 100))" \
                '{split($2, shape, ","); if (shape[1] >= limit) exit 1}' "${shape}"; then
                echo "Increase --enh_speech_fold_length to keep the requested batch size" >&2
                exit 2
            fi
        done
    done
fi

bash ./enh.sh \
    --stage "${stage}" --stop_stage "${stop_stage}" \
    --skip_data_prep true --skip_packing true --skip_upload_hf true \
    --train_set tr_mix_clean_reverb_min_8k \
    --valid_set cv_mix_clean_reverb_min_8k \
    --test_sets tt_mix_clean_reverb_min_8k \
    --fs 8k --ngpu "${ngpu}" --nj "${nj}" --ref_num 2 \
    --enh_config "${enh_config}" --enh_exp "${enh_exp}" \
    --expdir ../enh4_clean/exp --dumpdir dump_clean \
    --enh_speech_fold_length "${enh_speech_fold_length}" \
    --use_dereverb_ref false --use_noise_ref false \
    --inference_model "${inference_model}" \
    --audio_format wav --gpu_inference "${gpu_inference}" \
    --inference_nj "${inference_nj}" \
    --enh_args "--batch_size ${batch_size} --valid_batch_size ${valid_batch_size} --accum_grad ${accum_grad} ${enh_args}"

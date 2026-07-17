#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${script_dir}"

min_or_max=min # "min" or "max". This is to determine how the mixtures are generated in local/data.sh.
sample_rate=8k
ctf_sample_rate=8000
stage=5
stop_stage=6
src_dumpdir=dump_rir_clean
ctf_dumpdir=dump_rir_clean
expdir=exp/rir_cmha_film_all_ctf_clean_1ch
prepare_ctf=true
ctf_device=cuda
gpu=0

run_args=()
while [ $# -gt 0 ]; do
    case "$1" in
        --prepare_ctf|--prepare-ctf)
            prepare_ctf=$2
            shift 2
            ;;
        --ctf_device|--ctf-device)
            ctf_device=$2
            shift 2
            ;;
        --ctf_sample_rate|--ctf-sample-rate)
            ctf_sample_rate=$2
            shift 2
            ;;
        --src_dumpdir|--src-dumpdir)
            src_dumpdir=$2
            shift 2
            ;;
        --ctf_dumpdir|--ctf-dumpdir)
            ctf_dumpdir=$2
            shift 2
            ;;
        --expdir)
            expdir=$2
            shift 2
            ;;
        --stage)
            stage=$2
            shift 2
            ;;
        --stop_stage|--stop-stage)
            stop_stage=$2
            shift 2
            ;;
        --gpu)
            gpu=$2
            shift 2
            ;;
        *)
            run_args+=("$1")
            shift
            ;;
    esac
done

train_set=tr_mix_clean_reverb_${min_or_max}_${sample_rate}
valid_set=cv_mix_clean_reverb_${min_or_max}_${sample_rate}
test_sets="tt_mix_clean_reverb_${min_or_max}_${sample_rate}"

if "${prepare_ctf}"; then
    CUDA_VISIBLE_DEVICES=${gpu} bash local/precompute_rec_rir_ctf.sh \
        --sets "${train_set} ${valid_set} ${test_sets}" \
        --src_dumpdir "${src_dumpdir}" \
        --ctf_dumpdir "${ctf_dumpdir}" \
        --device "${ctf_device}" \
        --sample_rate "${ctf_sample_rate}"
fi

CUDA_VISIBLE_DEVICES=${gpu} ./enh_rir.sh \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --fs ${sample_rate} \
    --ngpu 1 \
    --ref_num 2 \
    --local_data_opts "--sample_rate ${sample_rate} --sample_rates ${sample_rate} --min_or_max ${min_or_max}" \
    --enh_config ./conf/tuning/train_enh_tflocoformer_small_rir_cmha_film_all_ctf.yaml \
    --enh_args "--batch_size 16" \
    --expdir "${expdir}" \
    --enh_exp "${expdir}/enh_train_enh_tflocoformer_small_rir_cmha_film_all_ctf_clean_1ch" \
    --use_dereverb_ref false \
    --use_noise_ref false \
    --inference_model "valid.loss.best.pth" \
    --audio_format wav \
    --dumpdir "${ctf_dumpdir}" \
    --rir_condition_type rir_ctf \
    --skip_data_prep true \
    --stage "${stage}" \
    --stop_stage "${stop_stage}" \
    --gpu_inference true \
    --inference_nj 1 \
    "${run_args[@]}"

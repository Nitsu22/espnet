#!/usr/bin/env bash
# Train and evaluate with a fixed, within-split derangement of estimated CTFs.
set -e
set -u
set -o pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${script_dir}"

min_or_max=min
sample_rate=8k
ctf_sample_rate=8000
stage=5
stop_stage=8
src_dumpdir=dump_rir_clean
matched_ctf_dumpdir=dump_rir_clean
random_ctf_dumpdir=
expdir=
prepare_ctf=false
ctf_device=cuda
ctf_random_seed=0
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
        --matched_ctf_dumpdir|--matched-ctf-dumpdir)
            matched_ctf_dumpdir=$2
            shift 2
            ;;
        --random_ctf_dumpdir|--random-ctf-dumpdir)
            random_ctf_dumpdir=$2
            shift 2
            ;;
        --expdir)
            expdir=$2
            shift 2
            ;;
        --ctf_random_seed|--ctf-random-seed)
            ctf_random_seed=$2
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

if ! [[ "${ctf_random_seed}" =~ ^-?[0-9]+$ ]]; then
    echo "--ctf_random_seed must be an integer, but got: ${ctf_random_seed}" >&2
    exit 2
fi
if [ "${prepare_ctf}" != true ] && [ "${prepare_ctf}" != false ]; then
    echo "--prepare_ctf must be true or false, but got: ${prepare_ctf}" >&2
    exit 2
fi

if [ -z "${random_ctf_dumpdir}" ]; then
    random_ctf_dumpdir="dump_rir_clean_random_ctf_seed${ctf_random_seed}"
fi
if [ -z "${expdir}" ]; then
    expdir="exp/rir_cmha_film_all_ctf_random_clean_1ch_seed${ctf_random_seed}"
fi

train_set=tr_mix_clean_reverb_${min_or_max}_${sample_rate}
valid_set=cv_mix_clean_reverb_${min_or_max}_${sample_rate}
test_sets="tt_mix_clean_reverb_${min_or_max}_${sample_rate}"
local_data_opts="--sample_rate ${sample_rate} --sample_rates ${sample_rate} --min_or_max ${min_or_max}"
enh_exp="${expdir}/enh_train_enh_tflocoformer_small_rir_cmha_film_all_ctf_random_clean_1ch_seed${ctf_random_seed}"

if "${prepare_ctf}"; then
    CUDA_VISIBLE_DEVICES="${gpu}" bash local/precompute_rec_rir_ctf.sh \
        --sets "${train_set} ${valid_set} ${test_sets}" \
        --src_dumpdir "${src_dumpdir}" \
        --ctf_dumpdir "${matched_ctf_dumpdir}" \
        --device "${ctf_device}" \
        --sample_rate "${ctf_sample_rate}"
fi

bash local/prepare_random_rir_ctf_data.sh \
    --sets "${train_set} ${valid_set} ${test_sets}" \
    --src_dumpdir "${matched_ctf_dumpdir}" \
    --dst_dumpdir "${random_ctf_dumpdir}" \
    --seed "${ctf_random_seed}"

CUDA_VISIBLE_DEVICES="${gpu}" ./enh_rir.sh \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --fs "${sample_rate}" \
    --ngpu 1 \
    --ref_num 2 \
    --local_data_opts "${local_data_opts}" \
    --enh_config ./conf/tuning/train_enh_tflocoformer_small_rir_cmha_film_all_ctf.yaml \
    --enh_args "--batch_size 16" \
    --expdir "${expdir}" \
    --enh_exp "${enh_exp}" \
    --use_dereverb_ref false \
    --use_noise_ref false \
    --inference_model valid.loss.best.pth \
    --audio_format wav \
    --dumpdir "${random_ctf_dumpdir}" \
    --rir_condition_type rir_ctf \
    --skip_data_prep true \
    --stage "${stage}" \
    --stop_stage "${stop_stage}" \
    --gpu_inference true \
    --inference_nj 1 \
    "${run_args[@]}"

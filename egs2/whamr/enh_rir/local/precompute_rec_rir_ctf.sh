#!/usr/bin/env bash

set -e
set -u
set -o pipefail

sets="tr_mix_clean_reverb_min_8k cv_mix_clean_reverb_min_8k tt_mix_clean_reverb_min_8k"
src_dumpdir=dump_rir_clean
ctf_dumpdir=dump_rec_rir_ctf_clean
rir_exp=../rir/exp/rir_train_rec_rir_2spk_pit_clean_8192
train_config=
model_file=
device=cuda
sample_rate=8000

. utils/parse_options.sh
. ./path.sh

[ -z "${train_config}" ] && train_config="${rir_exp}/config.yaml"
[ -z "${model_file}" ] && model_file="${rir_exp}/valid.loss.best.pth"

for dset in ${sets}; do
    src_dir="${src_dumpdir}/raw/${dset}"
    dst_dir="${ctf_dumpdir}/raw/${dset}"
    if [ ! -f "${src_dir}/wav.scp" ]; then
        echo "Missing required file: ${src_dir}/wav.scp" >&2
        exit 1
    fi

    utils/copy_data_dir.sh "${src_dir}" "${dst_dir}"
    for fname in feats_type spk1.scp spk2.scp spk2utt utt2num_samples utt2spk; do
        if [ -f "${src_dir}/${fname}" ]; then
            cp "${src_dir}/${fname}" "${dst_dir}/${fname}"
        fi
    done

    python -m espnet2.bin.rec_rir_pit_ctf_inference \
        --train_config "${train_config}" \
        --model_file "${model_file}" \
        --wav_scp "${dst_dir}/wav.scp" \
        --output_dir "${dst_dir}" \
        --device "${device}" \
        --sample_rate "${sample_rate}"
done

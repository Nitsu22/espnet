#!/usr/bin/env bash

set -e
set -u
set -o pipefail

rir_config=conf/tuning/train_rec_rir_tflocoformer_2spk_pit_clean_8192.yaml
rir_tag=train_rec_rir_tflocoformer_2spk_pit_clean_8192
rir_exp=exp/rir_${rir_tag}
rir_args=

stage=1
stop_stage=6
skip_train=false
train_set=tr_mix_clean_reverb_min_8k
valid_set=cv_mix_clean_reverb_min_8k
test_sets=tt_mix_clean_reverb_min_8k
fs=8k
ngpu=1
num_nodes=1
nj=32
dumpdir=dump_rir_clean
audio_format=wav
speech_fold_length=32000
prepare_rec_rir_pit_dump=true
enh_rir_data_dir=../enh_rir/data
rir_stats_dir=

. utils/parse_options.sh
. ./path.sh
. ./cmd.sh

python=${python:-python3}

if "${prepare_rec_rir_pit_dump}"; then
  local/prepare_rec_rir_2spk_pit_dump.sh \
    --dumpdir "${dumpdir}" \
    --enh_rir_data_dir "${enh_rir_data_dir}" \
    --sets "${train_set} ${valid_set} ${test_sets}"
fi

data_feats=${dumpdir}/raw
[ -z "${rir_stats_dir}" ] && rir_stats_dir=exp/rir_stats_${fs}_rec_rir_pit

if ! "${skip_train}"; then
  if [ "${stage}" -le 5 ] && [ "${stop_stage}" -ge 5 ]; then
    train_dir="${data_feats}/${train_set}"
    valid_dir="${data_feats}/${valid_set}"
    logdir="${rir_stats_dir}/logdir"
    mkdir -p "${logdir}"

    train_speech_mix_scp=wav.scp
    valid_speech_mix_scp=wav.scp
    if [ -f "${train_dir}/speech_mix_pit.scp" ]; then
      train_speech_mix_scp=speech_mix_pit.scp
    fi
    if [ -f "${valid_dir}/speech_mix_pit.scp" ]; then
      valid_speech_mix_scp=speech_mix_pit.scp
    fi

    nj_stats=$(python3 -c "print(min(${nj}, $(wc -l < "${train_dir}/${train_speech_mix_scp}"), $(wc -l < "${valid_dir}/${valid_speech_mix_scp}")))")
    split_scps=
    for n in $(seq "${nj_stats}"); do
      split_scps+=" ${logdir}/train.${n}.scp"
    done
    utils/split_scp.pl "${train_dir}/${train_speech_mix_scp}" ${split_scps}
    split_scps=
    for n in $(seq "${nj_stats}"); do
      split_scps+=" ${logdir}/valid.${n}.scp"
    done
    utils/split_scp.pl "${valid_dir}/${valid_speech_mix_scp}" ${split_scps}

    train_data_param="--train_data_path_and_name_and_type ${train_dir}/${train_speech_mix_scp},speech_mix,sound "
    train_data_param+="--train_data_path_and_name_and_type ${train_dir}/speech_direct1.scp,speech_direct1,sound "
    train_data_param+="--train_data_path_and_name_and_type ${train_dir}/speech_direct2.scp,speech_direct2,sound "
    train_data_param+="--train_data_path_and_name_and_type ${train_dir}/speech_reverb1.scp,speech_reverb1,sound "
    train_data_param+="--train_data_path_and_name_and_type ${train_dir}/speech_reverb2.scp,speech_reverb2,sound "
    valid_data_param="--valid_data_path_and_name_and_type ${valid_dir}/${valid_speech_mix_scp},speech_mix,sound "
    valid_data_param+="--valid_data_path_and_name_and_type ${valid_dir}/speech_direct1.scp,speech_direct1,sound "
    valid_data_param+="--valid_data_path_and_name_and_type ${valid_dir}/speech_direct2.scp,speech_direct2,sound "
    valid_data_param+="--valid_data_path_and_name_and_type ${valid_dir}/speech_reverb1.scp,speech_reverb1,sound "
    valid_data_param+="--valid_data_path_and_name_and_type ${valid_dir}/speech_reverb2.scp,speech_reverb2,sound "

    mkdir -p "${rir_stats_dir}"
    ${train_cmd} JOB=1:"${nj_stats}" "${logdir}"/stats.JOB.log \
      ${python} -m espnet2.bin.rir_tflocoformer_ctf_train \
        --collect_stats true \
        ${train_data_param} \
        ${valid_data_param} \
        --train_shape_file "${logdir}/train.JOB.scp" \
        --valid_shape_file "${logdir}/valid.JOB.scp" \
        --output_dir "${logdir}/stats.JOB" \
        --config "${rir_config}" ${rir_args}

    aggregate_opts=
    for n in $(seq "${nj_stats}"); do
      aggregate_opts+="--input_dir ${logdir}/stats.${n} "
    done
    ${python} -m espnet2.bin.aggregate_stats_dirs ${aggregate_opts} \
      --skip_sum_stats --output_dir "${rir_stats_dir}"
  fi

  if [ "${stage}" -le 6 ] && [ "${stop_stage}" -ge 6 ]; then
    train_dir="${data_feats}/${train_set}"
    valid_dir="${data_feats}/${valid_set}"

    train_speech_mix_scp=wav.scp
    valid_speech_mix_scp=wav.scp
    if [ -f "${train_dir}/speech_mix_pit.scp" ]; then
      train_speech_mix_scp=speech_mix_pit.scp
    fi
    if [ -f "${valid_dir}/speech_mix_pit.scp" ]; then
      valid_speech_mix_scp=speech_mix_pit.scp
    fi

    train_data_param="--train_data_path_and_name_and_type ${train_dir}/${train_speech_mix_scp},speech_mix,sound "
    train_data_param+="--train_data_path_and_name_and_type ${train_dir}/speech_direct1.scp,speech_direct1,sound "
    train_data_param+="--train_data_path_and_name_and_type ${train_dir}/speech_direct2.scp,speech_direct2,sound "
    train_data_param+="--train_data_path_and_name_and_type ${train_dir}/speech_reverb1.scp,speech_reverb1,sound "
    train_data_param+="--train_data_path_and_name_and_type ${train_dir}/speech_reverb2.scp,speech_reverb2,sound "
    valid_data_param="--valid_data_path_and_name_and_type ${valid_dir}/${valid_speech_mix_scp},speech_mix,sound "
    valid_data_param+="--valid_data_path_and_name_and_type ${valid_dir}/speech_direct1.scp,speech_direct1,sound "
    valid_data_param+="--valid_data_path_and_name_and_type ${valid_dir}/speech_direct2.scp,speech_direct2,sound "
    valid_data_param+="--valid_data_path_and_name_and_type ${valid_dir}/speech_reverb1.scp,speech_reverb1,sound "
    valid_data_param+="--valid_data_path_and_name_and_type ${valid_dir}/speech_reverb2.scp,speech_reverb2,sound "

    train_shape_param="--train_shape_file ${rir_stats_dir}/train/speech_mix_shape "
    train_shape_param+="--train_shape_file ${rir_stats_dir}/train/speech_direct1_shape "
    train_shape_param+="--train_shape_file ${rir_stats_dir}/train/speech_direct2_shape "
    train_shape_param+="--train_shape_file ${rir_stats_dir}/train/speech_reverb1_shape "
    train_shape_param+="--train_shape_file ${rir_stats_dir}/train/speech_reverb2_shape "
    valid_shape_param="--valid_shape_file ${rir_stats_dir}/valid/speech_mix_shape "
    valid_shape_param+="--valid_shape_file ${rir_stats_dir}/valid/speech_direct1_shape "
    valid_shape_param+="--valid_shape_file ${rir_stats_dir}/valid/speech_direct2_shape "
    valid_shape_param+="--valid_shape_file ${rir_stats_dir}/valid/speech_reverb1_shape "
    valid_shape_param+="--valid_shape_file ${rir_stats_dir}/valid/speech_reverb2_shape "
    fold_length_param="--fold_length ${speech_fold_length} "
    fold_length_param+="--fold_length ${speech_fold_length} "
    fold_length_param+="--fold_length ${speech_fold_length} "
    fold_length_param+="--fold_length ${speech_fold_length} "
    fold_length_param+="--fold_length ${speech_fold_length} "

    mkdir -p "${rir_exp}"
    ${python} -m espnet2.bin.launch \
      --cmd "${cuda_cmd}" \
      --log "${rir_exp}/train.log" \
      --ngpu "${ngpu}" \
      --num_nodes "${num_nodes}" \
      --init_file_prefix "${rir_exp}/.dist_init_" \
      --multiprocessing_distributed true -- \
      ${python} -m espnet2.bin.rir_tflocoformer_ctf_train \
        ${train_data_param} \
        ${valid_data_param} \
        ${train_shape_param} \
        ${valid_shape_param} \
        ${fold_length_param} \
        --resume true \
        --output_dir "${rir_exp}" \
        --config "${rir_config}" ${rir_args}
  fi
fi

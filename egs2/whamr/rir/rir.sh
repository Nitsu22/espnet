#!/usr/bin/env bash

set -e
set -u
set -o pipefail

stage=1
stop_stage=6
skip_data_prep=false
skip_train=false

train_set=
valid_set=
test_sets=

rir_config=conf/tuning/train_rir_tflocoformer.yaml
rir_args=
rir_tag=
rir_exp=
rir_model_type=

fs=8k
ngpu=1
num_nodes=1
nj=32
inference_nj=32

expdir=exp
dumpdir=dump
audio_format=wav
local_data_opts=

min_wav_duration=0.1
max_wav_duration=60
speech_fold_length=32000
rir_fold_length=32000

. utils/parse_options.sh
. ./path.sh
. ./cmd.sh

python=${python:-python3}

[ -z "${train_set}" ] && { echo "--train_set is required" >&2; exit 2; }
[ -z "${valid_set}" ] && { echo "--valid_set is required" >&2; exit 2; }
[ -z "${test_sets}" ] && { echo "--test_sets is required" >&2; exit 2; }

data_feats=${dumpdir}/raw
rir_stats_tag="${fs}"
if [ -n "${rir_model_type}" ] && [ "${rir_model_type}" != direct ]; then
  rir_stats_tag="${fs}_${rir_model_type}"
fi
rir_stats_dir="${expdir}/rir_stats_${rir_stats_tag}"

if [ -z "${rir_tag}" ]; then
  if [ -n "${rir_config}" ]; then
    rir_tag="$(basename "${rir_config}" .yaml)"
  else
    rir_tag="train"
  fi
fi

if [ -z "${rir_exp}" ]; then
  rir_exp="${expdir}/rir_${rir_tag}"
fi

if ! "${skip_data_prep}"; then
  if [ "${stage}" -le 1 ] && [ "${stop_stage}" -ge 1 ]; then
    echo "Stage 1: RIR data preparation"
    local/prepare_rir_data.sh ${local_data_opts}
  fi

  if [ "${stage}" -le 3 ] && [ "${stop_stage}" -ge 3 ]; then
    echo "Stage 3: Format wav.scp"
    for dset in "${train_set}" "${valid_set}" ${test_sets}; do
      if [ "${dset}" = "${train_set}" ] || [ "${dset}" = "${valid_set}" ]; then
        suf=/org
      else
        suf=
      fi
      utils/copy_data_dir.sh "data/${dset}" "${data_feats}${suf}/${dset}"
      rm -f "${data_feats}${suf}/${dset}"/{segments,wav.scp,reco2file_and_channel}

      scripts/audio/format_wav_scp.sh --nj "${nj}" --cmd "${train_cmd}" \
        --out-filename wav.scp \
        --audio-format "${audio_format}" --fs "${fs}" \
        "data/${dset}/wav.scp" "${data_feats}${suf}/${dset}" \
        "${data_feats}${suf}/${dset}/logs/wav" \
        "${data_feats}${suf}/${dset}/data/wav"

      utt_extra_files="rir_npz.scp room_param_npz.scp"
      if [ "${rir_model_type}" = rec_rir ]; then
        for name in speech_direct speech_reverb; do
          if [ ! -f "data/${dset}/${name}.scp" ]; then
            echo "Missing data/${dset}/${name}.scp for --rir_model_type rec_rir" >&2
            exit 1
          fi
          scripts/audio/format_wav_scp.sh --nj "${nj}" --cmd "${train_cmd}" \
            --out-filename "${name}.scp" \
            --audio-format "${audio_format}" --fs "${fs}" \
            "data/${dset}/${name}.scp" "${data_feats}${suf}/${dset}" \
            "${data_feats}${suf}/${dset}/logs/${name}" \
            "${data_feats}${suf}/${dset}/data/${name}"
          utt_extra_files="${utt_extra_files} ${name}.scp"
        done
      elif [ "${rir_model_type}" = rec_rir_pit ] \
        || [ "${rir_model_type}" = rec_rir_pit_ctf_split ]; then
        for name in speech_direct1 speech_direct2 speech_reverb1 speech_reverb2; do
          if [ ! -f "data/${dset}/${name}.scp" ]; then
            echo "Missing data/${dset}/${name}.scp for --rir_model_type ${rir_model_type}" >&2
            exit 1
          fi
          scripts/audio/format_wav_scp.sh --nj "${nj}" --cmd "${train_cmd}" \
            --out-filename "${name}.scp" \
            --audio-format "${audio_format}" --fs "${fs}" \
            "data/${dset}/${name}.scp" "${data_feats}${suf}/${dset}" \
            "${data_feats}${suf}/${dset}/logs/${name}" \
            "${data_feats}${suf}/${dset}/data/${name}"
          utt_extra_files="${utt_extra_files} ${name}.scp"
        done
      fi

      cp "data/${dset}/rir_npz.scp" "${data_feats}${suf}/${dset}/rir_npz.scp"
      cp "data/${dset}/room_param_npz.scp" \
        "${data_feats}${suf}/${dset}/room_param_npz.scp"
      echo "raw" > "${data_feats}${suf}/${dset}/feats_type"
      utils/fix_data_dir.sh \
        --utt_extra_files "${utt_extra_files}" \
        "${data_feats}${suf}/${dset}"
    done
  fi

  if [ "${stage}" -le 4 ] && [ "${stop_stage}" -ge 4 ]; then
    echo "Stage 4: Remove short/long training data"
    for dset in "${train_set}" "${valid_set}"; do
      utils/copy_data_dir.sh "${data_feats}/org/${dset}" "${data_feats}/${dset}"
      cp "${data_feats}/org/${dset}/rir_npz.scp" "${data_feats}/${dset}/rir_npz.scp"
      cp "${data_feats}/org/${dset}/room_param_npz.scp" \
        "${data_feats}/${dset}/room_param_npz.scp"
      cp "${data_feats}/org/${dset}/feats_type" "${data_feats}/${dset}/feats_type"
      utt_extra_files="rir_npz.scp room_param_npz.scp"
      if [ "${rir_model_type}" = rec_rir ]; then
        for name in speech_direct speech_reverb; do
          cp "${data_feats}/org/${dset}/${name}.scp" \
            "${data_feats}/${dset}/${name}.scp"
          utt_extra_files="${utt_extra_files} ${name}.scp"
        done
      elif [ "${rir_model_type}" = rec_rir_pit ] \
        || [ "${rir_model_type}" = rec_rir_pit_ctf_split ]; then
        for name in speech_direct1 speech_direct2 speech_reverb1 speech_reverb2; do
          cp "${data_feats}/org/${dset}/${name}.scp" \
            "${data_feats}/${dset}/${name}.scp"
          utt_extra_files="${utt_extra_files} ${name}.scp"
        done
      fi

      fs_int=$(python3 -c "import humanfriendly as h; print(h.parse_size('${fs}'))")
      min_length=$(python3 -c "print(int(${min_wav_duration} * ${fs_int}))")
      max_length=$(python3 -c "print(int(${max_wav_duration} * ${fs_int}))")

      awk -v min_length="${min_length}" -v max_length="${max_length}" \
        '{ if ($2 > min_length && $2 < max_length) print $0; }' \
        "${data_feats}/org/${dset}/utt2num_samples" \
        > "${data_feats}/${dset}/utt2num_samples"

      for scp in wav.scp ${utt_extra_files}; do
        utils/filter_scp.pl "${data_feats}/${dset}/utt2num_samples" \
          "${data_feats}/org/${dset}/${scp}" > "${data_feats}/${dset}/${scp}"
      done
      utils/fix_data_dir.sh \
        --utt_extra_files "${utt_extra_files}" \
        "${data_feats}/${dset}"
    done
  fi
fi

if ! "${skip_train}"; then
  if [ "${stage}" -le 5 ] && [ "${stop_stage}" -ge 5 ]; then
    train_dir="${data_feats}/${train_set}"
    valid_dir="${data_feats}/${valid_set}"
    logdir="${rir_stats_dir}/logdir"
    mkdir -p "${logdir}"

    train_speech_mix_scp=wav.scp
    valid_speech_mix_scp=wav.scp
    if [ "${rir_model_type}" = rec_rir_pit ] \
      || [ "${rir_model_type}" = rec_rir_pit_ctf_split ]; then
      if [ -f "${train_dir}/speech_mix_pit.scp" ]; then
        train_speech_mix_scp=speech_mix_pit.scp
      fi
      if [ -f "${valid_dir}/speech_mix_pit.scp" ]; then
        valid_speech_mix_scp=speech_mix_pit.scp
      fi
    elif [ "${rir_model_type}" = rec_rir ]; then
      if [ -f "${train_dir}/speech_mix.scp" ]; then
        train_speech_mix_scp=speech_mix.scp
      fi
      if [ -f "${valid_dir}/speech_mix.scp" ]; then
        valid_speech_mix_scp=speech_mix.scp
      fi
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

    opts=
    [ -n "${rir_config}" ] && opts+="--config ${rir_config} "
    [ -n "${rir_model_type}" ] && opts+="--rir_model_type ${rir_model_type} "

    train_data_param="--train_data_path_and_name_and_type ${train_dir}/${train_speech_mix_scp},speech_mix,sound "
    valid_data_param="--valid_data_path_and_name_and_type ${valid_dir}/${valid_speech_mix_scp},speech_mix,sound "
    if [ "${rir_model_type}" = rec_rir ]; then
      train_data_param+="--train_data_path_and_name_and_type ${train_dir}/speech_direct.scp,speech_direct,sound "
      train_data_param+="--train_data_path_and_name_and_type ${train_dir}/speech_reverb.scp,speech_reverb,sound "
      valid_data_param+="--valid_data_path_and_name_and_type ${valid_dir}/speech_direct.scp,speech_direct,sound "
      valid_data_param+="--valid_data_path_and_name_and_type ${valid_dir}/speech_reverb.scp,speech_reverb,sound "
    elif [ "${rir_model_type}" = rec_rir_pit ] \
      || [ "${rir_model_type}" = rec_rir_pit_ctf_split ]; then
      train_data_param+="--train_data_path_and_name_and_type ${train_dir}/speech_direct1.scp,speech_direct1,sound "
      train_data_param+="--train_data_path_and_name_and_type ${train_dir}/speech_direct2.scp,speech_direct2,sound "
      train_data_param+="--train_data_path_and_name_and_type ${train_dir}/speech_reverb1.scp,speech_reverb1,sound "
      train_data_param+="--train_data_path_and_name_and_type ${train_dir}/speech_reverb2.scp,speech_reverb2,sound "
      valid_data_param+="--valid_data_path_and_name_and_type ${valid_dir}/speech_direct1.scp,speech_direct1,sound "
      valid_data_param+="--valid_data_path_and_name_and_type ${valid_dir}/speech_direct2.scp,speech_direct2,sound "
      valid_data_param+="--valid_data_path_and_name_and_type ${valid_dir}/speech_reverb1.scp,speech_reverb1,sound "
      valid_data_param+="--valid_data_path_and_name_and_type ${valid_dir}/speech_reverb2.scp,speech_reverb2,sound "
    else
      train_data_param+="--train_data_path_and_name_and_type ${train_dir}/rir_npz.scp,rir_path,text "
      train_data_param+="--train_data_path_and_name_and_type ${train_dir}/room_param_npz.scp,room_param_path,text "
      valid_data_param+="--valid_data_path_and_name_and_type ${valid_dir}/rir_npz.scp,rir_path,text "
      valid_data_param+="--valid_data_path_and_name_and_type ${valid_dir}/room_param_npz.scp,room_param_path,text "
    fi

    mkdir -p "${rir_stats_dir}"
    ${train_cmd} JOB=1:"${nj_stats}" "${logdir}"/stats.JOB.log \
      ${python} -m espnet2.bin.rir_train \
        --collect_stats true \
        ${train_data_param} \
        ${valid_data_param} \
        --train_shape_file "${logdir}/train.JOB.scp" \
        --valid_shape_file "${logdir}/valid.JOB.scp" \
        --output_dir "${logdir}/stats.JOB" \
        ${opts} ${rir_args}

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
    opts=
    [ -n "${rir_config}" ] && opts+="--config ${rir_config} "
    [ -n "${rir_model_type}" ] && opts+="--rir_model_type ${rir_model_type} "

    train_speech_mix_scp=wav.scp
    valid_speech_mix_scp=wav.scp
    if [ "${rir_model_type}" = rec_rir_pit ] \
      || [ "${rir_model_type}" = rec_rir_pit_ctf_split ]; then
      if [ -f "${train_dir}/speech_mix_pit.scp" ]; then
        train_speech_mix_scp=speech_mix_pit.scp
      fi
      if [ -f "${valid_dir}/speech_mix_pit.scp" ]; then
        valid_speech_mix_scp=speech_mix_pit.scp
      fi
    elif [ "${rir_model_type}" = rec_rir ]; then
      if [ -f "${train_dir}/speech_mix.scp" ]; then
        train_speech_mix_scp=speech_mix.scp
      fi
      if [ -f "${valid_dir}/speech_mix.scp" ]; then
        valid_speech_mix_scp=speech_mix.scp
      fi
    fi

    train_data_param="--train_data_path_and_name_and_type ${train_dir}/${train_speech_mix_scp},speech_mix,sound "
    valid_data_param="--valid_data_path_and_name_and_type ${valid_dir}/${valid_speech_mix_scp},speech_mix,sound "
    if [ "${rir_model_type}" = rec_rir ]; then
      train_data_param+="--train_data_path_and_name_and_type ${train_dir}/speech_direct.scp,speech_direct,sound "
      train_data_param+="--train_data_path_and_name_and_type ${train_dir}/speech_reverb.scp,speech_reverb,sound "
      valid_data_param+="--valid_data_path_and_name_and_type ${valid_dir}/speech_direct.scp,speech_direct,sound "
      valid_data_param+="--valid_data_path_and_name_and_type ${valid_dir}/speech_reverb.scp,speech_reverb,sound "
      train_shape_param="--train_shape_file ${rir_stats_dir}/train/speech_mix_shape "
      train_shape_param+="--train_shape_file ${rir_stats_dir}/train/speech_direct_shape "
      train_shape_param+="--train_shape_file ${rir_stats_dir}/train/speech_reverb_shape "
      valid_shape_param="--valid_shape_file ${rir_stats_dir}/valid/speech_mix_shape "
      valid_shape_param+="--valid_shape_file ${rir_stats_dir}/valid/speech_direct_shape "
      valid_shape_param+="--valid_shape_file ${rir_stats_dir}/valid/speech_reverb_shape "
      fold_length_param="--fold_length ${speech_fold_length} "
      fold_length_param+="--fold_length ${speech_fold_length} "
      fold_length_param+="--fold_length ${speech_fold_length} "
    elif [ "${rir_model_type}" = rec_rir_pit ] \
      || [ "${rir_model_type}" = rec_rir_pit_ctf_split ]; then
      train_data_param+="--train_data_path_and_name_and_type ${train_dir}/speech_direct1.scp,speech_direct1,sound "
      train_data_param+="--train_data_path_and_name_and_type ${train_dir}/speech_direct2.scp,speech_direct2,sound "
      train_data_param+="--train_data_path_and_name_and_type ${train_dir}/speech_reverb1.scp,speech_reverb1,sound "
      train_data_param+="--train_data_path_and_name_and_type ${train_dir}/speech_reverb2.scp,speech_reverb2,sound "
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
    else
      train_data_param+="--train_data_path_and_name_and_type ${train_dir}/rir_npz.scp,rir_path,text "
      train_data_param+="--train_data_path_and_name_and_type ${train_dir}/room_param_npz.scp,room_param_path,text "
      valid_data_param+="--valid_data_path_and_name_and_type ${valid_dir}/rir_npz.scp,rir_path,text "
      valid_data_param+="--valid_data_path_and_name_and_type ${valid_dir}/room_param_npz.scp,room_param_path,text "
      train_shape_param="--train_shape_file ${rir_stats_dir}/train/speech_mix_shape "
      train_shape_param+="--train_shape_file ${rir_stats_dir}/train/rir_ref_shape "
      valid_shape_param="--valid_shape_file ${rir_stats_dir}/valid/speech_mix_shape "
      valid_shape_param+="--valid_shape_file ${rir_stats_dir}/valid/rir_ref_shape "
      fold_length_param="--fold_length ${speech_fold_length} --fold_length ${rir_fold_length} "
    fi

    mkdir -p "${rir_exp}"
    ${python} -m espnet2.bin.launch \
      --cmd "${cuda_cmd}" \
      --log "${rir_exp}/train.log" \
      --ngpu "${ngpu}" \
      --num_nodes "${num_nodes}" \
      --init_file_prefix "${rir_exp}/.dist_init_" \
      --multiprocessing_distributed true -- \
      ${python} -m espnet2.bin.rir_train \
        ${train_data_param} \
        ${valid_data_param} \
        ${train_shape_param} \
        ${valid_shape_param} \
        ${fold_length_param} \
        --resume true \
        --output_dir "${rir_exp}" \
        ${opts} ${rir_args}
  fi
fi

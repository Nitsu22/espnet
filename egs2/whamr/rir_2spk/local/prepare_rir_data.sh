#!/usr/bin/env bash

set -e
set -u
set -o pipefail

min_or_max=min
sample_rate=8k
rir_input=single_clean_reverb
source_data_dir=
audio_data_dir=../se2_data/data
npz_data_dir=../se_npz/data
audio_variant=rand
data_dir=./data
speech_direct_scp=
speech_reverb_scp=

. utils/parse_options.sh
. ./path.sh

if [ -n "${source_data_dir}" ]; then
  audio_data_dir="${source_data_dir}"
  npz_data_dir="${source_data_dir}"
  audio_variant=none
fi

case "${rir_input}" in
  single_clean_reverb)
    base_mixtype=single
    input_scp=spk1_reverb.scp
    wav_subdir=s1_reverb
    speech_direct_scp=spk1.scp
    speech_reverb_scp=spk1_reverb.scp
    ;;
  single_noisy_reverb)
    base_mixtype=single
    input_scp=wav.scp
    wav_subdir=mix_single_reverb
    ;;
  clean_reverb)
    base_mixtype=clean
    input_scp=wav.scp
    wav_subdir=mix_clean_reverb
    ;;
  both_reverb)
    base_mixtype=both
    input_scp=wav.scp
    wav_subdir=mix_both_reverb
    ;;
  *)
    echo "Unsupported --rir_input: ${rir_input}" >&2
    exit 1
    ;;
esac

if [ -n "${speech_direct_scp}" ]; then
  if [ "${audio_variant}" = rand ]; then
    speech_direct_subdir=s1_rand_anechoic
    speech_reverb_subdir=s1_rand_reverb
  else
    speech_direct_subdir=s1_anechoic
    speech_reverb_subdir=s1_reverb
  fi
fi

for split in tr cv tt; do
  if [ "${audio_variant}" = rand ]; then
    audio_set="${split}_mix_${base_mixtype}_rand_reverb_${min_or_max}_${sample_rate}"
  elif [ "${audio_variant}" = none ]; then
    audio_set="${split}_mix_${base_mixtype}_reverb_${min_or_max}_${sample_rate}"
  else
    echo "Unsupported --audio_variant: ${audio_variant}" >&2
    exit 1
  fi
  npz_set="${split}_mix_${base_mixtype}_reverb_${min_or_max}_${sample_rate}"
  audio_src_dir="${audio_data_dir}/${audio_set}"
  npz_src_dir="${npz_data_dir}/${npz_set}"
  dst_dir="${data_dir}/${split}_rir_${rir_input}_${min_or_max}_${sample_rate}"

  for required in "${input_scp}" utt2spk; do
    if [ ! -f "${audio_src_dir}/${required}" ]; then
      echo "Missing ${audio_src_dir}/${required}" >&2
      exit 1
    fi
  done
  for required in ${speech_direct_scp} ${speech_reverb_scp}; do
    if [ -n "${required}" ] && [ ! -f "${audio_src_dir}/${required}" ]; then
      echo "Missing ${audio_src_dir}/${required}" >&2
      exit 1
    fi
  done
  for required in rir_npz.scp room_param_npz.scp; do
    if [ ! -f "${npz_src_dir}/${required}" ]; then
      echo "Missing ${npz_src_dir}/${required}" >&2
      exit 1
    fi
  done

  mkdir -p "${dst_dir}"

  awk -v wav_subdir="${wav_subdir}" '
    function to_wav(path, out) {
      out = path
      if (out ~ /\.npz$/) {
        sub(/\.npz$/, ".wav", out)
        sub(/\/npz\//, "/" wav_subdir "/", out)
      }
      return out
    }
    { print $1, to_wav($2) }
  ' "${audio_src_dir}/${input_scp}" | sort > "${dst_dir}/wav.scp"

  extra_utt_files="rir_npz.scp room_param_npz.scp"
  if [ -n "${speech_direct_scp}" ]; then
    awk -v wav_subdir="${speech_direct_subdir}" '
      function to_wav(path, out) {
        out = path
        if (out ~ /\.npz$/) {
          sub(/\.npz$/, ".wav", out)
          sub(/\/npz\//, "/" wav_subdir "/", out)
        }
        return out
      }
      { print $1, to_wav($2) }
    ' "${audio_src_dir}/${speech_direct_scp}" | sort > "${dst_dir}/speech_direct.scp"

    awk -v wav_subdir="${speech_reverb_subdir}" '
      function to_wav(path, out) {
        out = path
        if (out ~ /\.npz$/) {
          sub(/\.npz$/, ".wav", out)
          sub(/\/npz\//, "/" wav_subdir "/", out)
        }
        return out
      }
      { print $1, to_wav($2) }
    ' "${audio_src_dir}/${speech_reverb_scp}" | sort > "${dst_dir}/speech_reverb.scp"
    extra_utt_files="${extra_utt_files} speech_direct.scp speech_reverb.scp"
  fi

  sort "${npz_src_dir}/rir_npz.scp" > "${dst_dir}/rir_npz.scp"
  sort "${npz_src_dir}/room_param_npz.scp" > "${dst_dir}/room_param_npz.scp"
  sort "${audio_src_dir}/utt2spk" > "${dst_dir}/utt2spk"
  utt2spk_to_spk2utt.pl "${dst_dir}/utt2spk" > "${dst_dir}/spk2utt"

  utils/fix_data_dir.sh \
    --utt_extra_files "${extra_utt_files}" \
    "${dst_dir}"
done

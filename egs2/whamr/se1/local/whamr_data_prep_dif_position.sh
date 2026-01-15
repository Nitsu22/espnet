#!/usr/bin/env bash

# Copyright 2020  Shanghai Jiao Tong University (Authors: Wangyou Zhang)
# Apache 2.0

min_or_max=min
sample_rate=8k

. utils/parse_options.sh
. ./path.sh

if [[ "$min_or_max" != "max" ]] && [[ "$min_or_max" != "min" ]]; then
  echo "Error: min_or_max must be either max or min: ${min_or_max}"
  exit 1
fi
if [[ "$sample_rate" != "16k" ]] && [[ "$sample_rate" != "8k" ]]; then
  echo "Error: sample rate must be either 16k or 8k: ${sample_rate}"
  exit 1
fi

if [ $# -ne 3 ]; then
  echo "Arguments should be WHAMR script path, WHAMR wav path and the WSJ0 path, see local/data.sh for example."
  exit 1;
fi

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

whamr_script_dir=$1
whamr_wav_dir=$2
wsj_full_wav=$3


# check if the wav dirs exist (dif_position data only: reverb_dif_position)
for x in tr cv tt; do
  for ddir in mix_both_reverb_dif_position mix_clean_reverb_dif_position; do
    f=${whamr_wav_dir}/wav${sample_rate}/${min_or_max}/${x}/${ddir}
    if [ ! -d $f ]; then
      echo "Error: $f is not a directory."
      exit 1;
    fi
  done
done

# check if existing datasets exist
data=./data
for x in tr cv tt; do
  for mixtype in both clean; do
    ddir=${x}_mix_${mixtype}_reverb_${min_or_max}_${sample_rate}
    if [ ! -d ${data}/${ddir} ]; then
      echo "Error: Existing dataset ${data}/${ddir} does not exist."
      echo "Please run data preparation for reverb data first."
      exit 1;
    fi
  done
done

# Create wav_dif_position.scp in existing datasets
for x in tr cv tt; do
  for mixtype in both clean; do
    ddir=${x}_mix_${mixtype}_reverb_${min_or_max}_${sample_rate}
    rootdir=${whamr_wav_dir}/wav${sample_rate}/${min_or_max}/${x}
    mixwav_dir=${rootdir}/mix_${mixtype}_reverb_dif_position
    awk -v dir="${mixwav_dir}" -F "," \
      'NR>1 {sub(/\.wav$/, "", $1); split($1, lst, "_"); spk=substr(lst[1],1,3)"_"substr(lst[3],1,3); print(spk "_" $1 "_reverb", dir "/" $1 ".wav")}' \
      ${whamr_script_dir}/data/mix_2_spk_filenames_${x}.csv | sort > ${data}/${ddir}/wav_dif_position.scp
  done
done

exit 0

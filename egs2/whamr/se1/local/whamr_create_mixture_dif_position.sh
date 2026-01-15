#!/usr/bin/env bash

# Copyright  2020  Shanghai Jiao Tong University (Authors: Wangyou Zhang)
# Apache 2.0

wham_noise=   # Path to the directory containing WHAM! noise
mono=False
min_or_max=min
sample_rate=8k

. utils/parse_options.sh
. path.sh
. cmd.sh

if [ $# -ne 4 ]; then
  echo "Usage: $0 <dir> <wsj0-path> <wsj0-full-wav> <whamr-wav>"
  echo " where <dir> is download space (whamr_scripts should already exist here),"
  echo " <wsj0-path> is the original wsj0 path (not used, kept for compatibility),"
  echo " <wsj0-full-wav> is wsj0 full wave files path (should already exist),"
  echo " <whamr-wav> is wav generation space (dif_position data will be added here)."
  echo "Note: Assumes data.sh has already been executed."
  echo "      whamr_scripts, wham_noise, and wsj_full_wav should already exist."
  exit 1;
fi

dir=$1
wsj0_path=$2
wsj_full_wav=$3
whamr_wav=$4

if [ -z "$wham_noise" ]; then
  wham_noise=${dir}/wham_noise
fi

# Assumes data.sh has already been executed, so:
# - whamr_scripts directory and create_wham_from_scratch_dif_position.py already exist
# - wham_noise already exists
# - wsj_full_wav already exists

# Check if pyroomacoustics is installed
if [ -z "$(python -m pip list | grep pyroomacoustics)" ]; then
  echo -e "Please install pyroomacoustics first:\n pip install pyroomacoustics==0.2.0"
  exit 1;
fi

# Configure and run create_wham_from_scratch_dif_position.py
cd ${dir}/whamr_scripts || exit 1
if [ -f ${dir}/whamr_scripts/create_wham_from_scratch_dif_position.py ]; then
  echo "Creating Mixtures with different speaker positions (reverb_dif_position)."
  sed -i -e "s#MONO = True#MONO = ${mono}#" \
         -e "s#DATA_LEN = \['max', 'min'\]#DATA_LEN = ['${min_or_max}']#" \
         -e "s#SAMPLE_RATES = \['16k', '8k'\]#SAMPLE_RATES = ['${sample_rate}']#" \
         ${dir}/whamr_scripts/create_wham_from_scratch_dif_position.py
  echo "Log is in ${dir}/whamr_scripts/mix_dif_position.log"
  ${train_cmd} ${dir}/whamr_scripts/mix_dif_position.log python create_wham_from_scratch_dif_position.py \
    --wsj0-root ${wsj_full_wav} \
    --wham-noise-root ${wham_noise} \
    --output-dir ${whamr_wav}
else
  echo "Error: create_wham_from_scratch_dif_position.py not found in ${dir}/whamr_scripts/"
  exit 1
fi

# In the default configuration, the script will write about 444 GB of data:
#  - min_8k: 52 GB (mono=True) / 102 GB (mono=False)
#  - min_16k: ? GB
#  - max_8k: ? GB
#  - max_16k: ? GB

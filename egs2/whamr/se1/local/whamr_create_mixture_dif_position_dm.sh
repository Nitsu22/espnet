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
  echo " where <dir> is download space,"
  echo " <wsj0-path> is the original wsj0 path"
  echo " <wsj0-full-wav> is wsj0 full wave files path, <whamr-wav> is wav generation space."
  echo "Note: this script will download whamr_scripts and wham_noise if needed,"
  echo "and convert WSJ0 to wav format if needed."
  exit 1;
fi

dir=$1
wsj0_path=$2
wsj_full_wav=$3
whamr_wav=$4

wdir=data_dif_position/local/downloads
mkdir -p ${dir}
mkdir -p ${wdir}
echo "Downloading WHAMR! data generation scripts and documentation."

if [ -z "$wham_noise" ]; then
  # 17.65 GB unzipping to 35 GB
  wham_noise_url=https://my-bucket-a8b4b49c25c811ee9a7e8bba05fa24c7.s3.amazonaws.com/wham_noise.zip
  wget --continue -O $wdir/wham_noise.zip ${wham_noise_url}
  if [ $(ls ${dir}/wham_noise 2>/dev/null | wc -l) -eq 4 ]; then
    echo "'${dir}/wham_noise/' already exists. Skipping..."
  else
    unzip ${wdir}/wham_noise.zip -d ${dir}
  fi
  wham_noise=${dir}/wham_noise
fi

# Download whamr_scripts if it doesn't exist
if [ ! -d ${dir}/whamr_scripts ]; then
  script_url=https://my-bucket-a8b4b49c25c811ee9a7e8bba05fa24c7.s3.amazonaws.com/whamr_scripts.tar.gz
  wget --continue -O $wdir/whamr_scripts.tar.gz ${script_url}
  tar -xzf ${wdir}/whamr_scripts.tar.gz -C ${dir}
fi

# Convert WSJ0 to wav format if needed
if [ ! -d ${wsj_full_wav} ] || [ -z "$(ls -A ${wsj_full_wav} 2>/dev/null)" ]; then
  echo "WSJ0 wav file conversion."
  local/convert2wav.sh ${wsj0_path} ${wsj_full_wav} || exit 1;
else
  echo "WSJ0 wav files already exist at ${wsj_full_wav}. Skipping conversion..."
fi

# Check if pyroomacoustics is installed
if [ -z "$(python -m pip list | grep pyroomacoustics)" ]; then
  echo -e "Please install pyroomacoustics first:\n pip install pyroomacoustics==0.2.0"
  exit 1;
fi

# Check if create_wham_from_scratch_dif_position_dm.py exists
if [ ! -f ${dir}/whamr_scripts/create_wham_from_scratch_dif_position_dm.py ]; then
  echo "Error: create_wham_from_scratch_dif_position_dm.py not found in ${dir}/whamr_scripts/"
  echo "Please ensure this file exists in the whamr_scripts directory."
  exit 1
fi

# Configure and run create_wham_from_scratch_dif_position_dm.py
cd ${dir}/whamr_scripts || exit 1
echo "Creating Mixtures with different speaker positions (reverb_dif_position)."
sed -i -e "s#MONO = True#MONO = ${mono}#" \
       -e "s#DATA_LEN = \['max', 'min'\]#DATA_LEN = ['${min_or_max}']#" \
       -e "s#SAMPLE_RATES = \['16k', '8k'\]#SAMPLE_RATES = ['${sample_rate}']#" \
       ${dir}/whamr_scripts/create_wham_from_scratch_dif_position_dm.py
echo "Log is in ${dir}/whamr_scripts/mix_dif_position.log"
${train_cmd} ${dir}/whamr_scripts/mix_dif_position.log python create_wham_from_scratch_dif_position_dm.py \
  --wsj0-root ${wsj_full_wav} \
  --wham-noise-root ${wham_noise} \
  --output-dir ${whamr_wav}

# In the default configuration, the script will write about 444 GB of data:
#  - min_8k: 52 GB (mono=True) / 102 GB (mono=False)
#  - min_16k: ? GB
#  - max_8k: ? GB
#  - max_16k: ? GB

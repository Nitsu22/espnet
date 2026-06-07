#!/usr/bin/env bash

# Copyright  2020  Shanghai Jiao Tong University (Authors: Wangyou Zhang)
# Apache 2.0

wham_noise=   # Path to the directory containing WHAM! noise
mono=False
min_or_max=min
sample_rate=8k
sample_rates=
wham_create_script=  # Path to the WHAMR mixture generation script

. utils/parse_options.sh
. path.sh
. cmd.sh

if [ $# -ne 4 ]; then
  echo "Usage: $0 <dir> <wsj0-path> <wsj0-full-wav> <whamr-wav>"
  echo " where <dir> is download space,"
  echo " <wsj0-path> is the original wsj0 path"
  echo " <wsj0-full-wav> is wsj0 full wave files path, <whamr-wav> is wav generation space."
  echo "Note: this script won't actually re-download things if called twice,"
  echo "because we use the --continue flag to 'wget'."
  echo "Options include --sample-rates \"8k 16k\" and --wham-create-script <path>."
  echo "Note: <wsj0-full-wav> contains all the wsj0 (or wsj) utterances in wav format,"
  echo "and the directory is organized according to"
  echo "  scripts/data/mix_2_spk_filenames_{tr,cv,tt}.csv"
  echo ", which are the mixture combination schemes."
  exit 1;
fi

dir=$1
wsj0_path=$2
wsj_full_wav=$3
whamr_wav=$4

if [ -z "${sample_rates}" ]; then
  sample_rates=${sample_rate}
fi
sample_rates_py="["
for sr in ${sample_rates}; do
  if [[ "${sr}" != "16k" ]] && [[ "${sr}" != "8k" ]]; then
    echo "Error: sample rate must be either 16k or 8k: ${sr}"
    exit 1
  fi
  sample_rates_py="${sample_rates_py}'${sr}', "
done
sample_rates_py="${sample_rates_py%, }]"

wdir=data/local/downloads
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

if [ -z "${wham_create_script}" ]; then
  script_url=https://my-bucket-a8b4b49c25c811ee9a7e8bba05fa24c7.s3.amazonaws.com/whamr_scripts.tar.gz
  wget --continue -O $wdir/whamr_scripts.tar.gz ${script_url}
  tar -xzf ${wdir}/whamr_scripts.tar.gz -C ${dir}
  wham_create_script=${dir}/whamr_scripts/create_wham_from_scratch.py
fi
if [ ! -f "${wham_create_script}" ]; then
  echo "Error: wham_create_script does not exist: ${wham_create_script}"
  exit 1
fi
wham_create_script_dir=$(dirname "${wham_create_script}")
wham_create_script_name=$(basename "${wham_create_script}")

# If you want to generate both min and max versions with 8k and 16k data,
#  remove lines 59 and 60.
sed -i -e "s#^MONO = .*#MONO = ${mono}#" \
       -e "s#^DATA_LEN = .*#DATA_LEN = ['${min_or_max}']#" \
       -e "s#^SAMPLE_RATES = .*#SAMPLE_RATES = ${sample_rates_py}#" \
       "${wham_create_script}"

echo "WSJ0 wav file."
local/convert2wav.sh ${wsj0_path} ${wsj_full_wav} || exit 1;

echo "Creating Mixtures."
if [ -z "$(python -m pip list | grep pyroomacoustics)" ]; then
  echo -e "Please install pyroomacoustics first:\n pip install pyroomacoustics==0.2.0"
  exit 1;
fi
# Run simulation (single-process)
# (This may take ~11 hours to generate min version, 8k data
#  on Intel(R) Xeon(R) CPU E5-2680 v3 @ 2.50GHz)
cd "${wham_create_script_dir}" || exit 1
echo "Log is in ${wham_create_script_dir}/mix.log"
${train_cmd} "${wham_create_script_dir}/mix.log" python "${wham_create_script_name}" \
  --wsj0-root ${wsj_full_wav} \
  --wham-noise-root ${wham_noise} \
  --output-dir ${whamr_wav}

# In the default configuration, the script will write about 444 GB of data:
#  - min_8k: 52 GB (mono=True) / 102 GB (mono=False)
#  - min_16k: ? GB
#  - max_8k: ? GB
#  - max_16k: ? GB

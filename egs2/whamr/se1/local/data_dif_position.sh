#!/usr/bin/env bash

# Copyright 2020  Shanghai Jiao Tong University (Authors: Chenda Li, Wangyou Zhang)
# Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

help_message=$(cat << EOF
Usage: $0 [--mono <True/False>] [--min_or_max <min/max>] [--sample_rate <8k/16k>]
  optional argument:
    [--mono]: True (Default), False
    [--min_or_max]: min (Default), max
    [--sample_rate]: 8k (Default), 16k
EOF
)

. ./db.sh

wsj_full_wav=$PWD/data/wsj0/wsj0_wav
# Path to the directory containing WHAM! noise (should already exist from data.sh)
wham_noise=/net/midgar/work/nitsu/data/wsj/wham_noise
whamr_wav=$PWD/data/whamr/2speakers
whamr_scripts=$PWD

mono=False
min_or_max=min
sample_rate=8k


. utils/parse_options.sh

if [ $# -ne 0 ]; then
    echo "${help_message}"
    exit 1;
fi

if [ ! -d "${wsj_full_wav}" ]; then
    log "Missing wsj0 wav directory: ${wsj_full_wav}"
    exit 1
fi


### This part is for WHAMR! different speaker position data
### Assumes data.sh has already been executed, so:
### - Normal data already exist
### - whamr_scripts, wham_noise, wsj_full_wav already exist
### - reverb_params_*_dif_position.csv already exist
### Only generates and prepares different speaker position data (reverb_dif_position)
### Only mix_both and mix_clean are generated
### wav_dif_position.scp will be added to existing {tr,cv,tt}_mix_{both,clean}_reverb_${min_or_max}_${sample_rate} datasets

# Generate different speaker position mixtures
log "Generating different speaker position mixtures"
local/whamr_create_mixture_dif_position.sh --mono ${mono} --min-or-max ${min_or_max} --sample-rate ${sample_rate} \
    ${wham_noise:+--wham_noise $wham_noise} \
    ${whamr_scripts} ${WSJ0} ${wsj_full_wav} \
    ${whamr_wav} || exit 1;

# Prepare different speaker position data (reverb_dif_position)
# wav_dif_position.scp will be added to existing datasets:
# {tr,cv,tt}_mix_{both,clean}_reverb_${min_or_max}_${sample_rate}
log "Preparing different speaker position data"
local/whamr_data_prep_dif_position.sh --min-or-max ${min_or_max} --sample-rate ${sample_rate} \
    ${whamr_scripts}/whamr_scripts ${whamr_wav} ${wsj_full_wav} || exit 1;

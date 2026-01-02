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

find_transcripts=$KALDI_ROOT/egs/wsj/s5/local/find_transcripts.pl
normalize_transcript=$KALDI_ROOT/egs/wsj/s5/local/normalize_transcript.pl

whamr_script_dir=$1
whamr_wav_dir=$2
wsj_full_wav=$3


# check if the wav dirs exist (only for reverse data: reverbse and anechoic_reverse)
for x in tr cv tt; do
  for ddir in mix_both_reverbse mix_clean_reverbse mix_both_anechoic_reverse mix_clean_anechoic_reverse; do
    f=${whamr_wav_dir}/wav${sample_rate}/${min_or_max}/${x}/${ddir}
    if [ ! -d $f ]; then
      echo "Error: $f is not a directory."
      exit 1;
    fi
  done
done

data=./data
rm -r ${data}/{tr,cv,tt}_mix_{both,clean}_reverb_{reverbse,anechoic_reverse}_${min_or_max}_${sample_rate} 2>/dev/null || true

for x in tr cv tt; do
  for mixtype in both clean; do
    # Process reverbse
    ddir=${x}_mix_${mixtype}_reverb_reverbse_${min_or_max}_${sample_rate}
    mkdir -p ${data}/${ddir}
    rootdir=${whamr_wav_dir}/wav${sample_rate}/${min_or_max}/${x}
    mixwav_dir=${rootdir}/mix_${mixtype}_reverbse
    awk -v dir="${mixwav_dir}" -v suffix="reverbse" -F "," \
      'NR>1 {sub(/\.wav$/, "", $1); split($1, lst, "_"); spk=substr(lst[1],1,3)"_"substr(lst[3],1,3); print(spk "_" $1 "_" suffix, dir "/" $1 ".wav")}' \
      ${whamr_script_dir}/data/mix_2_spk_filenames_${x}.csv | sort > ${data}/${ddir}/wav.scp

    awk '{split($1, lst, "_"); spk=lst[1]"_"lst[2]; print($1, spk)}' ${data}/${ddir}/wav.scp | \
      sort > ${data}/${ddir}/utt2spk
    utt2spk_to_spk2utt.pl ${data}/${ddir}/utt2spk > ${data}/${ddir}/spk2utt

    if [[ "$mixtype" != "clean" ]]; then
      noise_wav_dir=${rootdir}/noise
      sed -e "s#${mixwav_dir}#${noise_wav_dir}#g" ${data}/${ddir}/wav.scp \
        > ${data}/${ddir}/noise1.scp
    fi

    # For reverbse, use anechoic_reverse for spk references (if available) or original anechoic
    # Since positions are reversed, we need to handle spk references appropriately
    # Use anechoic_reverse if available, otherwise use original anechoic
    if [ -d ${rootdir}/s1_anechoic_reverse ] && [ -d ${rootdir}/s2_anechoic_reverse ]; then
      spk1_wav_dir=${rootdir}/s1_anechoic_reverse
      spk2_wav_dir=${rootdir}/s2_anechoic_reverse
    else
      # Fallback to original anechoic (positions are reversed in the room simulation)
      spk1_wav_dir=${rootdir}/s1_anechoic
      spk2_wav_dir=${rootdir}/s2_anechoic
    fi
    sed -e "s#${mixwav_dir}#${spk1_wav_dir}#g" ${data}/${ddir}/wav.scp \
      > ${data}/${ddir}/spk1.scp
    sed -e "s#${mixwav_dir}#${spk2_wav_dir}#g" ${data}/${ddir}/wav.scp \
      > ${data}/${ddir}/spk2.scp

    # reverb scps (use reverbse directory if available, otherwise use original reverb)
    # Note: s1_reverbse and s2_reverbse may not exist if create_wham_from_scratch_reverse.py doesn't save them
    if [ -d ${rootdir}/s1_reverbse ] && [ -d ${rootdir}/s2_reverbse ]; then
      spk1_wav_dir=${rootdir}/s1_reverbse
      spk2_wav_dir=${rootdir}/s2_reverbse
    else
      # Fallback to original reverb (positions are reversed in the room simulation)
      spk1_wav_dir=${rootdir}/s1_reverb
      spk2_wav_dir=${rootdir}/s2_reverb
    fi
    sed -e "s#${mixwav_dir}#${spk1_wav_dir}#g" ${data}/${ddir}/wav.scp \
      > ${data}/${ddir}/spk1_reverb.scp
    sed -e "s#${mixwav_dir}#${spk2_wav_dir}#g" ${data}/${ddir}/wav.scp \
      > ${data}/${ddir}/spk2_reverb.scp

    # dereverb1.scp (use anechoic_reverse if available)
    if [ -d ${rootdir}/mix_${mixtype}_anechoic_reverse ]; then
      anechoic_mixwav_dir=${rootdir}/mix_${mixtype}_anechoic_reverse
    else
      anechoic_mixwav_dir=${rootdir}/mix_${mixtype}_anechoic
    fi
    sed -e "s#${mixwav_dir}#${anechoic_mixwav_dir}#g" ${data}/${ddir}/wav.scp \
      > ${data}/${ddir}/dereverb1.scp

    # Process anechoic_reverse
    ddir=${x}_mix_${mixtype}_reverb_anechoic_reverse_${min_or_max}_${sample_rate}
    mkdir -p ${data}/${ddir}
    mixwav_dir=${rootdir}/mix_${mixtype}_anechoic_reverse
    awk -v dir="${mixwav_dir}" -v suffix="anechoic_reverse" -F "," \
      'NR>1 {sub(/\.wav$/, "", $1); split($1, lst, "_"); spk=substr(lst[1],1,3)"_"substr(lst[3],1,3); print(spk "_" $1 "_" suffix, dir "/" $1 ".wav")}' \
      ${whamr_script_dir}/data/mix_2_spk_filenames_${x}.csv | sort > ${data}/${ddir}/wav.scp

    awk '{split($1, lst, "_"); spk=lst[1]"_"lst[2]; print($1, spk)}' ${data}/${ddir}/wav.scp | \
      sort > ${data}/${ddir}/utt2spk
    utt2spk_to_spk2utt.pl ${data}/${ddir}/utt2spk > ${data}/${ddir}/spk2utt

    if [[ "$mixtype" != "clean" ]]; then
      noise_wav_dir=${rootdir}/noise
      sed -e "s#${mixwav_dir}#${noise_wav_dir}#g" ${data}/${ddir}/wav.scp \
        > ${data}/${ddir}/noise1.scp
    fi

    # For anechoic_reverse, spk references are the same (anechoic_reverse)
    spk1_wav_dir=${rootdir}/s1_anechoic_reverse
    spk2_wav_dir=${rootdir}/s2_anechoic_reverse
    if [ -d ${spk1_wav_dir} ] && [ -d ${spk2_wav_dir} ]; then
      sed -e "s#${mixwav_dir}#${spk1_wav_dir}#g" ${data}/${ddir}/wav.scp \
        > ${data}/${ddir}/spk1.scp
      sed -e "s#${mixwav_dir}#${spk2_wav_dir}#g" ${data}/${ddir}/wav.scp \
        > ${data}/${ddir}/spk2.scp
    else
      # Fallback to original anechoic
      spk1_wav_dir=${rootdir}/s1_anechoic
      spk2_wav_dir=${rootdir}/s2_anechoic
      sed -e "s#${mixwav_dir}#${spk1_wav_dir}#g" ${data}/${ddir}/wav.scp \
        > ${data}/${ddir}/spk1.scp
      sed -e "s#${mixwav_dir}#${spk2_wav_dir}#g" ${data}/${ddir}/wav.scp \
        > ${data}/${ddir}/spk2.scp
    fi
  done
done


# transcriptions (only for 'max' version)
# Assumes transcriptions already exist from data.sh execution
if [[ "$min_or_max" = "min" ]]; then
  exit 0
fi

# Use existing transcriptions from data.sh execution
# They should be in data/wsj/ or can be regenerated from existing tmp directory
# For simplicity, we'll use the same approach but assume tmp/ might already exist
if [ ! -d tmp ]; then
  mkdir -p tmp
  cd tmp
  for i in si_tr_s si_et_05 si_dt_05; do
      cp ${wsj_full_wav}/${i}.scp .
  done

  # Finding the transcript files:
  for x in `ls ${wsj_full_wav}/links/`; do find -L ${wsj_full_wav}/links/$x -iname '*.dot'; done > dot_files.flist

  # Convert the transcripts into our format (no normalization yet)
  for f in si_tr_s si_et_05 si_dt_05; do
    cat ${f}.scp | awk '{print $1}' | ${find_transcripts} dot_files.flist > ${f}.trans1

    # Do some basic normalization steps.  At this point we don't remove OOVs--
    # that will be done inside the training scripts, as we'd like to make the
    # data-preparation stage independent of the specific lexicon used.
    noiseword="<NOISE>"
    cat ${f}.trans1 | ${normalize_transcript} ${noiseword} | sort > ${f}.txt || exit 1;
  done
  cd ..
else
  # tmp directory already exists from data.sh, use it
  cd tmp
  # Check if transcript files exist, if not regenerate them
  if [ ! -f si_tr_s.txt ] || [ ! -f si_et_05.txt ] || [ ! -f si_dt_05.txt ]; then
    for i in si_tr_s si_et_05 si_dt_05; do
        if [ ! -f ${i}.scp ]; then
            cp ${wsj_full_wav}/${i}.scp .
        fi
    done
    if [ ! -f dot_files.flist ]; then
      for x in `ls ${wsj_full_wav}/links/`; do find -L ${wsj_full_wav}/links/$x -iname '*.dot'; done > dot_files.flist
    fi
    for f in si_tr_s si_et_05 si_dt_05; do
      if [ ! -f ${f}.txt ]; then
        cat ${f}.scp | awk '{print $1}' | ${find_transcripts} dot_files.flist > ${f}.trans1
        noiseword="<NOISE>"
        cat ${f}.trans1 | ${normalize_transcript} ${noiseword} | sort > ${f}.txt || exit 1;
      fi
    done
  fi
  cd ..
fi

for mixtype in both clean; do
  for cond_suffix in reverbse anechoic_reverse; do
    if [[ "$cond_suffix" = "reverbse" ]]; then
      cond_name="reverb_reverbse"
    else
      cond_name="reverb_anechoic_reverse"
    fi
    tr=tr_mix_${mixtype}_${cond_name}_${min_or_max}_${sample_rate}
    cv=cv_mix_${mixtype}_${cond_name}_${min_or_max}_${sample_rate}
    tt=tt_mix_${mixtype}_${cond_name}_${min_or_max}_${sample_rate}
    awk '(ARGIND==1) {txt[$1]=$0} (ARGIND==2) {split($1, lst, "_"); utt1=lst[3]; text=txt[utt1]; print($1, text)}' tmp/si_tr_s.txt ${data}/${tr}/wav.scp | awk '{$2=""; print $0}' > ${data}/${tr}/text_spk1
    awk '(ARGIND==1) {txt[$1]=$0} (ARGIND==2) {split($1, lst, "_"); utt2=lst[5]; text=txt[utt2]; print($1, text)}' tmp/si_tr_s.txt ${data}/${tr}/wav.scp | awk '{$2=""; print $0}' > ${data}/${tr}/text_spk2
    awk '(ARGIND==1) {txt[$1]=$0} (ARGIND==2) {split($1, lst, "_"); utt1=lst[3]; text=txt[utt1]; print($1, text)}' tmp/si_tr_s.txt ${data}/${cv}/wav.scp | awk '{$2=""; print $0}' > ${data}/${cv}/text_spk1
    awk '(ARGIND==1) {txt[$1]=$0} (ARGIND==2) {split($1, lst, "_"); utt2=lst[5]; text=txt[utt2]; print($1, text)}' tmp/si_tr_s.txt ${data}/${cv}/wav.scp | awk '{$2=""; print $0}' > ${data}/${cv}/text_spk2
    awk '(ARGIND<=2) {txt[$1]=$0} (ARGIND==3) {split($1, lst, "_"); utt1=lst[3]; text=txt[utt1]; print($1, text)}' tmp/si_dt_05.txt tmp/si_et_05.txt ${data}/${tt}/wav.scp | awk '{$2=""; print $0}' > ${data}/${tt}/text_spk1
    awk '(ARGIND<=2) {txt[$1]=$0} (ARGIND==3) {split($1, lst, "_"); utt2=lst[5]; text=txt[utt2]; print($1, text)}' tmp/si_dt_05.txt tmp/si_et_05.txt ${data}/${tt}/wav.scp | awk '{$2=""; print $0}' > ${data}/${tt}/text_spk2
  done
done
rm -r tmp


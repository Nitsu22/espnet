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
script_dir=$(cd "$(dirname "$0")" && pwd)
recipe_dir=$(cd "${script_dir}/.." && pwd)
npz_root=${recipe_dir}/data/whamr/2speakers


# check if the npz dirs exist.
for x in tr cv tt; do
  f=${whamr_wav_dir}/wav${sample_rate}/${min_or_max}/${x}/npz
  if [ ! -d $f ]; then
    echo "Error: $f is not a directory."
    exit 1;
  fi
done

# check if the base npy dirs exist.
for x in tr cv tt; do
  for ddir in s1_base_npz s2_base_npz s1_temp_npz s2_temp_npz noise_base_npz rir_npz room_param_npz; do
    f=${npz_root}/wav${sample_rate}/${min_or_max}/${x}/${ddir}
    if [ ! -d $f ]; then
      echo "Error: $f is not a directory."
      exit 1;
    fi
  done
done

data=./data
rm -r ${data}/{tr,cv,tt}_mix_{both,clean,single}_{anechoic,reverb}_${min_or_max}_${sample_rate} 2>/dev/null || true

for x in tr cv tt; do
  for mixtype in both clean single; do
    for cond in anechoic reverb; do
      ddir=${x}_mix_${mixtype}_${cond}_${min_or_max}_${sample_rate}
      mkdir -p ${data}/${ddir}
      rootdir=${whamr_wav_dir}/wav${sample_rate}/${min_or_max}/${x}/npz
      mixwav_dir=${rootdir}
      awk -v dir="${mixwav_dir}" -v suffix="${cond}" -F "," \
        'NR>1 {sub(/\.wav$/, "", $1); split($1, lst, "_"); spk=substr(lst[1],1,3)"_"substr(lst[3],1,3); print(spk "_" $1 "_" suffix, dir "/" $1 ".npz")}' \
        ${whamr_script_dir}/data/mix_2_spk_filenames_${x}.csv | sort > ${data}/${ddir}/wav.scp

      # npz.scp (same keys as wav.scp; base npz path under se_npz data dir)
      npz_dir=${npz_root}/wav${sample_rate}/${min_or_max}/${x}/npz
      awk -v dir="${npz_dir}" -v suffix="${cond}" -F "," \
        'NR>1 {sub(/\.wav$/, "", $1); split($1, lst, "_"); spk=substr(lst[1],1,3)"_"substr(lst[3],1,3); print(spk "_" $1 "_" suffix, dir "/" $1 ".npz")}' \
        ${whamr_script_dir}/data/mix_2_spk_filenames_${x}.csv | sort > ${data}/${ddir}/npz.scp

      awk '{split($1, lst, "_"); spk=lst[1]"_"lst[2]; print($1, spk)}' ${data}/${ddir}/wav.scp | \
        sort > ${data}/${ddir}/utt2spk
      utt2spk_to_spk2utt.pl ${data}/${ddir}/utt2spk > ${data}/${ddir}/spk2utt

      if [[ "$mixtype" != "clean" ]]; then
        noise_wav_dir=${rootdir}
        sed -e "s#${mixwav_dir}#${noise_wav_dir}#g" ${data}/${ddir}/wav.scp \
          > ${data}/${ddir}/noise1.scp
      fi


      # NOTE: modified to do dereverberation and separation
      if [[ "$cond" = "reverb" ]]; then
        # make anechoic spk scp files
        spk1_wav_dir=${rootdir}
        sed -e "s#${mixwav_dir}#${spk1_wav_dir}#g" ${data}/${ddir}/wav.scp \
          > ${data}/${ddir}/spk1.scp
        if [[ "$mixtype" != "single" ]]; then
          spk2_wav_dir=${rootdir}
          sed -e "s#${mixwav_dir}#${spk2_wav_dir}#g" ${data}/${ddir}/wav.scp \
            > ${data}/${ddir}/spk2.scp
        fi

        # reverb scps
        spk1_wav_dir=${rootdir}
        sed -e "s#${mixwav_dir}#${spk1_wav_dir}#g" ${data}/${ddir}/wav.scp \
          > ${data}/${ddir}/spk1_reverb.scp
        if [[ "$mixtype" != "single" ]]; then
          spk2_wav_dir=${rootdir}
          sed -e "s#${mixwav_dir}#${spk2_wav_dir}#g" ${data}/${ddir}/wav.scp \
            > ${data}/${ddir}/spk2_reverb.scp
        fi
      else
        # original code
        spk1_wav_dir=${rootdir}
        sed -e "s#${mixwav_dir}#${spk1_wav_dir}#g" ${data}/${ddir}/wav.scp \
          > ${data}/${ddir}/spk1.scp
        if [[ "$mixtype" != "single" ]]; then
          spk2_wav_dir=${rootdir}
          sed -e "s#${mixwav_dir}#${spk2_wav_dir}#g" ${data}/${ddir}/wav.scp \
            > ${data}/${ddir}/spk2.scp
        fi
      fi

      if [[ "$cond" = "reverb" ]]; then
        anechoic_mixwav_dir=${rootdir}
        sed -e "s#${mixwav_dir}#${anechoic_mixwav_dir}#g" ${data}/${ddir}/wav.scp \
          > ${data}/${ddir}/dereverb1.scp
      fi

      # base npy scps (always generated)
      s1_base_npz_dir=${npz_root}/wav${sample_rate}/${min_or_max}/${x}/s1_base_npz
      s2_base_npz_dir=${npz_root}/wav${sample_rate}/${min_or_max}/${x}/s2_base_npz
      awk -v dir="${s1_base_npz_dir}" -v suffix="${cond}" -F "," \
        'NR>1 {sub(/\.wav$/, "", $1); split($1, lst, "_"); spk=substr(lst[1],1,3)"_"substr(lst[3],1,3); print(spk "_" $1 "_" suffix, dir "/" $1 ".npy")}' \
        ${whamr_script_dir}/data/mix_2_spk_filenames_${x}.csv | sort > ${data}/${ddir}/spk1_base_npz.scp
      awk -v dir="${s2_base_npz_dir}" -v suffix="${cond}" -F "," \
        'NR>1 {sub(/\.wav$/, "", $1); split($1, lst, "_"); spk=substr(lst[1],1,3)"_"substr(lst[3],1,3); print(spk "_" $1 "_" suffix, dir "/" $1 ".npy")}' \
        ${whamr_script_dir}/data/mix_2_spk_filenames_${x}.csv | sort > ${data}/${ddir}/spk2_base_npz.scp

      s1_temp_npz_dir=${npz_root}/wav${sample_rate}/${min_or_max}/${x}/s1_temp_npz
      s2_temp_npz_dir=${npz_root}/wav${sample_rate}/${min_or_max}/${x}/s2_temp_npz
      noise_base_npz_dir=${npz_root}/wav${sample_rate}/${min_or_max}/${x}/noise_base_npz
      rir_npz_dir=${npz_root}/wav${sample_rate}/${min_or_max}/${x}/rir_npz
      room_param_npz_dir=${npz_root}/wav${sample_rate}/${min_or_max}/${x}/room_param_npz
      awk -v dir="${s1_temp_npz_dir}" -v suffix="${cond}" -F "," \
        'NR>1 {sub(/\.wav$/, "", $1); split($1, lst, "_"); spk=substr(lst[1],1,3)"_"substr(lst[3],1,3); print(spk "_" $1 "_" suffix, dir "/" $1 ".npy")}' \
        ${whamr_script_dir}/data/mix_2_spk_filenames_${x}.csv | sort > ${data}/${ddir}/spk1_temp_npz.scp
      awk -v dir="${s2_temp_npz_dir}" -v suffix="${cond}" -F "," \
        'NR>1 {sub(/\.wav$/, "", $1); split($1, lst, "_"); spk=substr(lst[1],1,3)"_"substr(lst[3],1,3); print(spk "_" $1 "_" suffix, dir "/" $1 ".npy")}' \
        ${whamr_script_dir}/data/mix_2_spk_filenames_${x}.csv | sort > ${data}/${ddir}/spk2_temp_npz.scp
      awk -v dir="${noise_base_npz_dir}" -v suffix="${cond}" -F "," \
        'NR>1 {sub(/\.wav$/, "", $1); split($1, lst, "_"); spk=substr(lst[1],1,3)"_"substr(lst[3],1,3); print(spk "_" $1 "_" suffix, dir "/" $1 ".npy")}' \
        ${whamr_script_dir}/data/mix_2_spk_filenames_${x}.csv | sort > ${data}/${ddir}/noise_base_npz.scp
      awk -v dir="${rir_npz_dir}" -v suffix="${cond}" -F "," \
        'NR>1 {sub(/\.wav$/, "", $1); split($1, lst, "_"); spk=substr(lst[1],1,3)"_"substr(lst[3],1,3); print(spk "_" $1 "_" suffix, dir "/" $1 ".npz")}' \
        ${whamr_script_dir}/data/mix_2_spk_filenames_${x}.csv | sort > ${data}/${ddir}/rir_npz.scp
      awk -v dir="${room_param_npz_dir}" -v suffix="${cond}" -F "," \
        'NR>1 {sub(/\.wav$/, "", $1); split($1, lst, "_"); spk=substr(lst[1],1,3)"_"substr(lst[3],1,3); print(spk "_" $1 "_" suffix, dir "/" $1 ".npz")}' \
        ${whamr_script_dir}/data/mix_2_spk_filenames_${x}.csv | sort > ${data}/${ddir}/room_param_npz.scp
    done
  done
done


# transcriptions (only for 'max' version)
if [[ "$min_or_max" = "min" ]]; then
  exit 0
fi

# rm -r tmp/ 2>/dev/null
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

# change to the original path
cd ..

for mixtype in both clean single; do
  for cond in anechoic reverb; do
    tr=tr_mix_${mixtype}_${cond}_${min_or_max}_${sample_rate}
    cv=cv_mix_${mixtype}_${cond}_${min_or_max}_${sample_rate}
    tt=tt_mix_${mixtype}_${cond}_${min_or_max}_${sample_rate}
    awk '(ARGIND==1) {txt[$1]=$0} (ARGIND==2) {split($1, lst, "_"); utt1=lst[3]; text=txt[utt1]; print($1, text)}' tmp/si_tr_s.txt ${data}/${tr}/wav.scp | awk '{$2=""; print $0}' > ${data}/${tr}/text_spk1
    awk '(ARGIND==1) {txt[$1]=$0} (ARGIND==2) {split($1, lst, "_"); utt2=lst[5]; text=txt[utt2]; print($1, text)}' tmp/si_tr_s.txt ${data}/${tr}/wav.scp | awk '{$2=""; print $0}' > ${data}/${tr}/text_spk2
    awk '(ARGIND==1) {txt[$1]=$0} (ARGIND==2) {split($1, lst, "_"); utt1=lst[3]; text=txt[utt1]; print($1, text)}' tmp/si_tr_s.txt ${data}/${cv}/wav.scp | awk '{$2=""; print $0}' > ${data}/${cv}/text_spk1
    awk '(ARGIND==1) {txt[$1]=$0} (ARGIND==2) {split($1, lst, "_"); utt2=lst[5]; text=txt[utt2]; print($1, text)}' tmp/si_tr_s.txt ${data}/${cv}/wav.scp | awk '{$2=""; print $0}' > ${data}/${cv}/text_spk2
    awk '(ARGIND<=2) {txt[$1]=$0} (ARGIND==3) {split($1, lst, "_"); utt1=lst[3]; text=txt[utt1]; print($1, text)}' tmp/si_dt_05.txt tmp/si_et_05.txt ${data}/${tt}/wav.scp | awk '{$2=""; print $0}' > ${data}/${tt}/text_spk1
    awk '(ARGIND<=2) {txt[$1]=$0} (ARGIND==3) {split($1, lst, "_"); utt2=lst[5]; text=txt[utt2]; print($1, text)}' tmp/si_dt_05.txt tmp/si_et_05.txt ${data}/${tt}/wav.scp | awk '{$2=""; print $0}' > ${data}/${tt}/text_spk2

    if [[ "$mixtype" = "single" ]]; then
      for x in "${tr}" "${cv}" "${tt}"; do
        rm ${data}/${x}/text_spk2
        mv ${data}/${x}/text_spk1 ${data}/${x}/text
      done
    fi
  done
done
rm -r tmp

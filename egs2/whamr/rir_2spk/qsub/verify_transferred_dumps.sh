#!/usr/bin/env bash
# Submit from rir_2spk after both rsync transfers finish.
#$ -cwd
#$ -l cpu_4=1
#$ -l h_rt=02:00:00
#$ -p -5
#$ -N rir_dump_verify
#$ -o qsub_logs
#$ -e qsub_logs
set -e
recipe=/gs/bs/tga-shinoda/nitsu/research/tf-locoformer/espnet/egs2/whamr/rir_2spk
cd "${recipe}"
source /gs/bs/tga-shinoda/nitsu/anaconda3/etc/profile.d/conda.sh
conda activate tf-locoformer
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
nf_incoming=${recipe}/.dump_nf_2spk_16k_min.incoming_20261008
but_parent=${recipe}/../rir_1ch
but_incoming=${but_parent}/.dump_but_2spk.incoming_20261008
nf_final=${recipe}/dump_nf_2spk_16k_min
but_final=${but_parent}/dump_but_2spk
logs=${recipe}/exp/tsubame_transfer_20261008
mkdir -p "${logs}"
trap 'code=$?; echo "$code" > "$logs/verification_exit_status"; date -Is' EXIT
date -Is
hostname
git rev-parse HEAD
[[ ! -e ${nf_final} && ! -L ${nf_final} && ! -e ${but_final} && ! -L ${but_final} ]]
python local/portable_rir_dump.py verify --root "${nf_incoming}" --workers 4
python local/portable_rir_dump.py verify --root "${but_incoming}" --workers 4
# Publish only after all files in BOTH dumps passed full SHA256/header checks.
[[ ! -e ${nf_final} && ! -L ${nf_final} && ! -e ${but_final} && ! -L ${but_final} ]]
mv -T --no-clobber "${nf_incoming}" "${nf_final}"
[[ ! -d ${nf_incoming} ]]
mv -T --no-clobber "${but_incoming}" "${but_final}"
[[ ! -d ${but_incoming} ]]
if [[ ! -e ${recipe}/dump_but_2spk && ! -L ${recipe}/dump_but_2spk ]]; then
    ln -s ../rir_1ch/dump_but_2spk "${recipe}/dump_but_2spk"
fi
python local/prepare_two_speaker_dump.py --output "${nf_final}"
python local/prepare_input_shapes.py \
    --config conf/tuning/train_pooled_bimamba_2spk_nf_16k_sweep_v2.yaml \
    --dump "${nf_final}" --output "${logs}/verified_input_shapes"
echo 'Both portable dumps published and NF input shapes verified'

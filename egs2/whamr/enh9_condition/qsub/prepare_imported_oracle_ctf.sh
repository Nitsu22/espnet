#!/usr/bin/env bash
# Relocate transferred CTFs and verify all exact DAI-8 audio bytes on CPU.
#$ -cwd
#$ -l cpu_8=1
#$ -l h_rt=01:00:00
#$ -p -5
#$ -N oracle_ctf_prep
#$ -o qsub_logs
#$ -e qsub_logs
set -e
source /gs/bs/tga-shinoda/nitsu/anaconda3/etc/profile.d/conda.sh
conda activate tf-locoformer
set -euo pipefail
cd "${SGE_O_WORKDIR:?Submit from enh9_condition}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="${PWD}/../../..${PYTHONPATH:+:${PYTHONPATH}}"
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/oracle_ctf_numba_${JOB_ID}"
shared=/gs/bs/tga-shinoda/nitsu/research/tf-locoformer/espnet
source_dump=${1:?Pass the imported dump directory}
python local/prepare_oracle_ctf_dump.py --mode relocate \
    --source-dump "${source_dump}" \
    --baseline-dump "${shared}/egs2/whamr/enh4_clean/dump_clean" \
    --baseline-recipe "${shared}/egs2/whamr/enh7_baseline" \
    --output dump_oracle_ctf_nf8k --workers 8
python local/check_oracle_ctf_inputs.py --dump dump_oracle_ctf_nf8k

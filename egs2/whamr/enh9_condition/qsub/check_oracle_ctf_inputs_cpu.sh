#!/usr/bin/env bash
# Revalidate a published dump without copying or recomputing CTFs.
#$ -cwd
#$ -l cpu_4=1
#$ -l h_rt=00:30:00
#$ -p -5
#$ -N oracle_ctf_inputs
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
python local/check_oracle_ctf_inputs.py --dump dump_oracle_ctf_nf8k

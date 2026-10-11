#!/usr/bin/env bash
# Resource/time selection and dependencies are recorded by the controller.
#$ -cwd
#$ -o qsub_logs
#$ -e qsub_logs
#$ -p -5
set -e
source /gs/bs/tga-shinoda/nitsu/anaconda3/etc/profile.d/conda.sh
conda activate tf-locoformer
set -euo pipefail
cd "${SGE_O_WORKDIR:?Submit from enh9_condition}"
export PATH="/apps/t4/rhel9/uge/latest/bin/lx-amd64:${PATH}"
command -v qsub qacct >/dev/null
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="${PWD}/../../..${PYTHONPATH:+:${PYTHONPATH}}"
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/oracle_pipeline_numba_${JOB_ID}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/oracle_pipeline_mpl_${JOB_ID}"
case " $* " in
    *" --phase train "*|*" --phase infer "*) module load cuda/11.8.0 ;;
esac
python local/oracle_ctf_pipeline.py "$@"

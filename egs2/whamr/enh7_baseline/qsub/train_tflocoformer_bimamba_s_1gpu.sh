#!/usr/bin/env bash
# Submit from enh7_baseline in an isolated checkout of the BiMamba branch.
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=24:00:00
#$ -N tfs_bimamba_wh8
#$ -o qsub_logs
#$ -e qsub_logs
#$ -p -5

set -eo pipefail
. /gs/bs/tga-shinoda/nitsu/anaconda3/etc/profile.d/conda.sh
conda activate tf-locoformer
module load cuda/11.8.0
set -u
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

# Reuse the verified, isolated official CUDA wheels; do not modify conda.
mamba_deps=${1:?Pass the existing Mamba dependency directory}
test -d "${mamba_deps}/mamba_ssm"
export PYTHONPATH="${PWD}/../../..:${mamba_deps}${PYTHONPATH:+:${PYTHONPATH}}"
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/nitsu_tfs_bimamba_numba_${JOB_ID}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/nitsu_tfs_bimamba_mpl_${JOB_ID}"
mkdir -p "${NUMBA_CACHE_DIR}" "${MPLCONFIGDIR}"
printf 'Commit: %s\n' "$(git rev-parse HEAD)"

python - <<'PY'
import torch
from mamba_ssm import Mamba

if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise RuntimeError("Expected exactly one allocated CUDA device")
print("CUDA:", torch.__version__, torch.cuda.get_device_name(0), flush=True)
PY

python -m unittest \
    test.espnet2.enh.separator.test_tflocoformer_bimamba.TestBiMamba.test_cuda_matches_reference_outputs_and_gradients -v

expdir=exp/enh_train_tflocoformer_bimamba_s_whamr_8k_1gpu_batch4
config=conf/tuning/train_enh_tflocoformer_bimamba_s_whamr_8k.yaml
python local/check_baseline_inputs.py \
    --condition whamr --expdir "${expdir}" --config "${config}"

bash run_whamr_tflocoformer_bimamba_s.sh \
    --stage 6 --stop_stage 6 --ngpu 1 --batch_size 4 --accum_grad 1 \
    --valid_batch_size 1 --enh_exp "${expdir}"

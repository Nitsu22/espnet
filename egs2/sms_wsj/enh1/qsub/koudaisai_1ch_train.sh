#!/usr/bin/env bash
#$ -cwd
#$ -l node_f=1
#$ -N koudai1ch_train
#$ -o qsub_logs/
#$ -e qsub_logs/
# Supply h_rt at submission after reviewing the measured preparation timings.
set -euo pipefail
source /etc/profile.d/modules.sh
module load cuda/11.8.0
source /gs/bs/tga-shinoda/nitsu/anaconda3/etc/profile.d/conda.sh
conda activate tf-locoformer
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
export PYTHONPATH="$(cd ../../.. && pwd)${PYTHONPATH:+:${PYTHONPATH}}"
batch_size=${KOUDAISAI_BATCH_SIZE:-4}
test "$batch_size" -gt 0
test "$((batch_size % 4))" -eq 0
for split in train valid; do
    test -s "exp_tfgridnet_koudaisai_1ch/enh_stats_8k/${split}/speech_mix_shape"
done
# Check the four-GPU path immediately before the separately authorized training.
python -m torch.distributed.run --standalone --nproc_per_node=4 \
    local/check_tfgridnet_koudaisai.py --ddp --batch-size-per-gpu "$((batch_size / 4))" \
    > exp_tfgridnet_koudaisai_1ch/smoke_ddp.log 2>&1
bash run_tfgridnet_koudaisai_1ch.sh --stage 6 --stop_stage 6 \
    --enh_args "--batch_size ${batch_size}"

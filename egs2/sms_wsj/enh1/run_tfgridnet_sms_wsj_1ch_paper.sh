#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
repo_root=$(cd ../../.. && pwd)
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
export NUMBA_CACHE_DIR="/tmp/nitsu_tfgridnet_numba"
export OMP_NUM_THREADS=1
export PATH="/home/kslab/nitsu/.conda/envs/tf-locoformer/bin:${PATH}"

# Only metadata under this private dump is writable; audio stays in Roland's dump.
exec bash enh.sh \
  --stage 5 --stop_stage 6 \
  --ngpu 4 --num_nodes 1 --nj 8 \
  --python /home/kslab/nitsu/.conda/envs/tf-locoformer/bin/python \
  --train_set train_si284_directpath \
  --valid_set cv_dev93_directpath \
  --test_sets test_eval92_directpath \
  --fs 8k --ref_num 2 --audio_format wav \
  --dumpdir "${TFGRIDNET_DUMP_DIR:-dump_roland_sms_wsj_1ch}" \
  --expdir exp_tfgridnet_sms_wsj_1ch_paper \
  --enh_exp exp_tfgridnet_sms_wsj_1ch_paper/tfgridnet_1ch_scratch \
  --enh_config conf/tuning/train_tfgridnet_sms_wsj_1ch_paper.yaml \
  --use_noise_ref false --use_dereverb_ref false \
  --inference_model valid.loss.ave_5best.pth \
  "$@"

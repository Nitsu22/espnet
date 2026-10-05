#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
repo_root=$(cd ../../.. && pwd)
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
python_bin=${KOUDAISAI_PYTHON:-/gs/bs/tga-shinoda/nitsu/anaconda3/envs/tf-locoformer/bin/python}
dump_dir=/gs/bs/tga-shinoda/nitsu/data/sms_wsj_4mic_koudaisai/dump_4mic
test -f "${dump_dir}/COPY_VERIFIED.json"
# Defaults stop before training. Explicit --stage 6 --stop_stage 6 starts it.
exec bash enh.sh --stage 5 --stop_stage 5 \
    --ngpu 4 --num_nodes 1 --nj 8 --python "${python_bin}" \
    --train_set train_si284_4mic --valid_set cv_dev93_4mic \
    --test_sets test_eval92_4mic --fs 8k --ref_num 2 --ref_channel 0 \
    --audio_format wav --dumpdir "${dump_dir}" \
    --expdir exp_tfgridnet_koudaisai_1ch \
    --enh_exp exp_tfgridnet_koudaisai_1ch/tfgridnet_2block_1ch \
    --enh_config conf/tuning/train_tfgridnet_koudaisai_1ch.yaml \
    --use_noise_ref false --use_dereverb_ref false \
    --inference_model 30epoch.pth "$@"

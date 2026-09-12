#!/usr/bin/env bash
set -euo pipefail
. ./path.sh
python local/prepare_small_bigdeltanet_smoke.py
smoke_exp=exp/enh_train_enh_tflocoformer_small_nocashe_bigdeltanet_4gpu_smoke
mkdir -p "${smoke_exp}"
args=()
for split in train valid; do
    for entry in 'wav.scp speech_mix' 'spk1.scp speech_ref1' 'spk2.scp speech_ref2'; do
        read -r scp name <<< "${entry}"
        args+=("--${split}_data_path_and_name_and_type" "dump/bigdeltanet_4gpu_smoke/${split}/${scp},${name},sound")
        args+=("--${split}_shape_file" "dump/bigdeltanet_4gpu_smoke/${split}/${name}_shape")
    done
done
python -m espnet2.bin.launch \
    --cmd run.pl --log "${smoke_exp}/train.log" --ngpu 4 --num_nodes 1 \
    --init_file_prefix "${smoke_exp}/.dist_init_" --multiprocessing_distributed true -- \
    python -m espnet2.bin.enh_train \
    --config conf/tuning/train_enh_tflocoformer_small_nocashe_bigdeltanet.yaml \
    --output_dir "${smoke_exp}" --ngpu 4 --resume false \
    --fold_length 80000 --fold_length 80000 --fold_length 80000 \
    --max_epoch 1 --num_workers 0 --num_att_plot 0 --log_interval 1 \
    "${args[@]}"
touch "${smoke_exp}/SUCCESS"

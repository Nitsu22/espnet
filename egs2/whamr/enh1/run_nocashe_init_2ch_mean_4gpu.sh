#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

min_or_max=min # "min" or "max". This is to determine how the mixtures are generated in local/data.sh.
sample_rate=8k

train_set=tr_mix_both_reverb_${min_or_max}_${sample_rate}
valid_set=cv_mix_both_reverb_${min_or_max}_${sample_rate}
test_sets="tt_mix_both_reverb_${min_or_max}_${sample_rate}"

# 2ch (multi-channel) で学習したモデルで 1ch (single-channel) モデルを初期化する。
# - モデルの差分は separator の先頭 conv (separator.conv.0) の入力チャネル数のみ:
#   2ch側: in_channels = 2 * n_imics (real/imag を concat するため)
#   1ch側: in_channels = 2
# - そこで 2ch側の conv 重みを「mic 方向平均」で 1ch 側へ縮約してから --init_param でロードする。
#
# NOTE:
# tflocoformer_nocashe_mc は入力を torch.cat((real, imag), dim=1) で作るため、
# in_channels の並びは [real_m0..real_m{M-1}, imag_m0..imag_m{M-1}] になる。
# 従って、平均は real/imag を混ぜずに別々に行う:
#   dst[:, 0] = mean(src[:, 0:M])   (real)
#   dst[:, 1] = mean(src[:, M:2M])  (imag)
init_param_src_path="exp/enh_train_enh_tflocoformer_nocashe_2ch_4gpu/valid.loss.best.pth"
init_param_path="${init_param_src_path%.pth}.convavg_to_1ch.pth"

if [ ! -f "${init_param_path}" ]; then
    # Make sure torch is available (same as enh.sh does internally).
    . ./path.sh
    INIT_PARAM_SRC_PATH="${init_param_src_path}" INIT_PARAM_DST_PATH="${init_param_path}" \
        python3 - <<'PY'
import os
import torch

src_path = os.environ["INIT_PARAM_SRC_PATH"]
dst_path = os.environ["INIT_PARAM_DST_PATH"]
module_key = "separator.conv.0"

state = torch.load(src_path, map_location="cpu")
weight_key = module_key + ".weight"
bias_key = module_key + ".bias"
if weight_key not in state:
    raise KeyError(f"'{weight_key}' not found in state_dict: {src_path}")
if bias_key not in state:
    raise KeyError(f"'{bias_key}' not found in state_dict: {src_path}")

w = state[weight_key]
b = state[bias_key]
if w.ndim != 4:
    raise ValueError(f"Expected 4D conv weight, but got shape={tuple(w.shape)}")

out_ch, in_ch, k_h, k_w = w.shape
if in_ch % 2 != 0:
    raise ValueError(f"Expected even in_channels (real+imag), but got shape={tuple(w.shape)}")
m = in_ch // 2
if m < 1:
    raise ValueError(f"Invalid in_channels: {in_ch}")

# Average across microphone channels, separately for real and imag.
w_real = w[:, :m, :, :].mean(dim=1)
w_imag = w[:, m:, :, :].mean(dim=1)
w_new = torch.stack([w_real, w_imag], dim=1)  # (out_ch, 2, k_h, k_w)

state[weight_key] = w_new
state[bias_key] = b

torch.save(state, dst_path)
print(f"[INFO] Saved conv-averaged checkpoint: {dst_path}")
print(f"[INFO] {weight_key}: {tuple(w.shape)} -> {tuple(w_new.shape)} (M={m})")
PY
fi

init_param="${init_param_path}"

CUDA_VISIBLE_DEVICES=0,1,2,3 ./enh.sh \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --fs ${sample_rate} \
    --ngpu 4 \
    --ref_num 2 \
    --local_data_opts "--sample_rate ${sample_rate} --min_or_max ${min_or_max}" \
    --enh_config ./conf/tuning/train_enh_tflocoformer_nocashe.yaml \
    --enh_exp exp/enh_train_enh_tflocoformer_nocashe_2ch_init_mean_4gpu \
    --use_dereverb_ref false \
    --use_noise_ref false \
    --inference_model "valid.loss.best.pth" \
    --audio_format wav \
    --init_param "${init_param}" \
    --stage 6 \
    --stop_stage 8 \
    "$@"

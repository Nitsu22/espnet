module load cuda/11.8.0

conda activate tf-locoformer

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

CUDA_VISIBLE_DEVICES=7 ./enh_se_condition.sh \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --fs ${sample_rate} \
    --ngpu 1 \
    --ref_num 2 \
    --local_data_opts "--sample_rate ${sample_rate} --min_or_max ${min_or_max}" \
    --enh_config ./conf/tuning/ttrain_enh_tflocoformer_nocashe_se_aeaall_cold_lr3_resnet_256.yaml \
    --enh_exp exp/enh_train_enh_tflocoformer_nocashe_se_aeafusion_cold_lr3_resnet_256_4gpu \
    --use_dereverb_ref false \
    --use_noise_ref true \
    --inference_model "valid.loss.best.pth" \
    --audio_format wav \
    --gpu_inference true \
    --stage 7 \
    --stop_stage 8 \
    "$@"

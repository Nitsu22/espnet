# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

min_or_max=min # "min" or "max". This is to determine how the mixtures are generated in local/data.sh.
sample_rate=8k
ablation_mode=swap_rir # oracle | swap_rir | rand_sample | mean
ablation_npz_root=../se_npz/data
ablation_seed=1234
ablation_epoch=0
ablation_mean_embedding_path=
ablation_mean_source_mode=oracle
eval_dsets_mode=test_only # all | test_only

passthrough_opts=()
while [ $# -gt 0 ]; do
    case "$1" in
        --ablation_mode)
            ablation_mode="$2"; shift 2 ;;
        --ablation_mode=*)
            ablation_mode="${1#*=}"; shift ;;
        --ablation_npz_root)
            ablation_npz_root="$2"; shift 2 ;;
        --ablation_npz_root=*)
            ablation_npz_root="${1#*=}"; shift ;;
        --ablation_seed)
            ablation_seed="$2"; shift 2 ;;
        --ablation_seed=*)
            ablation_seed="${1#*=}"; shift ;;
        --ablation_epoch)
            ablation_epoch="$2"; shift 2 ;;
        --ablation_epoch=*)
            ablation_epoch="${1#*=}"; shift ;;
        --ablation_mean_embedding_path)
            ablation_mean_embedding_path="$2"; shift 2 ;;
        --ablation_mean_embedding_path=*)
            ablation_mean_embedding_path="${1#*=}"; shift ;;
        --ablation_mean_source_mode)
            ablation_mean_source_mode="$2"; shift 2 ;;
        --ablation_mean_source_mode=*)
            ablation_mean_source_mode="${1#*=}"; shift ;;
        --eval_dsets_mode)
            eval_dsets_mode="$2"; shift 2 ;;
        --eval_dsets_mode=*)
            eval_dsets_mode="${1#*=}"; shift ;;
        *)
            passthrough_opts+=("$1"); shift ;;
    esac
done

case "${ablation_mode}" in
    oracle|swap_rir|rand_sample|mean) ;;
    *)
        echo "Invalid --ablation_mode: ${ablation_mode}" >&2
        exit 2 ;;
esac

train_set=tr_mix_both_reverb_${min_or_max}_${sample_rate}
valid_set=cv_mix_both_reverb_${min_or_max}_${sample_rate}
test_sets="tt_mix_both_reverb_${min_or_max}_${sample_rate}"

mean_emb_opts=()
if [ -n "${ablation_mean_embedding_path}" ]; then
    mean_emb_opts+=(--ablation_mean_embedding_path "${ablation_mean_embedding_path}")
fi

CUDA_VISIBLE_DEVICES=7 ./enh_se_condition_ablation.sh \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --fs ${sample_rate} \
    --ngpu 1 \
    --ref_num 2 \
    --local_data_opts "--sample_rate ${sample_rate} --min_or_max ${min_or_max}" \
    --enh_config ./conf/tuning/ttrain_enh_tflocoformer_nocashe_se_aeaall_cold_lr3_resnet_256.yaml \
    --enh_exp exp/enh_train_enh_tflocoformer_nocashe_se_aeafusion_cold_lr3_resnet_256_4gpu_small \
    --use_dereverb_ref false \
    --use_noise_ref true \
    --inference_model "valid.loss.best.pth" \
    --audio_format wav \
    --gpu_inference true \
    --inference_tag "enhanced_ablation_${ablation_mode}" \
    --spatial_ablation_mode "${ablation_mode}" \
    --ablation_npz_root "${ablation_npz_root}" \
    --ablation_seed "${ablation_seed}" \
    --ablation_epoch "${ablation_epoch}" \
    --ablation_mean_source_mode "${ablation_mean_source_mode}" \
    --eval_dsets_mode "${eval_dsets_mode}" \
    "${mean_emb_opts[@]}" \
    --stage 7 \
    --stop_stage 8 \
    "${passthrough_opts[@]}"

# bash run_eval_ablation_small .sh --ablation_mode oracle
# bash run_eval_ablation_small.sh --ablation_mode swap_rir
# bash run_eval_ablation_small.sh --ablation_mode rand_sample
# bash run_eval_ablation_small.sh --ablation_mode mean

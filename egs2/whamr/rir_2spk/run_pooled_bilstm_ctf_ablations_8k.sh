#!/usr/bin/env bash
# Select one independent architecture/loss ablation; never launches a sweep.
set -euo pipefail
recipe=$(cd "$(dirname "$0")" && pwd)
if [[ $# -lt 1 ]]; then
    echo 'Usage: bash run_pooled_bilstm_ctf_ablations_8k.sh VARIANT [--print-config | recipe options]' >&2
    echo 'VARIANT: baseline, pre_last2_bilstm, rank32, speaker_query, rir_mse, or ARCH_rir_mse' >&2
    exit 2
fi
variant=$1
shift
prefix=train_pooled_bilstm_2spk_nf_8k_sweep_v2_pre8_post4
case "$variant" in
    baseline) config="${prefix}_time_l1_plateau_200ep.yaml" ;;
    pre_last2_bilstm|rank32|speaker_query|rir_mse|pre_last2_bilstm_rir_mse|rank32_rir_mse|speaker_query_rir_mse)
        config="${prefix}_${variant}.yaml" ;;
    *) echo "Unknown variant: $variant" >&2; exit 2 ;;
esac
config="$recipe/conf/tuning/$config"
[[ -f "$config" ]] || { echo "Missing config: $config" >&2; exit 1; }
if [[ ${1:-} == --print-config ]]; then
    [[ $# == 1 ]] || { echo '--print-config cannot be combined with recipe options' >&2; exit 2; }
    printf '%s\n' "$config"
    exit 0
fi

# Keep transient caches inside a dedicated, owned directory on authorized storage.
# A caller may override data/output locations through the existing recipe options.
runtime_parent=$(realpath -m "$recipe/exp")
case "$runtime_parent" in
    /net/midgar/work2/nitsu/*|/gs/bs/tga-shinoda/nitsu/*) ;;
    *) echo 'Run from an authorized nitsu checkout' >&2; exit 1 ;;
esac
[[ ! -e "$runtime_parent" || ( -d "$runtime_parent" && -O "$runtime_parent" ) ]] || exit 1
mkdir -p "$runtime_parent"
runtime_tmp=$(mktemp -d "$runtime_parent/.bilstm_ctf_ablation.XXXXXX")
trap 'rm -rf -- "$runtime_tmp"' EXIT
export TMPDIR="$runtime_tmp" PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export NUMBA_CACHE_DIR="$runtime_tmp/numba" XDG_CACHE_HOME="$runtime_tmp/cache"
export TRITON_CACHE_DIR="$runtime_tmp/triton" TORCH_EXTENSIONS_DIR="$runtime_tmp/torch_extensions"
export CUDA_CACHE_PATH="$runtime_tmp/cuda" TORCH_HOME="$runtime_tmp/torch"
export MPLCONFIGDIR="$runtime_tmp/matplotlib"
bash "$recipe/run_pooled_bimamba_2spk_nf_8k_sweep_v2.sh" \
    --rir_config "$config" --rir_tag "${prefix}_ablation_${variant}" "$@"

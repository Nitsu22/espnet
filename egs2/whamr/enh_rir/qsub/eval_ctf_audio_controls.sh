#!/usr/bin/env bash
# Submit three paid jobs with explicit resource/time overrides:
# prepare: cpu_4=1; optional third argument overrides the source audio dump.
# infer:   gpu_1=1, 01:00:00, hold_jid PREP_ID; args: infer OUTPUT PREP_ID
# score:   cpu_4=1, 01:00:00, hold_jid INFER_ID; args: score OUTPUT INFER_ID
#$ -cwd
#$ -p -5
#$ -N ctf_input_ctrl
#$ -o qsub_logs
#$ -e qsub_logs
set -eo pipefail
[[ $# -eq 2 || $# -eq 3 ]] || exit 2
mode=$1
output=$2
predecessor=
if [[ $# -eq 3 ]]; then predecessor=$3; fi
case $mode in prepare|infer|score) ;; *) exit 2 ;; esac
case $output in /gs/bs/tga-shinoda/nitsu/*/exp/*) ;; *) exit 2 ;; esac
cd "$SGE_O_WORKDIR"
logs="$PWD/exp/ctf_controls_test_jobs/job_$JOB_ID"
mkdir -p "$logs"
export TMPDIR="$logs/tmp"
mkdir -m 700 "$TMPDIR"
export NUMBA_CACHE_DIR="$TMPDIR/numba" TRITON_CACHE_DIR="$TMPDIR/triton"
export MPLCONFIGDIR="$TMPDIR/matplotlib" XDG_CACHE_HOME="$TMPDIR/xdg"
export PYTHONDONTWRITEBYTECODE=1
finish() {
    status=$?
    trap - EXIT
    printf '%s\n' "$status" > "$logs/exit_status"
    if [[ $TMPDIR == "$logs/tmp" && -d $TMPDIR && ! -L $TMPDIR ]]; then
        rm -r -- "$TMPDIR"
    fi
    date -Is
    exit "$status"
}
trap finish EXIT
date -Is
hostname
git rev-parse HEAD | tee "$logs/commit.txt"
if [[ $mode != prepare ]]; then
    [[ $predecessor =~ ^[0-9]+$ ]]
    [[ $(<"exp/ctf_controls_test_jobs/job_$predecessor/exit_status") == 0 ]]
    [[ $(git rev-parse HEAD) == $(<"exp/ctf_controls_test_jobs/job_$predecessor/commit.txt") ]]
fi
source /gs/bs/tga-shinoda/nitsu/anaconda3/etc/profile.d/conda.sh
conda activate tf-locoformer
if [[ $mode == infer ]]; then module load cuda/11.8.0; fi
set -u
shared=/gs/bs/tga-shinoda/nitsu/research/tf-locoformer/espnet
repo=$(cd ../../.. && pwd -P)
export PYTHONPATH="$repo:$shared/egs2/whamr/damsep_clean/.deps/mamba-cu118-torch21-py310"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
python="$CONDA_PREFIX/bin/python"
if [[ $mode == prepare ]]; then
    dump_root="$PWD/dump_nf8k_ctf_controls_seed0"
    if [[ -n $predecessor ]]; then dump_root=$predecessor; fi
    previous=/gs/bs/tga-shinoda/nitsu/research/tf-locoformer/espnet_nf8k_bimamba_concat_s/egs2/whamr/enh_rir
    model="$previous/exp/enh_train_tflocoformer_s_nf8k_bimamba_ctf_concat_seed0"
    "$python" -B local/evaluate_ctf_audio_controls.py prepare \
        --dump-root "$dump_root" \
        --original-test "$previous/exp/bimamba_nf8k_baseline_inputs/raw/tt_mix_clean_reverb_min_8k" \
        --original-recipe "$shared/egs2/whamr/enh7_baseline" \
        --model-exp "$model" --output "$output" \
        --baseline-score "$shared/egs2/whamr/enh7_baseline/exp/enh_train_tflocoformer_s_nf_8k_1gpu_batch4/enhanced_tt_mix_clean_reverb_min_8k/scoring" \
        --matched-score "$model/inference_normalized_tt/score"
else
    if [[ $mode == infer ]]; then
        "$python" -B -c 'import torch; assert torch.cuda.is_available(); print("Allocated CUDA:", torch.cuda.device_count(), torch.cuda.get_device_name(0))'
    else
        export CUDA_VISIBLE_DEVICES=''
    fi
    "$python" -B local/evaluate_ctf_audio_controls.py "$mode" --output "$output"
fi

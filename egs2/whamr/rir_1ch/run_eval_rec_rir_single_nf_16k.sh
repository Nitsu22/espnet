#!/usr/bin/env bash
set -euo pipefail
recipe=$(cd "$(dirname "$0")" && pwd)
cd "${recipe}"
python=/home/kslab/nitsu/.conda/envs/tf-locoformer/bin/python
rir_exp=exp/rir_train_rec_rir_single_nf_16k
model_file=
device=cuda:0
dataset=whamr
dumpdir=dump_nf_16k_min
output_dir=
. utils/parse_options.sh
[[ $# == 0 ]] || exit 2
case ${dataset} in
    whamr) data_dir="${dumpdir}/raw/tt_rir_single_nf_min_16k" ;;
    ace_clean) data_dir=dump_ace/raw/tt_ace_single_clean_reverb_min_16k ;;
    ace_noisy) data_dir=dump_ace/raw/tt_ace_single_noisy_reverb_min_16k ;;
    *) echo '--dataset must be whamr, ace_clean or ace_noisy' >&2; exit 2 ;;
esac
[[ -n ${model_file} ]] || model_file="${rir_exp}/valid.loss.best.pth"
[[ -n ${output_dir} ]] || output_dir="${rir_exp}/evaluation/${dataset}"
[[ ! -e ${output_dir} ]] || { echo 'Use a new output_dir' >&2; exit 1; }
export PYTHONPATH="${recipe}/../../..${PYTHONPATH:+:${PYTHONPATH}}"
export NUMBA_CACHE_DIR=${NUMBA_CACHE_DIR:-/tmp/nitsu-rir1ch-numba}
"${python}" -m espnet2.bin.rec_rir_inference \
    --train_config "${rir_exp}/config.yaml" --model_file "${model_file}" \
    --wav_scp "${data_dir}/wav.scp" --output_dir "${output_dir}/rir" \
    --sample_rate 16000 --rir_length 32000 --device "${device}"
"${python}" local/score_ace_rir.py --pred-scp "${output_dir}/rir/wav.scp" \
    --ref-scp "${data_dir}/rir_ref.scp" --output-dir "${output_dir}/score" --channel 0

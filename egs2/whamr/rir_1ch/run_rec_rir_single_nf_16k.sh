#!/usr/bin/env bash
set -euo pipefail
recipe=$(cd "$(dirname "$0")" && pwd)
cd "${recipe}"
stage=1
stop_stage=6
resume=false
ngpu=1
num_nodes=1
nj=4
python=/home/kslab/nitsu/.conda/envs/tf-locoformer/bin/python
dumpdir=dump_nf_16k_min
rir_config=conf/tuning/train_rec_rir_single_nf_16k.yaml
rir_tag=train_rec_rir_single_nf_16k
rir_exp=
rir_args=
rir_model_type=rec_rir
. utils/parse_options.sh
[[ $# == 0 ]] || { echo 'Unexpected positional arguments' >&2; exit 2; }
[[ -n ${rir_exp} ]] || rir_exp="exp/rir_${rir_tag}"
case ${resume} in true|false) ;; *) exit 2 ;; esac
if [[ ${stage} -le 1 && ${stop_stage} -ge 1 ]]; then
    "${python}" local/prepare_single_speaker_dump.py --source data_rir_plus --output "${dumpdir}"
fi
if [[ ${stop_stage} -lt 5 ]]; then exit 0; fi
[[ -f ${dumpdir}/preparation.json ]] || { echo 'Run Stage 1 to prepare the dump first' >&2; exit 1; }
if [[ ${stage} -le 6 && ${stop_stage} -ge 6 ]]; then
    if [[ ${resume} == false && -e ${rir_exp}/checkpoint.pth ]]; then
        echo 'Existing checkpoint: use --resume true or a new --rir_exp' >&2; exit 1
    fi
    if [[ ${resume} == true && ! -f ${rir_exp}/checkpoint.pth ]]; then
        echo 'Cannot resume without checkpoint.pth' >&2; exit 1
    fi
fi
export NUMBA_CACHE_DIR=${NUMBA_CACHE_DIR:-/tmp/nitsu-rir1ch-numba}
bash ./rir.sh --stage "${stage}" --stop_stage "${stop_stage}" \
    --skip_data_prep true --resume "${resume}" --python "${python}" \
    --fs 16k --ngpu "${ngpu}" --num_nodes "${num_nodes}" --nj "${nj}" \
    --dumpdir "${dumpdir}" --train_set tr_rir_single_nf_min_16k \
    --valid_set cv_rir_single_nf_min_16k --test_sets tt_rir_single_nf_min_16k \
    --rir_model_type "${rir_model_type}" --rir_config "${rir_config}" --rir_tag "${rir_tag}" \
    --rir_exp "${rir_exp}" --rir_stats_dir "exp/rir_stats_${rir_tag}" \
    --speech_fold_length 64000 --rir_fold_length 32000 --rir_args "${rir_args}"

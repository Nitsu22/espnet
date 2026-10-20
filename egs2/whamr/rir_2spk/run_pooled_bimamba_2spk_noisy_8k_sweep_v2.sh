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
python=python3
source_data=data_rir_plus
dumpdir=dump_noisy_2spk_8k_min
rir_config=conf/tuning/train_pooled_bimamba_2spk_noisy_8k_sweep_v2.yaml
rir_tag=train_pooled_bimamba_2spk_noisy_8k_sweep_v2
rir_exp=
rir_stats_dir=
. utils/parse_options.sh
[[ $# == 0 ]] || { echo 'Unexpected positional arguments' >&2; exit 2; }
[[ ${stage} -ge 1 && ${stage} -le ${stop_stage} && ${stop_stage} -le 6 ]]
case ${resume} in true|false) ;; *) exit 2 ;; esac
[[ -n ${rir_exp} ]] || rir_exp="exp/rir_${rir_tag}"
[[ -n ${rir_stats_dir} ]] || rir_stats_dir="exp/rir_stats_${rir_tag}"
if (( stage <= 1 )); then
    "${python}" local/prepare_two_speaker_dump.py --source "${source_data}" \
        --sample-rate 8k --noise --output "${dumpdir}"
fi
if (( stop_stage < 5 )); then exit 0; fi
"${python}" - "${dumpdir}/preparation.json" <<'PY'
import json, sys
spec = json.load(open(sys.argv[1]))['spec']
assert spec['noise'] is True and spec['sample_rate'] == '8k'
assert spec['length'] == 'min' and spec['num_spk'] == 2
PY
if (( stage <= 5 )); then
    "${python}" local/prepare_input_shapes.py --config "${rir_config}" \
        --dump "${dumpdir}" --output "${rir_stats_dir}"
fi
if (( stop_stage < 6 )); then exit 0; fi
if [[ ${resume} == false && -e ${rir_exp}/checkpoint.pth ]]; then
    echo 'Existing checkpoint: use --resume true or a new --rir_exp' >&2; exit 1
fi
if [[ ${resume} == true && ! -s ${rir_exp}/checkpoint.pth ]]; then
    echo 'Cannot resume without checkpoint.pth' >&2; exit 1
fi
bash ./rir.sh --stage 6 --stop_stage 6 --skip_data_prep true \
    --resume "${resume}" --python "${python}" --fs 8k --ngpu "${ngpu}" \
    --num_nodes "${num_nodes}" --nj "${nj}" --dumpdir "${dumpdir}" \
    --train_set tr_rir_2spk_noisy_min_8k --valid_set cv_rir_2spk_noisy_min_8k \
    --test_sets tt_rir_2spk_noisy_min_8k --rir_model_type pooled_bimamba_sweep_v2_pit \
    --rir_config "${rir_config}" --rir_tag "${rir_tag}" --rir_exp "${rir_exp}" \
    --rir_stats_dir "${rir_stats_dir}" --batch_by_input_only true \
    --speech_fold_length 80000 --rir_fold_length 16000 --rir_args '--valid_batch_size 1'

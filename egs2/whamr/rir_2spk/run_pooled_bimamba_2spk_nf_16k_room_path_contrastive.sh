#!/usr/bin/env bash
# Fixed paired dataset: 2 room items become 4 waveform views inside the model.
set -euo pipefail
recipe=$(cd "$(dirname "$0")" && pwd)
cd "${recipe}"
stage=5
stop_stage=6
resume=false
ngpu=1
python=/home/kslab/nitsu/.conda/envs/tf-locoformer/bin/python
dumpdir=/net/midgar/work2/nitsu/learning/tf-locoformer/espnet/egs2/whamr/rir_2spk/dump_nf_2spk_16k_min_contrastive
rir_config=conf/tuning/train_pooled_bimamba_2spk_nf_16k_room_path_contrastive.yaml
rir_tag=train_pooled_bimamba_2spk_nf_16k_room_path_contrastive
rir_exp=
rir_stats_dir=
. utils/parse_options.sh
[[ $# == 0 ]] || { echo 'Unexpected positional arguments' >&2; exit 2; }
[[ ${ngpu} == 1 ]] || { echo 'This paired contrastive recipe requires one GPU' >&2; exit 2; }
[[ -f ${dumpdir}/completion.json ]] || { echo 'Paired dump generation is incomplete' >&2; exit 1; }
[[ -n ${rir_exp} ]] || rir_exp="exp/rir_${rir_tag}"
[[ -n ${rir_stats_dir} ]] || rir_stats_dir="exp/rir_stats_${rir_tag}"
code_root=$(cd "${recipe}/../../.." && pwd)
export PYTHONPATH="${code_root}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "${rir_stats_dir}/runtime_cache"
runtime_cache=$(realpath "${rir_stats_dir}/runtime_cache")
export NUMBA_CACHE_DIR="${runtime_cache}/numba"
export MPLCONFIGDIR="${runtime_cache}/matplotlib"
if (( stage <= 1 && stop_stage >= 1 )); then
    "${python}" local/prepare_room_path_contrastive_dump.py --verify-existing "${dumpdir}"
fi
if (( stop_stage < 5 )); then exit 0; fi
if (( stage >= 6 )); then
    [[ -f ${rir_stats_dir}/train/speech_mix_shape && -f ${rir_stats_dir}/valid/speech_mix_shape ]] || {
        echo 'Run Stage 5 first to prepare paired input shapes' >&2; exit 1;
    }
fi
if (( stage < 5 )); then stage=5; fi
exec bash run_rec_rir_2spk_nf_16k.sh \
    --stage "${stage}" --stop_stage "${stop_stage}" \
    --dumpdir "${dumpdir}" --python "${python}" --ngpu "${ngpu}" \
    --resume "${resume}" --rir_model_type pooled_bimamba_room_path_contrastive \
    --rir_config "${rir_config}" --rir_tag "${rir_tag}" --rir_exp "${rir_exp}" \
    --rir_stats_dir "${rir_stats_dir}" --rir_args '--valid_batch_size 1'

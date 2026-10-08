#!/usr/bin/env bash
# Submit from egs2/whamr/rir_2spk: qsub -g tga-shinoda qsub/pooled_bimamba_ablations_stage5.sh
# Batch-by-input Stage 5 reuses verified lengths and needs no model forward pass.
#$ -cwd
#$ -l cpu_4=1
#$ -l h_rt=00:10:00
#$ -p -5
#$ -N bimamba_abl_stats
#$ -o qsub_logs
#$ -e qsub_logs

set -e
source /gs/bs/tga-shinoda/nitsu/anaconda3/etc/profile.d/conda.sh
conda activate tf-locoformer
set -euo pipefail
recipe=/gs/bs/tga-shinoda/nitsu/research/tf-locoformer/espnet/egs2/whamr/rir_2spk
cd "${recipe}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/rir2_bimamba_stage5_${JOB_ID}"
mkdir -p "${NUMBA_CACHE_DIR}"
job_logs="exp/pooled_bimamba_ablations_stage5/job_${JOB_ID}"
mkdir -p "${job_logs}"
trap 'code=$?; echo "$code" > "$job_logs/exit_status"; date -Is' EXIT
date -Is
hostname
git rev-parse HEAD
python_bin="${CONDA_PREFIX}/bin/python"
dump="${recipe}/dump_nf_2spk_16k_min"
# Check the verified portable dump before creating any batching artifacts.
"${python_bin}" local/prepare_two_speaker_dump.py --output "${dump}"
for variant in ffn_only freq_before_pool 2blocks bilstm; do
    echo "Stage 5: ${variant}"
    bash "run_pooled_bimamba_2spk_nf_16k_sweep_v2_${variant}.sh" \
        --stage 5 --stop_stage 5 --ngpu 0 --nj 4 \
        --python "${python_bin}" --dumpdir "${dump}" \
        --batch_by_input_only true > "${job_logs}/${variant}.log" 2>&1
    cat "${job_logs}/${variant}.log"
done
"${python_bin}" - "${job_logs}" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path.cwd()
report = dict(stage=5, batch_by_input_only=True, ngpu=0,
              job_id=os.environ['JOB_ID'],
              git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              variants={})
common_shapes = {}
for variant in ('ffn_only', 'freq_before_pool', '2blocks', 'bilstm'):
    tag = f'train_pooled_bimamba_2spk_nf_16k_sweep_v2_{variant}'
    stats = root / 'exp' / f'rir_stats_{tag}'
    config = root / 'conf/tuning' / f'{tag}.yaml'
    entry = dict(config=str(config), config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
                 stats=str(stats), splits={})
    shapes_report = json.loads((stats / 'input_shapes.json').read_text())
    for split, count in (('train', 20000), ('valid', 5000)):
        shape = (stats / split / 'speech_mix_shape').read_bytes()
        assert len(shape.splitlines()) == count, (variant, split)
        sha = hashlib.sha256(shape).hexdigest()
        assert shapes_report[split] == dict(count=count, sha256=sha)
        if split in common_shapes:
            assert common_shapes[split] == shape, (variant, split)
        else:
            common_shapes[split] = shape
        entry['splits'][split] = dict(count=count, sha256=sha)
    report['variants'][variant] = entry
output = Path(sys.argv[1]) / 'complete.json'
output.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
PY
echo 'Stage 5 completed for all four ablations; identical batching shapes verified'

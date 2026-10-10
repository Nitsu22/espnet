#!/usr/bin/env bash
# Recover only missing WHAMR predictions from an interrupted evaluation.
#$ -cwd
#$ -l gpu_h=1
#$ -l h_rt=00:15:00
#$ -p -5
#$ -N whamr_rir_resume
#$ -o qsub_logs
#$ -e qsub_logs
set -e
source /gs/bs/tga-shinoda/nitsu/anaconda3/etc/profile.d/conda.sh
conda activate tf-locoformer
module load cuda/11.8.0
set -euo pipefail
output=${1:?Specify the absolute existing WHAMR output directory}
[[ ${output} == /* ]]
cd "${SGE_O_WORKDIR:?}"
repo=$(git rev-parse --show-toplevel)
shared_recipe=/gs/bs/tga-shinoda/nitsu/research/tf-locoformer/espnet/egs2/whamr/rir_2spk
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="${repo}:${shared_recipe}/../damsep_clean/.deps/mamba-cu118-torch21-py310${PYTHONPATH:+:${PYTHONPATH}}"
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/whamr_resume_numba_${JOB_ID}"
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/whamr_resume_triton_${JOB_ID}"
mkdir -p "${NUMBA_CACHE_DIR}" "${TRITON_CACHE_DIR}"
trap 'status=$?; printf "%s\n" "$status" > "${output}/resume_exit_status"' EXIT
python - "${output}" "${JOB_ID}" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import soundfile as sf

root = Path(sys.argv[1])
metadata = json.loads((root / 'metadata.json').read_text())
assert metadata['count'] == 3000 and metadata['sample_rate'] == 16000
assert metadata['rir_length'] == 32000 and metadata['output_subtype'] == 'FLOAT'
assert metadata['inference'] == 'ordinary sweep PIM'
assert hashlib.sha256((root / 'valid.loss.best.pth').read_bytes()).hexdigest() == metadata['checkpoint_sha256']
rows = [line.split(maxsplit=1) for line in (root / 'wav.scp').read_text().splitlines()]
assert len(rows) == 3000 and len(dict(rows)) == 3000
assert not (root / 'inference_complete.json').exists(), 'Inference already complete'
predictions = {}
missing = []
def valid(path):
    try:
        header = sf.info(path)
        samples, rate = sf.read(path, dtype='float32')
        return (rate == 16000 and samples.shape == (32000,) and
                header.subtype == 'FLOAT' and np.isfinite(samples).all())
    except (OSError, RuntimeError):
        return False
for uid, wave in rows:
    assert Path(uid).name == uid
    pair = [root / 'inference' / f'rir{i}' / f'{uid}.wav' for i in (1, 2)]
    if all(valid(path) for path in pair):
        predictions[uid] = pair
    else:
        missing.append((uid, wave))
segment = root / f'resume_{sys.argv[2]}'
segment.mkdir(exist_ok=False)
remaining = segment / 'wav.scp'
remaining.write_text(''.join(f'{uid} {wave}\n' for uid, wave in missing))
print(f'Reusing {len(predictions)} verified pairs; inferring {len(missing)} remaining', flush=True)
start = time.monotonic()
if missing:
    subprocess.run([sys.executable, '-m', 'espnet2.bin.rec_rir_pit_inference',
                    '--train_config', str(root / 'config.yaml'),
                    '--model_file', str(root / 'valid.loss.best.pth'),
                    '--wav_scp', str(remaining), '--output_dir', str(segment / 'inference'),
                    '--sample_rate', '16000', '--rir_length', '32000',
                    '--output_subtype', 'FLOAT', '--device', 'cuda:0'], check=True)
    recovered = [line.split() for line in (segment / 'inference/rir.scp').read_text().splitlines()]
    assert len(recovered) == len(missing)
    assert {row[0] for row in recovered} == {uid for uid, _ in missing}
    for uid, first, second in recovered:
        pair = [Path(first), Path(second)]
        assert all(valid(path) for path in pair)
        predictions[uid] = pair
assert len(predictions) == 3000
temporary = root / 'inference/rir.scp.tmp'
temporary.write_text(''.join(f'{uid} {predictions[uid][0]} {predictions[uid][1]}\n' for uid, _ in rows))
temporary.replace(root / 'inference/rir.scp')
report = dict(count=3000, inference_seconds=time.monotonic()-start,
              timing_scope='resume segment only; preceding interrupted segment excluded',
              reused_pairs=3000-len(missing), resumed_pairs=len(missing),
              resume_job=sys.argv[2],
              resume_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              checkpoint_sha256=metadata['checkpoint_sha256'])
(root / 'inference_complete.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
PY

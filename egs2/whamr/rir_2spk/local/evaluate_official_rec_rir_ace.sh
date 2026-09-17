#!/usr/bin/env bash
# Run on an available lab GPU inside tmux. Stop before evaluation if parity fails.
set -euo pipefail
recipe=$(cd "$(dirname "$0")/.." && pwd)
repo=$(cd "${recipe}/../../.." && pwd)
official=${OFFICIAL_REPO:-/net/midgar/work2/nitsu/learning/Rec-RIR}
python=${RECRIR_PYTHON:-/home/kslab/nitsu/.conda/envs/recrir/bin/python}
output=${OUTPUT_DIR:-${recipe}/exp/official_rec_rir_ace_16k}
export PYTHONPATH="${repo}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TRITON_PTXAS_PATH=/home/kslab/nitsu/.conda/envs/recrir/libexec/recrir/ptxas-cu118
export TRITON_CACHE_DIR=/tmp/nitsu-recrir-ace-triton
export TORCHINDUCTOR_CACHE_DIR=/tmp/nitsu-recrir-ace-inductor
export NUMBA_CACHE_DIR=/tmp/nitsu-recrir-ace-numba
mkdir -p "${output}"
cd "${repo}"
for condition in clean noisy; do
    dataset="${recipe}/dump_ace/raw/tt_ace_single_${condition}_reverb_min_16k"
    # One complete utterance for every measured RIR in each noise condition.
    "${python}" - "${dataset}" "${output}/parity_${condition}.scp" <<'PY'
import sys
from pathlib import Path
root, out = map(Path, sys.argv[1:])
wavs = dict(line.split(maxsplit=1) for line in (root / 'wav.scp').read_text().splitlines())
seen = set()
with out.open('w') as stream:
    for line in (root / 'rir_ref.scp').read_text().splitlines():
        uid, rir = line.split(maxsplit=1)
        if rir not in seen:
            stream.write(f'{uid} {wavs[uid]}\n')
            seen.add(rir)
assert len(seen) == 14, len(seen)
PY
    "${python}" -m espnet2.bin.rec_rir_official_inference \
        --config "${official}/config/Rec-RIR.toml" \
        --checkpoint "${official}/ckpt/epoch35.tar" \
        --wav-scp "${output}/parity_${condition}.scp" \
        --output-dir "${output}/parity_${condition}" --parity-repo "${official}"
done
for condition in clean noisy; do
    dataset="${recipe}/dump_ace/raw/tt_ace_single_${condition}_reverb_min_16k"
    "${python}" -m espnet2.bin.rec_rir_official_inference \
        --config "${official}/config/Rec-RIR.toml" \
        --checkpoint "${official}/ckpt/epoch35.tar" \
        --wav-scp "${dataset}/wav.scp" --output-dir "${output}/${condition}"
    "${python}" "${recipe}/local/score_ace_rir.py" \
        --pred-scp "${output}/${condition}/wav.scp" \
        --ref-scp "${dataset}/rir_ref.scp" --output-dir "${output}/score_${condition}"
done
echo 'ACE evaluation complete'

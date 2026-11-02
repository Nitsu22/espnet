#!/usr/bin/env bash
# Submit infer on node_q, then score on cpu_4 with -hold_jid.
#$ -cwd
#$ -p -5
#$ -o qsub_logs
#$ -e qsub_logs
set -e
source /gs/bs/tga-shinoda/nitsu/anaconda3/etc/profile.d/conda.sh
conda activate tf-locoformer
set -euo pipefail
mode=${1:?infer or score}
case ${mode} in infer|score) ;; *) exit 2 ;; esac
cd "${SGE_O_WORKDIR:?Submit from enh_rir}"
case ${PWD} in /gs/bs/tga-shinoda/nitsu/*) ;; *) exit 2 ;; esac
[[ ${JOB_ID:?} =~ ^[0-9]+$ ]]
expdir=exp/enh_train_tflocoformer_s_whamr8k_bimamba_ctf_concat_seed0
pipeline=exp/bimamba_whamr8k_concat_eval_normalized
export WHAMR_EVAL_EXPDIR=${expdir}
logs=${PWD}/${pipeline}/job_${JOB_ID}
mkdir -p "${logs}"
task_tmp=$(mktemp -d "${logs}/tmp.XXXXXX")
export TMPDIR=${task_tmp} XDG_CACHE_HOME=${task_tmp}/cache
export NUMBA_CACHE_DIR=${task_tmp}/numba TRITON_CACHE_DIR=${task_tmp}/triton
export MPLCONFIGDIR=${task_tmp}/matplotlib PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
shared=/gs/bs/tga-shinoda/nitsu/research/tf-locoformer/espnet
export PYTHONPATH="${PWD}/../../..:${shared}/egs2/whamr/damsep_clean/.deps/mamba-cu118-torch21-py310${PYTHONPATH:+:${PYTHONPATH}}"
cleanup() {
    code=$?
    trap - EXIT
    printf '%s\n' "${code}" > "${logs}/exit_status"
    case ${task_tmp} in "${logs}"/tmp.*) rm -rf -- "${task_tmp}" ;; esac
    date -Is
    exit "${code}"
}
trap cleanup EXIT
date -Is
git rev-parse HEAD | tee "${logs}/commit.txt"
qstat -j "${JOB_ID}" > "${logs}/scheduler.txt" 2>&1 || true
test -s "${expdir}/valid.loss.best.pth"
grep -Eq '\[Early stopping\]|INFO: 150epoch results:' "${expdir}/train.log"
args=(--inference_model valid.loss.best.pth --normalize_output_wav true
      --inference_tag inference_normalized --python "${CONDA_PREFIX}/bin/python")
if [[ ${mode} == infer ]]; then
    module load cuda/11.8.0
    for split in cv tt; do
        test ! -e "${expdir}/inference_normalized_${split}"
    done
    sha256sum "${expdir}/valid.loss.best.pth" > "${pipeline}/model.sha256"
    bash run_whamr8k_bimamba_ctf_concat_s.sh --stage 7 --stop_stage 7 --ngpu 1 "${args[@]}"
else
    previous_job=${2:?inference job ID}
    [[ ${previous_job} =~ ^[0-9]+$ ]]
    test "$(<"${pipeline}/job_${previous_job}/exit_status")" = 0
    test "$(<"${pipeline}/inference_complete")" = "${previous_job}"
    sha256sum -c "${pipeline}/model.sha256"
fi
# Validate both speakers, exact UIDs, sample counts, rate, and finite unclipped WAVs.
python - <<'PY'
from pathlib import Path
import os
import numpy as np
import soundfile as sf

exp = Path(os.environ['WHAMR_EVAL_EXPDIR'])

def rows(path):
    pairs = [line.split(maxsplit=1) for line in path.read_text().splitlines() if line.strip()]
    assert all(len(pair) == 2 for pair in pairs), path
    mapping = dict(pairs)
    assert len(pairs) == len(mapping), f'Duplicate UID: {path}'
    return mapping

for split, count in (('cv', 5000), ('tt', 3000)):
    source = Path(f'exp/bimamba_whamr8k_baseline_inputs/raw/{split}_mix_both_reverb_min_8k')
    expected = rows(source / 'wav.scp')
    assert len(expected) == count, source
    for speaker in (1, 2):
        refs = rows(source / f'spk{speaker}.scp')
        output = rows(exp / f'inference_normalized_{split}/spk{speaker}.scp')
        assert output.keys() == expected.keys() == refs.keys()
        for uid, path in output.items():
            info = sf.info(path)
            assert info.samplerate == 8000 and info.channels == 1, uid
            assert info.frames == sf.info(refs[uid]).frames, uid
        for uid in sorted(output)[::max(1, count // 100)]:
            audio, _ = sf.read(output[uid])
            assert np.isfinite(audio).all() and np.max(np.abs(audio)) <= .901, uid
PY
if [[ ${mode} == infer ]]; then
    printf '%s\n' "${JOB_ID}" > "${pipeline}/inference_complete"
else
    bash run_whamr8k_bimamba_ctf_concat_s.sh --stage 8 --stop_stage 8 --ngpu 0 "${args[@]}"
    python - <<'PY'
from pathlib import Path
import json
import os
import numpy as np

exp = Path(os.environ['WHAMR_EVAL_EXPDIR'])
results = {}
for split, count in (('cv', 5000), ('tt', 3000)):
    source = Path(f'exp/bimamba_whamr8k_baseline_inputs/raw/{split}_mix_both_reverb_min_8k/wav.scp')
    expected = {line.split()[0] for line in source.read_text().splitlines()}
    results[split] = {}
    for metric in ('SI_SNR', 'STOI', 'ESTOI', 'SDR', 'SAR', 'SIR'):
        values = []
        for speaker in (1, 2):
            path = exp / f'inference_normalized_{split}/score/{metric}_spk{speaker}'
            pairs = [line.split() for line in path.read_text().splitlines()]
            assert len(pairs) == count and {p[0] for p in pairs} == expected, path
            scores = np.array([float(p[1]) for p in pairs])
            assert np.isfinite(scores).all(), path
            values.extend(scores)
        results[split][metric] = float(np.mean(values))
(exp / 'evaluation_normalized.json').write_text(json.dumps(results, indent=2) + '\n')
lines = ['# Noisy WHAMR 8 kHz BiMamba CTF concat S', '',
         'Model: valid.loss.best.pth; normalize_output_wav=true.', '',
         '| split | SI-SNR [dB] | SDR [dB] | SAR [dB] | SIR [dB] | STOI [%] |',
         '|---|---:|---:|---:|---:|---:|']
for split, scores in results.items():
    values = [scores[k] for k in ('SI_SNR', 'SDR', 'SAR', 'SIR', 'STOI')]
    lines.append('| ' + split + ' | ' + ' | '.join(f'{v:.3f}' for v in values) + ' |')
(exp / 'RESULTS_normalized.md').write_text('\n'.join(lines) + '\n')
PY
    printf '%s\n' "${JOB_ID}" > "${pipeline}/scoring_complete"
fi

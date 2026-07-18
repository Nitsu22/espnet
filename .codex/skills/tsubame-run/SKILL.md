---
name: tsubame-run
description: Run this ESPnet workspace's jobs on TSUBAME using qsub. Use when Codex needs to move committed midgar changes to TSUBAME, create or adapt qsub*.sh files, submit test or production jobs, or record concise reusable TSUBAME run knowledge for this repository.
---

# TSUBAME Run

## Scope

Use this skill only for Agents working in this ESPnet workspace. Keep it focused on the steps needed to run repository-created `run*.sh` scripts on TSUBAME via `qsub`.

Do not add broad TSUBAME documentation, unused scheduler notes, one-off experiment logs, or hyperparameter details. If a newly discovered TSUBAME detail will be reused for future qsub jobs in this repo, add it here briefly.

## Fixed Paths

- midgar workspace: `/net/midgar/work2/nitsu/learning/tf-locoformer/espnet`
- TSUBAME repository: `/gs/bs/tga-shinoda/nitsu/research/tf-locoformer/espnet`
- Existing qsub examples: `/gs/bs/tga-shinoda/nitsu/research/tf-locoformer/espnet/egs2/whamr/enh_rir/qsub*.sh`
- Main TSUBAME run directory for current work: `/gs/bs/tga-shinoda/nitsu/research/tf-locoformer/espnet/egs2/whamr/enh_rir`

## Workflow

1. On midgar, inspect the repository state before any run handoff.
   - Use `git status --short` and inspect relevant diffs.
   - Commit only the changes required for the requested TSUBAME run.
   - Do not commit unrelated dirty files. Ask the user if the commit scope is ambiguous.
   - Push the branch before moving to TSUBAME.

2. On TSUBAME, update the same repository.
   - Connect with `ssh tsubame`.
   - Run `cd /gs/bs/tga-shinoda/nitsu/research/tf-locoformer/espnet`.
   - Check `git status --short`; stop if there are unexpected local changes.
   - Pull the pushed commit, preferably with a fast-forward-only pull.
   - Treat TSUBAME as an execution environment only. Do not edit or commit tracked source/config files on TSUBAME.
   - If a tracked code/config change is needed, make it on midgar, commit it, push it, then pull it on TSUBAME.

3. Move to the recipe directory for the run.
   - For current `enh_rir` work, use `cd /gs/bs/tga-shinoda/nitsu/research/tf-locoformer/espnet/egs2/whamr/enh_rir`.
   - If the user targets another ESPnet recipe, use that recipe's directory but still use the closest working qsub example.

4. Create a qsub script from a nearby working example.
   - Prefer copying an existing `qsub*.sh` from the same recipe and same GPU scale.
   - Do not overwrite an existing qsub script without checking its contents.
   - Limit TSUBAME-side edits to qsub scripts and run-only artifacts unless the user explicitly asks otherwise.
   - Match the `run*.sh` command, `--ngpu`, `--stage`, `--stop_stage`, and optional args to the requested run.
   - Use the nearest successful qsub example for `node_f` versus `node_q`; do not guess resource names.
   - Existing examples use `module load cuda/11.8.0`, `conda activate tf-locoformer`, and conda under `/gs/bs/tga-shinoda/nitsu/anaconda3`.

5. Submit the job.
   - Test run: set `#$ -l h_rt=00:03:00` or less, then run `qsub qsub...sh`.
   - Production run: use the intended walltime and run `qsub -g tga-shinoda qsub...sh`.
   - Do not add `-g tga-shinoda` for the short test run unless the user explicitly requests it.
   - If asked to continue after another job finishes, check `qstat`, identify the correct job id, and submit with `qsub -hold_jid <job_id> ...`.
   - Do not guess `hold_jid`; confirm the dependency job id before submitting.

6. After submission, report the qsub command used and the job id if qsub prints one.
   - Use `qstat` only as needed to confirm queue state.
   - For failures, inspect the generated qsub script and ESPnet `exp/...` logs; do not paste long logs into this skill.

## qsub Script Pattern

Keep qsub scripts close to the existing examples. The stable structure is:

```bash
#!/usr/bin/env bash
#$ -cwd
#$ -l node_f=1        # or node_q=1, following the closest working example
#$ -l h_rt=00:03:00   # test; use the requested longer time for production
#$ -N short_job_name
#$ -m abe
#$ -M daichi2ni2two@icloud.com
#$ -o /dev/null
#$ -e /dev/null
#$ -p -5

__conda_setup="$('/gs/bs/tga-shinoda/nitsu/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
elif [ -f "/gs/bs/tga-shinoda/nitsu/anaconda3/etc/profile.d/conda.sh" ]; then
    . "/gs/bs/tga-shinoda/nitsu/anaconda3/etc/profile.d/conda.sh"
else
    export PATH="/gs/bs/tga-shinoda/nitsu/anaconda3/bin:$PATH"
fi
unset __conda_setup

module load cuda/11.8.0
conda activate tf-locoformer

CUDA_VISIBLE_DEVICES=0,1,2,3
./run_target.sh --ngpu 4 --stage 6 --stop_stage 8
```

Treat this as a pattern, not a template to apply blindly. Preserve known working details from the closest existing qsub script unless the current run requires a specific change.

## Update Policy

- Add new facts only when they are stable and likely to be reused.
- Keep additions short and operational.
- If queue, node type, group, module, conda path, or run directory cannot be verified from existing qsub scripts or the environment, ask the user instead of guessing.

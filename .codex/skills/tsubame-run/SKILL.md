---
name: tsubame-run
description: Run this ESPnet workspace on TSUBAME via qsub, including Git synchronization, qsub script preparation, submission, and job inspection.
---

# TSUBAME Run

Use for this ESPnet workspace's TSUBAME jobs. For lab GPU hosts, use `.codex/skills/lab-gpu-run/SKILL.md`.

## Paths

- midgar repository: `/net/midgar/work2/nitsu/learning/tf-locoformer/espnet`
- TSUBAME: `ssh tsubame`; repository `/gs/bs/tga-shinoda/nitsu/research/tf-locoformer/espnet`
- Current recipe: `egs2/whamr/enh_rir`; qsub examples: its `qsub/*.sh`. Use the requested recipe when different.

## Sync Before Execution

A request to run on TSUBAME includes pushing the required midgar commits and pulling them on TSUBAME. Read-only queue or log checks do not trigger synchronization.

1. On midgar, inspect the diff, prepare any required code/configuration/qsub changes, validate, and commit only changes needed for the run. Push the target branch; leave unrelated changes out.
2. After connecting to TSUBAME, first check the repository status and branch. Use the same target branch and `git pull --ff-only` from its corresponding remote branch before run preparation or submission. Verify that HEAD matches the intended midgar commit.
3. If local changes, branch divergence, or an unclear branch mapping prevents safe synchronization, report the blocker without overwriting changes or force-pushing.

Edit version-controlled code, configuration, and qsub scripts on midgar. If further changes are needed after synchronization, repeat the sync above. TSUBAME is for execution and run artifacts unless the user explicitly requests otherwise.

## Prepare qsub Scripts on midgar

- Inspect existing scripts in the recipe's `qsub/` directory. Reuse the same experiment's script for continuation/reruns where suitable; inspect before overwriting. Create new scripts there only when needed.
- Follow the closest successful example for the same recipe and GPU scale. Match `run*.sh`, `--ngpu`, `--stage`, `--stop_stage`, and optional arguments to the request.
- Verify `node_f` versus `node_q` from successful examples; do not guess resource names. Existing examples use `module load cuda/11.8.0`, `conda activate tf-locoformer`, and conda under `/gs/bs/tga-shinoda/nitsu/anaconda3`.
- If no suitable example exists, read [the qsub pattern](references/qsub-script.md) as a fallback. If resources, environment, paths, or requested run conditions remain unverified, ask before submission.

## Submit and Inspect

- After synchronization, submit from the recipe directory, not from inside `qsub/`.
- Use short test jobs for startup/configuration checks before costly runs: omit `-g tga-shinoda`, set `#$ -l h_rt=00:03:00` or less, and submit with `qsub qsub/<script>.sh`.
- Production run: estimate runtime from prior logs or measured progress and set `h_rt` with a modest margin; avoid unnecessarily long limits to keep resource consumption low. Submit with `qsub -g tga-shinoda qsub/<script>.sh`.
- When asked to wait for another job, identify its actual job ID with `qstat` and use `qsub -hold_jid <job_id> ...`; never guess the dependency.
- Report the submitted command, job ID when available, and synchronized commit. Check queue state with `qstat` as needed. For failures, inspect the qsub script and ESPnet `exp/...` logs.

## Maintain This Skill

Keep only concise, verified facts reusable for this repository's qsub workflow. Do not add broad scheduler documentation, one-off logs, or experiment hyperparameters.

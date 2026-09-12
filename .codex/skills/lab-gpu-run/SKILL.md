---
name: lab-gpu-run
description: Run this ESPnet workspace on lab GPU servers. Use when Codex needs to choose a lab GPU host, check live GPU availability, run scripts over SSH, manage tmux for long jobs, or avoid GPUs used by others.
---

# Lab GPU Run

Use this skill for lab GPU servers only. For TSUBAME/qsub jobs, use `.codex/skills/tsubame-run/SKILL.md`.

## Host Selection

- Read `.codex/skills/lab-gpu-hosts/SKILL.md` when choosing or naming lab GPU SSH targets.
- Check live GPU usage on candidate hosts before running.
- Do not use GPUs currently used by other users.
- If the user does not specify a host or GPU, choose based on task size and live availability.
- For clearly light jobs, prefer weaker or single-GPU hosts before high-end multi-GPU servers.

## Sync Before Execution

A request to run on a lab GPU host includes pushing the required midgar commits and pulling them on the execution host. Read-only availability or log checks do not trigger synchronization.

1. On midgar, inspect the diff, validate and commit only changes needed for the run, then push the target branch. Leave unrelated changes out.
2. On the execution host, first check the repository status and branch. Use the same target branch and `git pull --ff-only` from its corresponding remote branch before run preparation. Verify that HEAD matches the intended midgar commit.
3. If local changes, branch divergence, or an unclear repository/branch mapping prevents safe synchronization, report the blocker without overwriting changes or force-pushing.

## Run Policy

- Edit version-controlled code, configuration, and run scripts on midgar, then repeat the sync above. Lab hosts are for execution and run artifacts unless the user explicitly requests otherwise.
- Run potentially long jobs inside `tmux` or an equivalent persistent session; clean up sessions created for the run when finished and no longer needed.

## Host Integrity

- Treat the lab GPU server itself as immutable.
- Never modify the server hardware, firmware, NVIDIA driver, kernel modules, system CUDA, system packages, module files, or host configuration.
- Never use `sudo`, `apt`, `dnf`, `yum`, driver installers, CUDA system installers, or write under `/etc`, `/usr`, `/opt`, or `/usr/local` for a job.
- Use `nvidia-smi` only for read-only inspection. Do not change MIG mode, persistence mode, power limits, clocks, or compute mode.
- Restrict dependency changes to the explicitly requested user-owned conda or virtual environment. Do not modify the conda base environment.
- Prefer environment-bundled CUDA runtime packages or wheels. Do not install a system CUDA toolkit to satisfy a project dependency.
- If a task cannot run without a host-level change, stop and report the requirement; do not perform the change.

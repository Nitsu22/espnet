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

## Run Policy

- Run potentially long jobs inside `tmux` or an equivalent persistent session.
- Clean up tmux sessions created for the run after the job finishes, if they are no longer needed.
- Treat lab GPU hosts as execution environments. Do not edit or commit tracked source/config files there unless the user explicitly asks.
- If tracked code/config changes are needed, make them on midgar, commit, push, then pull on the lab GPU host.


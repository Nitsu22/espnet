# Continue all baselines through Stage 8

Submit one CPU controller per condition (`whamr8`, `nf8`, `nf16`) from a
dedicated checkout with existing dump, statistics and experiment directories
linked into the recipe. First verify that no other job is using those outputs.

```
qsub -g tga-shinoda -l cpu_4=1 -l h_rt=00:10:00 \
  -N tfs_nf8_control qsub/baseline_pipeline.sh --condition nf8 --phase control
```

The controller reads the saved reporter and stopping criteria. It submits a
24-hour `gpu_1=1` training/inference job and a CPU controller held on that job.
On h_rt timeout (`failed=44`, `exit_status=137`), the next controller resumes
the checkpoint. It stops on other failures or a timeout without epoch progress.
Training completion requires max_epoch or the same Reporter early-stop rule as
ESPnet. An interrupted or invalid-gradient training exit cannot trigger scoring.

The GPU job reuses the verified batch-four training script and then performs
Stage 7 with `valid.loss.best.pth`. It records inference completion only after
successful inference and checking both speaker SCPs for all 5,000 validation
and 3,000 test examples. An inference timeout retries Stage 7 in a two-hour GPU
allocation without retraining. Stage 8 runs in `cpu_8=1`, six-hour allocation,
eight scoring workers and one thread per process; it requires Stage 7's marker
and saves the RESULTS.md marker only after the scoring command succeeds.

This is an event-driven qsub dependency chain, not a periodic monitor. History
and active job IDs are in each experiment's `.pipeline/jobs.json`; stage markers
are in `.pipeline/{training,inference,score}.done.json`. Unexpected accounting,
non-timeout errors, more than 12 GPU segments, or 3 scoring timeouts stop the
chain for diagnosis instead of repeatedly charging new jobs.

Per-allocation upper estimates at priority -5: GPU training 3.84 points (24 h),
GPU inference-only 0.32 (2 h), CPU control 0.002 (10 min), CPU scoring 0.144
(6 h). Subsequent allocations are created only when necessary. These are upper
estimates, not confirmed charges or promises of total runtime.

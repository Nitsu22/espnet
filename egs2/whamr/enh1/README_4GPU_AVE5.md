# WHAMR four-GPU S/M: evaluation of five-best averaged checkpoints

This branch starts at `2a1bc031d8596aa19c91c61edad358cbf7ed8217`, the
recorded evaluation commit of the existing Small run. The historical no-cache
separator is unchanged. Both existing runs use that separator implementation.
No training is performed.

The evaluator reads the existing runs in the original TSUBAME `enh1/exp`:

- Small: `enh_train_enh_tflocoformer_nocashe_small_4gpu`
- Medium: `enh_train_enh_tflocoformer_nocashe_repro_4gpu`

It verifies that `valid.loss.ave_5best.pth` is exactly the average of the five
epochs ranked by saved validation loss, then copies only that model and its
saved config into `exp/<original-tag>_ave5_eval` in this isolated checkout.
The existing dump is reused through a symlink. Existing single-best outputs
are read for comparison and never overwritten.

From this recipe directory, create `qsub_logs`, then submit:

```bash
qsub -terse -g tga-shinoda -l gpu_1=1 -l h_rt=01:00:00 -N whamr_s_ave5 qsub/eval_4gpu_ave5.sh --size small --phase infer
# Hold the CPU job on the returned GPU job ID:
qsub -terse -g tga-shinoda -l cpu_8=1 -l h_rt=01:00:00 -N whamr_s_score5 -hold_jid GPU_JOB_ID qsub/eval_4gpu_ave5.sh --size small --phase score
```

For Medium use `--size medium`, distinct job names, and a two-hour GPU limit.
Inference uses one GPU, one worker, and full validation/test utterances.
Scoring uses eight CPU workers. CPU scoring requires the successful inference
marker in addition to the scheduler dependency. Observation scores must already
exist, so this evaluation does not rescore or write into the source dump.

`RESULTS.md` contains the normal ESPnet scores. `COMPARISON.md` and
`comparison.json` contain absolute SI-SNR/SDR, input-relative improvements,
and changes from the existing single-best evaluation. Input, single-best, and
averaged-output score IDs are matched exactly, with 5,000 validation and 3,000
test mixtures. The `.evaluation` directory records checkpoint selection,
model/config/input hashes, code commit, and successful stage completion.

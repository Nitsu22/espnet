# Rec-RIR CTF-only loss experiment

Retain the baseline network including auxiliary speech decoders; change only
loss_w_cln and loss_w_rvb from 1 to 0. loss_w_rec remains 1: reference direct-path
speech convolved with estimated CTF is compared with reference reverberant speech.
Auxiliary losses remain logged but do not contribute to optimization.

NF-WHAMR speaker 1, left channel, 16 kHz, seed 0, FP32, batch size 4, optimizer,
scheduler and stopping settings match the baseline. Checkpoint selection uses
CTF reconstruction loss; compare baseline loss_rec rather than total loss.
This matches the Var-1 loss setting, not the paper's training dataset.

Run from this recipe directory:

```bash
bash run_rec_rir_single_nf_16k_ctf_only.sh
```

Audio is reused. Statistics and checkpoints are isolated under
exp/rir_stats_train_rec_rir_single_nf_16k_ctf_only and
exp/rir_train_rec_rir_single_nf_16k_ctf_only.
Resume with --stage 6 --resume true.
Evaluate using run_eval_rec_rir_single_nf_16k.sh with
--rir_exp exp/rir_train_rec_rir_single_nf_16k_ctf_only.

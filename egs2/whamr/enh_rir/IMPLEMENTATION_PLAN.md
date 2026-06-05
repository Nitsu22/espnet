# RIR-conditioned enhancement plan

## Goal

Build a new `enh_rir` recipe that keeps the existing enhancement behavior, but adds oracle RIR as an extra input.

## Fixed requirements

- Scope is `egs2/whamr/enh_rir` and new Python classes/functions only.
- Do not change existing `enh1` behavior or shared default behavior.
- Use `rir_reverberant`.
- Use left-channel RIR only: `LEFT_CH_IND = 0`.
- Add oracle RIR for train/valid only in the first implementation.
- Inference support is deferred.
- Use the full RIR length.
- `rir_ref` before batching is `[T_rir, 2]`, where the second axis is the two speakers' left-channel RIRs.
- Keep model objective, scoring, and baseline training settings the same except for the added RIR input.
- For files based on existing implementation, copy the original file to a new RIR-specific file before editing.

## Implementation Direction

1. Create an `enh_rir`-local training script instead of modifying the TEMPLATE `enh.sh` symlink target.
2. Add `rir_npz.scp` input from `egs2/whamr/se_npz/data/{tr,cv}_mix_both_reverb_min_8k/rir_npz.scp`.
3. Add a new preprocessor that loads full-length `rir_reverberant[:, LEFT_CH_IND, :]`, resamples it to the enhancement sample rate, and passes it as oracle RIR.
4. STFT the RIR with the same STFT settings as speech, so frequency bins match.
5. Convert speech STFT and RIR STFT to Conv2D features:
   - speech: `[B, 2, T1, F] -> [B, C, T1, F]`
   - RIR: `[B, 4, T2, F] -> [B, C, T2, F]`
6. Add a new TF-Locoformer separator under `espnet2/enh/separator`.
7. In the new separator, apply one CMHA fusion block only after the first speech/RIR Conv2D:
   - query: speech feature `[B, C, T1, F]`
   - key/value: RIR feature `[B, C, T2, F]`
   - attention unit: frequency-wise CMHA, `Q=[B*F, T1, C]`, `K,V=[B*F, T2, C]`
   - output: RIR context `[B, C, T1, F]`
   - concat: `[speech, RIR context] -> [B, 2C, T1, F]`
   - 1x1 Conv2D: `[B, 2C, T1, F] -> [B, C, T1, F]`
8. Feed the fused `[B, C, T1, F]` into the normal TF-Locoformer blocks.

## Files

- `espnet2/train/preprocessor_enh_rir.py`
- `espnet2/enh/espnet_model_rir.py`
- `espnet2/enh/separator/tflocoformer_separator_nocashe_rir_cmha.py`
- `espnet2/tasks/enh_rir.py`
- `espnet2/bin/enh_rir_train.py`
- `egs2/whamr/enh_rir/enh_rir.sh`
- `egs2/whamr/enh_rir/conf/tuning/train_enh_tflocoformer_small_rir_cmha.yaml`
- `egs2/whamr/enh_rir/run_rir_cmha.sh`

## Validation

- First verify key alignment between `wav.scp` and `rir_npz.scp`.
- Run config/compile checks.
- Run a small stage 5/6 job before full training.

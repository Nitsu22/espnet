# WHAMR RIR estimation task implementation plan

This document is a plan only. It intentionally does not implement code.

## Goal

Build an ESPnet2 RIR-estimation task for WHAMR that can train a TF-Locoformer-based model from reverberant speech to a time-domain RIR target, using `rir_npz.scp` and `room_param_npz.scp`.

The first comparable setting should be:

- input: monaural reverberant clean speech from one source
- target: time-domain reverberant RIR
- model input must not include oracle room parameters
- `room_param_npz.scp` is used for metadata and evaluation only

This is the setting closest to SG-RIR and FiNS, because both estimate a time-domain RIR from reverberant speech.

## Facts from local code

The local NPZ generation script is:

`egs2/whamr/se2_data/whamr_scripts/create_wham_from_scratch_npz.py`

It writes RIR and room metadata into NPZ files. Relevant keys include:

- `rir_anechoic`
- `rir_reverberant`
- `room_dim`
- `mic_pos`
- `s1_pos`
- `s2_pos`
- `T60`
- `max_rir_len`

The existing NPZ data preparation script:

`egs2/whamr/se_npz/local/whamr_data_prep.sh`

creates:

- `rir_npz.scp`
- `room_param_npz.scp`

Observed local RIR NPZ example:

- keys: `rir_anechoic`, `rir_reverberant`, `fs`, `rir_len`
- `rir_reverberant` shape: `(2, 2, 9664)`
- `fs`: `16000`

Important local compatibility issue:

- `_stack_rir()` in `create_wham_from_scratch_npz.py` saves RIR arrays as `[source, mic, time]`.
- `espnet2/train/preprocessor_npz.py::_rir_to_list()` reads arrays as `[mic, source, time]`.
- Because WHAMR currently has 2 sources and 2 microphones, the shape alone cannot reveal this semantic swap.
- The RIR task must handle this explicitly in `preprocessor_rir.py`.
- Do not change `espnet2/train/preprocessor_npz.py`, because that could alter existing NPZ/contrastive experiments.

Current speech-separation sample-rate handling:

- WHAMR audio generation is configured for `wav8k`; `create_wham_from_scratch_npz.py` has `SAMPLE_RATES = ['8k']` and calls `room.generate_audio(..., fs=SAMPLE_RATES)`.
- The enhancement/separation recipe uses `sample_rate=8k` and passes `--fs ${sample_rate}` into `enh.sh`.
- `egs2/TEMPLATE/enh1/enh.sh` formats `wav.scp` through `scripts/audio/format_wav_scp.sh --fs "${fs}"`, so training mixtures and speech references are 8 kHz in the current recipe.
- Existing NPZ contrastive configs set `preprocessor_conf.sample_rate: 8000`, and `espnet2/train/preprocessor_npz.py` synthesizes audio with `room.generate_audio(fs=sample_rate)`.
- Therefore, current speech separation trains on 8 kHz speech audio.
- Current speech separation does not train with the RIR waveform as the target. The RIR NPZ can remain at 16 kHz internally because it is used to synthesize 8 kHz speech, not as a direct supervised output.

## Literature facts

### SG-RIR

Source:

- Paper page: https://www.isca-archive.org/interspeech_2023/liao23_interspeech.html
- PDF: https://www.isca-archive.org/interspeech_2023/liao23_interspeech.pdf
- Code/sample repository: https://github.com/ffxiong/sg-rir

Facts:

- Task: blind estimation of a time-domain RIR from monaural reverberant speech.
- Input: complex spectrogram of the reverberant speech.
- Model structure: encoder produces an acoustic embedding; segmental generator produces the RIR.
- Evaluation includes direct RIR error and room-acoustic parameter accuracy.
- Reported room-parameter metrics include `RMSE_RT`, `rho_RT`, `RMSE_DRR`, and `rho_DRR`.
- The paper compares SG-RIR with Wave-U-Net and FiNS on a real measured RIR test set.

### FiNS

Sources:

- Project page: https://facebookresearch.github.io/FiNS/
- arXiv: https://arxiv.org/abs/2107.07503
- PDF: https://arxiv.org/pdf/2107.07503
- Code: https://github.com/egrinstein/FiNS

Facts:

- Task: estimate a time-domain RIR from reverberant speech.
- Input/output notation in the paper: reverberant speech `x_r(n)` and RIR `h(n)`.
- Architecture: time-domain encoder plus a decoder that models direct sound, early reflections, and late reverberation.
- Training target length in the FiNS paper: 48,000 samples, i.e. 1 second at 48 kHz.
- Training loss: multi-resolution STFT loss.
- Objective evaluation reports multi-resolution STFT reconstruction error plus `T60` and `DRR` bias, MSE, and Pearson correlation.
- Subjective evaluation uses speech produced by convolving clean speech with the predicted RIR.

## Answer on `single_clean_reverb`

The WHAMR recipe naming found locally uses:

- `mix_single_reverb`
- `mix_clean_reverb`
- `mix_both_reverb`

There is no confirmed local dataset name `single_clean_reverb`.

For a fair first comparison to SG-RIR and FiNS, `single_clean_reverb` should mean:

- one clean speech source
- reverberant
- monaural input
- no WHAM noise

Confirmed implementation decision:

- `single_clean_reverb` means using `s1_reverb.scp` directly as the input `wav.scp`.
- `mix_single_reverb` is not the primary comparison input, because the recipe comments define `single` as speech1 plus noise.

Planned configurable input modes:

- `single_clean_reverb`: `wav.scp` is derived from `s1_reverb.scp`
- `single_noisy_reverb`: `wav.scp` uses WHAMR `mix_single_reverb`
- `clean_reverb`: `wav.scp` uses WHAMR `mix_clean_reverb`
- `both_reverb`: `wav.scp` uses WHAMR `mix_both_reverb`

Primary comparison condition:

- Use `single_clean_reverb`.

Secondary robustness conditions:

- `single_noisy_reverb`
- `clean_reverb`
- `both_reverb`

Reason:

- SG-RIR and FiNS are RIR estimation from reverberant speech.
- `clean_reverb` and `both_reverb` contain two speech sources, so the target RIR is ambiguous unless the task explicitly defines which source RIR is the target or predicts multiple source RIRs.
- `single_noisy_reverb` adds noise and should not be mixed with the clean-speech comparison unless the baseline is trained/evaluated under the same noise condition.

## Target definition

Primary target:

- `rir_reverberant`

Reason:

- SG-RIR and FiNS target a full time-domain room impulse response, including direct path, early reflections, and late reverberation.
- `rir_anechoic` is not the same target; it is useful for ablation/direct-path analysis, not for the primary RIR-estimation comparison.

Target variants to support by config:

- `rir_target: reverberant`
- `rir_target: anechoic`
- `rir_target: both`

Primary output shape:

- single source, single reference microphone, fixed-length waveform: `[T_rir]`

Configurable selectors:

- `rir_source_index: 0`
- `rir_mic_index: 0`
- `predict_all_sources: false`
- `predict_all_mics: false`

Later multi-channel/multi-source extension:

- source and microphone dimensions can be retained as `[source, mic, T_rir]`.
- If multiple source RIRs are predicted, PIT is applied only over the source dimension.
- PIT is not applied over the microphone dimension, because microphone order is physically defined.

## RIR length and sampling rate

Local WHAMR RIR NPZ uses `fs=16000` in the inspected example.

The model must use a fixed target length for batching. The implementation should support:

- `rir_sample_rate`
- `rir_length`
- `rir_crop_mode`
- `rir_pad_mode`

The first implementation should not hard-code FiNS's 48 kHz / 1 second protocol, because the local WHAMR recipe trains speech models at 8 kHz and the local RIR NPZ is currently 16 kHz. For fair comparison inside this WHAMR setup, both input speech and target RIR should be handled at 8 kHz.

Confirmed first WHAMR-local setting:

- `speech_sample_rate: 8000`
- `rir_sample_rate: 8000`
- `rir_length: config-required`
- load the 16 kHz RIR from NPZ
- downsample the RIR target to 8 kHz inside `preprocessor_rir.py`
- pad or crop targets deterministically
- log the original `rir_fs`, original `rir_len`, target `rir_sample_rate`, and final `rir_length`

Implementation requirement:

- Do not rewrite existing RIR NPZ files.
- Do not change existing separation data directories.
- Use an existing dependency for resampling. The `tf-locoformer` environment has `scipy` and `torchaudio`; using `scipy.signal.resample_poly` in the preprocessor is sufficient and avoids adding dependencies.

Still required before training:

- choose the first fixed `rir_length` in 8 kHz samples.

## Losses

Initial training losses to implement:

- multi-resolution STFT loss on the predicted RIR
- time-domain L1 loss
- time-domain MSE loss
- normalized correlation loss

Recommended first loss:

- multi-resolution STFT loss plus a small time-domain L1 term

Reason:

- FiNS reports multi-resolution STFT loss as the training loss that worked best.
- Time-domain L1 stabilizes direct-path and early-reflection alignment checks.

Losses not recommended as the primary first loss:

- SI-SNR, because scale-invariant speech separation objectives can hide absolute RIR gain errors.
- PIT for the primary single-source task. The primary task has a fixed source/mic target.
- If predicting multiple source RIRs, PIT is allowed only across source outputs.

## Metrics

Metrics to implement for validation/test:

- RIR RMSE
- RIR normalized cross-correlation or Pearson correlation
- multi-resolution STFT reconstruction loss
- `T60` error: bias, MSE/RMSE, Pearson correlation
- `DRR` error: bias, MSE/RMSE, Pearson correlation

Optional metrics:

- EDT, C50, C80 if the exact comparison protocol requires them
- downstream speech reconstruction metric: convolve clean speech with predicted RIR and compare against the reference reverberant speech
- subjective-listening export script, mirroring FiNS-style evaluation

Metric implementation requirement:

- `T60` and `DRR` extraction must be identical for predicted and reference RIRs.
- If using external packages for acoustic metrics, pin the exact version and document the formula.

## ESPnet2 files to add

Core task files:

- `espnet2/tasks/rir.py`
- `espnet2/rir/__init__.py`
- `espnet2/rir/espnet_model.py`
- `espnet2/bin/rir_train.py`
- `espnet2/bin/rir_inference.py`

Model components:

- `espnet2/rir/encoder/__init__.py`
- `espnet2/rir/encoder/abs_encoder.py`
- `espnet2/rir/encoder/tf_locoformer_encoder.py`
- `espnet2/rir/decoder/__init__.py`
- `espnet2/rir/decoder/abs_decoder.py`
- `espnet2/rir/decoder/direct_rir_decoder.py`

Optional later decoder:

- `espnet2/rir/decoder/fins_decoder.py`

Reason:

- A direct decoder is simpler for the first TF-Locoformer baseline.
- A FiNS-style decoder can be added later as a separate architecture, not mixed into the first baseline.

Loss and metrics:

- `espnet2/rir/loss/__init__.py`
- `espnet2/rir/loss/criterions/__init__.py`
- `espnet2/rir/loss/criterions/time_domain.py`
- `espnet2/rir/loss/criterions/multi_resolution_stft.py`
- `espnet2/rir/loss/wrappers/__init__.py`
- `espnet2/rir/loss/wrappers/fixed_order.py`
- `espnet2/rir/metrics/__init__.py`
- `espnet2/rir/metrics/rir.py`

Preprocessor:

- `espnet2/train/preprocessor_rir.py`

Preprocessor responsibilities:

- read `speech_mix`
- read `rir_path` from `rir_npz.scp`
- read `room_param_path` from `room_param_npz.scp`
- select `rir_reverberant` or `rir_anechoic`
- verify and standardize RIR axis order
- select source and microphone
- crop or pad RIR target
- normalize RIR target only if the experiment config explicitly requests it
- pass room metadata for metrics without making it model input

## ESPnet2 task design

`RIRTask` should follow `espnet2/tasks/enh.py` structurally but should not reuse `EnhancementTask` directly.

Reasons:

- RIR estimation has different required data names.
- RIR estimation has fixed-order targets, not speech-source permutation targets.
- RIR targets are much shorter than input speech and need separate length handling.
- Evaluation metrics are RIR/acoustic-parameter metrics, not SI-SNR/STOI/SDR.

Required data names:

- train/valid:
  - `speech_mix`
  - `rir_path`
  - `room_param_path`

Optional data names:

- `speech_ref1` or `clean_speech`, only if downstream convolution metrics are enabled

Model forward contract:

- input:
  - `speech_mix`
  - `speech_mix_lengths`
  - `rir_ref`
  - `rir_ref_lengths`
  - optional room metadata for metrics
- output:
  - loss
  - stats
  - weight

No PIT in the primary task.

## Recipe files to add or change under `egs2/whamr/rir`

The existing `egs2/whamr/rir` is currently an enhancement-style recipe using `enh.sh`.

Planned files:

- `egs2/whamr/rir/rir.sh`
- `egs2/whamr/rir/run_rir.sh`
- `egs2/whamr/rir/conf/tuning/train_rir_tflocoformer.yaml`
- `egs2/whamr/rir/local/prepare_rir_data.sh`
- `egs2/whamr/rir/local/score_rir.py`
- `egs2/whamr/rir/README.md` update after implementation

`prepare_rir_data.sh` should create task-specific data dirs, for example:

- `tr_rir_single_clean_reverb_min_8k`
- `cv_rir_single_clean_reverb_min_8k`
- `tt_rir_single_clean_reverb_min_8k`

Each RIR data dir should contain:

- `wav.scp`: selected input speech
- `rir_npz.scp`: RIR NPZ path
- `room_param_npz.scp`: room metadata NPZ path
- `utt2spk`
- `spk2utt`

For the primary setting:

- `wav.scp` should be derived from `s1_reverb.scp`
- `rir_npz.scp` and `room_param_npz.scp` should come from the matching utterance IDs

For configurable mixture settings:

- `--rir_input single_clean_reverb`
- `--rir_input single_noisy_reverb`
- `--rir_input clean_reverb`
- `--rir_input both_reverb`

## Fair-comparison rules

Use these rules for SG-RIR/FiNS comparison:

- Do not use `room_param_npz.scp` as model input.
- Use the same input speech condition for all compared models.
- Use the same sample rate.
- Use the same RIR target length.
- Use the same train/valid/test split.
- Use the same target definition: full reverberant RIR.
- Report both waveform reconstruction metrics and `T60`/`DRR` metrics.
- Keep noisy/two-speaker conditions as separate experiments unless baselines are trained and evaluated under the same conditions.

## Implementation phases

### Phase 0: Verification before coding

Required checks:

- Verify all target RIR lengths in train/valid/test.
- Create or validate RIR-task data dirs whose `wav.scp` points to `s1_reverb.scp`.
- Decide the first fixed `rir_length` in 8 kHz samples.
- Validate 16 kHz -> 8 kHz RIR downsampling once before training.

Implementation has started. Do not start a fair-comparison training run until the
final `rir_length` is confirmed.

### Phase 1: Data plumbing

Add RIR-specific data preparation under `egs2/whamr/rir/local`.

Expected output:

- task-specific data directories
- `wav.scp`
- `rir_npz.scp`
- `room_param_npz.scp`
- successful key matching validation

Implemented note:

- `local/prepare_rir_data.sh` supports separate audio and RIR-NPZ sources.
- The current runnable path uses speech wav scps from `egs2/whamr/se2_data/data`
  and `rir_npz.scp` / `room_param_npz.scp` from `egs2/whamr/se_npz/data`.
- This split is necessary because the inspected `se_npz` speech scps point to
  NPZ paths and the corresponding non-random speech wav directories are empty.

### Phase 2: Minimal task

Add:

- `espnet2/train/preprocessor_rir.py`
- `espnet2/rir/espnet_model.py`
- `espnet2/tasks/rir.py`
- `espnet2/bin/rir_train.py`

Model:

- STFT or TF-Locoformer encoder for reverberant speech
- direct RIR decoder
- fixed-order loss

### Phase 3: TF-Locoformer baseline

Implement the TF-Locoformer-based RIR model while reusing patterns from:

- `espnet2/enh/`
- `espnet2/enh/separator/tflocoformer_separator.py`
- `espnet2/tasks/enh.py`

The RIR model must output an RIR waveform, not separated speech.

### Phase 4: Scoring

Add RIR scoring:

- RIR RMSE
- RIR correlation
- multi-resolution STFT loss
- `T60` metrics
- `DRR` metrics

### Phase 5: Comparison protocol

Run primary comparison:

- `single_clean_reverb`
- monaural input
- `rir_reverberant`
- one source and one reference microphone
- 8 kHz RIR target
- no oracle room parameters

Then run secondary robustness conditions only after the primary condition is stable.

## Decisions required before implementation

These are blocking decisions for code implementation:

1. Confirm the first `rir_length` in 8 kHz samples.

Resolved decisions:

- `single_clean_reverb` uses `s1_reverb.scp` directly as input `wav.scp`.
- Primary target is `rir_reverberant`.
- Primary target selector is `source_index=0, mic_index=0`.
- RIR target is downsampled to 8 kHz.
- `room_param_npz.scp` is not model input in the main comparison.
- If multiple source RIRs are predicted, PIT is applied only over the source dimension.
- RIR NPZ axis order is handled as `[source, mic, time]` in the RIR task.

Until `rir_length` is resolved, implementation can proceed structurally, but the first training run should not be started.

## Implementation status

Implemented files:

- `espnet2/train/preprocessor_rir.py`
- `espnet2/rir/espnet_model.py`
- `espnet2/rir/predictor/abs_predictor.py`
- `espnet2/rir/predictor/tflocoformer_predictor.py`
- `espnet2/rir/loss/criterions/time_domain.py`
- `espnet2/tasks/rir.py`
- `espnet2/bin/rir_train.py`
- `egs2/whamr/rir/local/prepare_rir_data.sh`
- `egs2/whamr/rir/rir.sh`
- `egs2/whamr/rir/run_rir.sh`
- `egs2/whamr/rir/conf/tuning/train_rir_tflocoformer.yaml`

Verification completed:

- `python -m espnet2.bin.rir_train --print_config` passes with the RIR config.
- `RIRPreprocessor` loads existing 16 kHz `rir_npz.scp` targets and produces 8 kHz fixed-length `rir_ref`.
- The TF-Locoformer RIR model forward pass works on a small dummy batch.
- `local/prepare_rir_data.sh` was verified on `/tmp` output and produced key-matched RIR data dirs.
- `rir_train --collect_stats true` passed on a 2-utterance `/tmp` subset and produced `speech_mix_shape` and `rir_ref_shape`.

Compatibility change:

- `espnet2/enh/separator/tflocoformer_separator.py` was minimally changed from `.view(...)` to `.reshape(...)` at the non-contiguous tensor reshape points used by the base TF-Locoformer. This is required for the RIR predictor forward pass and preserves behavior for contiguous tensors.

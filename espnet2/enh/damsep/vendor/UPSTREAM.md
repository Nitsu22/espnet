# DAMSEP network provenance

Source: https://github.com/Wenanzhi/DAMSEP
Commit: `c37eeaade1c4e90ecde9901c95bcfb37c93be749` (retrieved 2026-10-06).

Copied from `look2hear/models`, the three `models/base` helpers,
`layers/stft_tfgn.py`, and the four utility modules required by this network.
Apache-2.0 `LICENSE` and the retained Rec-RIR MIT `Rec-RIR-LICENSE` are included.
Original author notices are preserved. The package initializers are minimal
so using the network does not import Lightning, SpeechBrain or plotting tools.

Local changes to `models/SPMamba.py`:

- Initialize `backward_blocks` to `None` for the unidirectional constructor.
- Use `reshape` for speaker/spectrum views that can be noncontiguous.
- Repeat waveform lengths for the flattened batch/speaker separation decoder.
- Add an opt-in `estimate_ctf=False` path that retains separation/dereverberation
  but skips the CTF fusion, temporal blocks, pooling and decoder. The default
  full-model computation is unchanged. `models/RecRIR.py` accepts the same flag.
- Add opt-in `architecture="spmamba"`, returning the separator's own two-source
  waveform output without executing any dereverberation/CTF module.
- Add opt-in `architecture="feature_dereverb"`, connecting the common separator
  features directly to dereverberation through a pointwise Linear expansion
  (16 to 64 channels with the released separator configuration). This path skips
  the separator's output ConvTranspose2d, intermediate waveform iSTFT/STFT,
  and the dereverberation Conv2d encoder. Dereverberation uses the separator's
  129-bin, 256-point / 64-hop grid and a joint four-channel RI output decoder
  for two sources, followed by one final iSTFT and mixture-scale restoration.
  `BiSpatialNet` accepts `bypass_encoder=True` for already expanded features.

`layers/stft_tfgn.py` uses the current `@typechecked` constructor decorator
instead of the removed `typeguard.check_argument_types` API.
Python sources also follow this repository's Black/isort formatting.

The default DAMSEP path retains the released separator, shared
dereverberation/CTF branch, trainable scalar fusion, Mamba parameters, temporal
softmax pooling, architecture and initializers. Both opt-in ablations retain
the same separator backbone and its initial weights. Their unused parameters
are frozen by the ESPnet adapter, and both require clean-only supervision.

The new ESPnet adapter lives outside this directory. It intentionally replaces
fixed distance order with clean-waveform PIT for NF-WHAMR and corrects CTF batch
alignment. It retains the released auxiliary-loss analysis (FFT 512, window
256, hop 128, square-root Hann), distinct from the network's own analyses.

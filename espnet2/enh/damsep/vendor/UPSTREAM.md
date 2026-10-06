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

`layers/stft_tfgn.py` uses the current `@typechecked` constructor decorator
instead of the removed `typeguard.check_argument_types` API.
Python sources also follow this repository's Black/isort formatting.

The separator, shared dereverberation/CTF branch, trainable scalar fusion,
Mamba parameters, temporal softmax pooling, and decoders retain the release's
architecture and initializers. No alternative separator is substituted.

The new ESPnet adapter lives outside this directory. It intentionally replaces
fixed distance order with clean-waveform PIT for NF-WHAMR and corrects CTF batch
alignment. It retains the released auxiliary-loss analysis (FFT 512, window
256, hop 128, square-root Hann), distinct from the network's own analyses.

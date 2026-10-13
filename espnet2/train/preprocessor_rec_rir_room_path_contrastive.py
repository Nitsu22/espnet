"""Two speech views with one shared pair of physical direct/reverb RIRs."""

from espnet2.train.preprocessor_rec_rir_sweep_v2_pit import (
    RecRIRSweepV2PITPreprocessor,
)


class RecRIRRoomPathContrastivePreprocessor(RecRIRSweepV2PITPreprocessor):
    """Crop each view independently and retain the original RIR normalization.

    The views are already rendered from the same ordered RIR pair. Cropping
    does not change the target CTF: Sweep v2 supervision uses the complete
    physical RIRs rather than speech reconstruction at crop boundaries.
    Validation keeps full utterances, matching the existing recipe.
    """

    def __call__(self, uid, data):
        if self.train and "speech_mix_view2" not in data:
            raise ValueError(f"{uid}: paired training requires speech_mix_view2")
        result = super().__call__(uid, data)
        if "speech_mix_view2" in data:
            second = dict(data)
            second["speech_mix"] = data["speech_mix_view2"]
            result["speech_mix_view2"] = super().__call__(uid, second)["speech_mix"]
        return result

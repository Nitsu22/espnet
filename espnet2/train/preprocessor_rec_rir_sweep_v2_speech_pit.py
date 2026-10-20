"""One synchronized speech crop plus the unchanged physical Sweep v2 pairs."""
from espnet2.train.preprocessor_rec_rir_pit import RecRIRPITPreprocessor
from espnet2.train.preprocessor_rec_rir_sweep_v2_pit import RecRIRSweepV2PITPreprocessor


class RecRIRSweepV2SpeechPITPreprocessor(RecRIRPITPreprocessor):
    def __init__(self, train, rir_length=16000, **kwargs):
        super().__init__(train=train, **kwargs)
        if self.use_sweep_target:
            raise ValueError('Joint supervision requires the original speech pairs')
        self.physical_pairs = RecRIRSweepV2PITPreprocessor(
            train=False, speech_segment=None, rir_length=rir_length,
            force_single_channel=self.force_single_channel)

    def __call__(self, uid, data):
        # The existing speech preprocessor selects ONE crop for mixture,
        # direct1/2 and reverb1/2. RIRs are never cropped with the utterance.
        prepared = super().__call__(uid, dict(data))
        pairs = self.physical_pairs(uid, prepared)
        prepared.update(pairs)
        return prepared

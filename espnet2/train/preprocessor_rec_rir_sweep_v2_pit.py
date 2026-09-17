"""Keep two physical direct/reverb RIR pairs for sweep v2 PIT."""
from espnet2.train.preprocessor_rec_rir_sweep_v2 import RecRIRSweepV2Preprocessor


class RecRIRSweepV2PITPreprocessor(RecRIRSweepV2Preprocessor):
    def __call__(self, uid, data):
        # Crop mixture once; the RIR pairs have independent common gains but
        # their physical direct/reverb timing and relative amplitude are kept.
        first = super().__call__(uid, dict(speech_mix=data['speech_mix'],
                                         rir_direct=data['rir_direct1'],
                                         rir_ref=data['rir_ref1']))
        second = super().__call__(uid, dict(speech_mix=first['speech_mix'],
                                          rir_direct=data['rir_direct2'],
                                          rir_ref=data['rir_ref2']))
        return dict(speech_mix=first['speech_mix'], rir_ref1=first['rir_ref'],
                    rir_direct1=first['rir_direct'], rir_ref2=second['rir_ref'],
                    rir_direct2=second['rir_direct'])

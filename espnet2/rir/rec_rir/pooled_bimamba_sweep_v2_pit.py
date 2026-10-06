"""New predictor with the unchanged TF-Locoformer Sweep v2 PIT objective/PIM."""
from espnet2.rir.rec_rir.pooled_bimamba import PooledBiMambaCTFPredictor
from espnet2.rir.rec_rir.tflocoformer_sweep_v2_pit import ESPnetTFLocoformerSweepV2PITModel


class ESPnetPooledBiMambaSweepV2PITModel(ESPnetTFLocoformerSweepV2PITModel):
    def __init__(self, num_freqs=257, num_spk=2, ctf_taps=60, predictor_conf=None, **kwargs):
        conf = dict(predictor_conf or {})
        reserved = {'input_dim', 'num_spk', 'ctf_taps'} & conf.keys()
        if reserved:
            raise ValueError(f"Specify {sorted(reserved)} in model_conf, not predictor_conf")
        predictor = PooledBiMambaCTFPredictor(input_dim=num_freqs, num_spk=num_spk,
                                            ctf_taps=ctf_taps, **conf)
        super().__init__(num_freqs=num_freqs, num_spk=num_spk, ctf_taps=ctf_taps,
                         ctf_predictor=predictor, **kwargs)

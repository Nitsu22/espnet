#!/usr/bin/env python3
from typing import List, Union

import numpy as np
import torch
from typeguard import typechecked

import espnet2.bin.enh_inference as _base
from espnet2.tasks.enh_rir import EnhancementRIRCTFTask
from espnet2.torch_utils.device_funcs import to_device


_base.EnhancementTask = EnhancementRIRCTFTask


class SeparateSpeech(_base.SeparateSpeech):
    @torch.no_grad()
    @typechecked
    def __call__(
        self,
        speech_mix: Union[torch.Tensor, np.ndarray],
        rir_ctf: Union[torch.Tensor, np.ndarray],
        fs: int = 8000,
        **kwargs,
    ) -> List[Union[torch.Tensor, np.ndarray]]:
        if isinstance(speech_mix, np.ndarray):
            speech_mix = torch.as_tensor(speech_mix)
        if isinstance(rir_ctf, np.ndarray):
            rir_ctf = torch.as_tensor(rir_ctf)

        assert speech_mix.dim() > 1, speech_mix.size()
        batch_size = speech_mix.size(0)
        speech_mix = speech_mix.to(getattr(torch, self.dtype))
        rir_ctf = rir_ctf.to(getattr(torch, self.dtype))

        if speech_mix.dim() == 3:
            speech_mix = speech_mix[:, :, self.ref_channel]

        lengths = speech_mix.new_full(
            [batch_size], dtype=torch.long, fill_value=speech_mix.size(1)
        )
        rir_ctf_lengths = kwargs.get("rir_ctf_lengths", None)
        if rir_ctf_lengths is None:
            rir_ctf_lengths = rir_ctf.new_full(
                [batch_size], dtype=torch.long, fill_value=rir_ctf.size(1)
            )
        elif isinstance(rir_ctf_lengths, np.ndarray):
            rir_ctf_lengths = torch.as_tensor(rir_ctf_lengths)
        rir_ctf_lengths = rir_ctf_lengths.to(dtype=torch.long)

        speech_mix = to_device(speech_mix, device=self.device)
        lengths = to_device(lengths, device=self.device)
        rir_ctf = to_device(rir_ctf, device=self.device)
        rir_ctf_lengths = to_device(rir_ctf_lengths, device=self.device)

        if getattr(self.enh_model, "normalize_variance_per_ch", False):
            mix_std_ = torch.std(speech_mix, dim=1, keepdim=True)
            speech_mix = speech_mix / mix_std_
        elif getattr(self.enh_model, "normalize_variance", False):
            mix_std_ = torch.std(speech_mix, dim=1, keepdim=True)
            speech_mix = speech_mix / mix_std_

        if self.segmenting and lengths[0] > self.segment_size * fs:
            overlap_length = int(np.round(fs * (self.segment_size - self.hop_size)))
            num_segments = int(
                np.ceil((speech_mix.size(1) - overlap_length) / (self.hop_size * fs))
            )
            t = T = int(self.segment_size * fs)
            pad_shape = speech_mix[:, :T].shape
            enh_waves = []
            range_ = _base.trange if self.show_progressbar else range
            for i in range_(num_segments):
                st = int(i * self.hop_size * fs)
                en = st + T
                if en >= lengths[0]:
                    en = lengths[0]
                    speech_seg = speech_mix.new_zeros(pad_shape)
                    t = en - st
                    speech_seg[:, :t] = speech_mix[:, st:en]
                else:
                    t = T
                    speech_seg = speech_mix[:, st:en]

                lengths_seg = speech_mix.new_full(
                    [batch_size], dtype=torch.long, fill_value=T
                )
                processed_wav = self._forward_enhance(
                    speech_seg, lengths_seg, rir_ctf, rir_ctf_lengths
                )

                if self.normalize_segment_scale:
                    mix_energy = torch.sqrt(
                        torch.mean(speech_seg[:, :t].pow(2), dim=1, keepdim=True)
                    )
                    enh_energy = torch.sqrt(
                        torch.mean(
                            sum(processed_wav)[:, :t].pow(2), dim=1, keepdim=True)
                        )
                    processed_wav = [
                        w * (mix_energy / enh_energy) for w in processed_wav
                    ]
                enh_waves.append(torch.stack(processed_wav, dim=0))

            waves = enh_waves[0]
            for i in range(1, num_segments):
                perm = self.cal_permumation(
                    waves[:, :, -overlap_length:],
                    enh_waves[i][:, :, :overlap_length],
                    criterion="si_snr",
                )
                for batch in range(batch_size):
                    enh_waves[i][:, batch] = enh_waves[i][perm[batch], batch]

                if i == num_segments - 1:
                    enh_waves[i][:, :, t:] = 0
                    enh_waves_res_i = enh_waves[i][:, :, overlap_length:t]
                else:
                    enh_waves_res_i = enh_waves[i][:, :, overlap_length:]

                waves[:, :, -overlap_length:] = (
                    waves[:, :, -overlap_length:] + enh_waves[i][:, :, :overlap_length]
                ) / 2
                waves = torch.cat([waves, enh_waves_res_i], dim=2)
            assert waves.size(2) == speech_mix.size(1), (waves.shape, speech_mix.shape)
            waves = torch.unbind(waves, dim=0)
        else:
            waves = self._forward_enhance(
                speech_mix, lengths, rir_ctf, rir_ctf_lengths
            )

        if getattr(self.enh_model, "normalize_variance_per_ch", False):
            waves = [w * mix_std_ for w in waves]
        elif getattr(self.enh_model, "normalize_variance", False):
            waves = [w * mix_std_ for w in waves]

        assert len(waves) == self.num_spk, len(waves) == self.num_spk
        assert len(waves[0]) == batch_size, (len(waves[0]), batch_size)
        if self.normalize_output_wav:
            waves = [
                (w / abs(w).max(dim=1, keepdim=True)[0] * 0.9).cpu().numpy()
                for w in waves
            ]
        else:
            waves = [w.cpu().numpy() for w in waves]

        return waves

    def _forward_enhance(
        self,
        speech_mix: torch.Tensor,
        lengths: torch.Tensor,
        rir_ctf: torch.Tensor,
        rir_ctf_lengths: torch.Tensor,
    ) -> List[torch.Tensor]:
        additional = {
            "rir_ctf": rir_ctf,
            "rir_ctf_lengths": rir_ctf_lengths,
        }
        speech_pre, _, _, _ = self.enh_model.forward_enhance(
            speech_mix, lengths, additional, fs=None
        )
        return speech_pre


_base.SeparateSpeech = SeparateSpeech

get_parser = _base.get_parser
inference = _base.inference
main = _base.main


if __name__ == "__main__":
    main()

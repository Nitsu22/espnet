"""Paired fixed-RIR room/path contrastive learning with Sweep v2 PIT.

Each batch item is one distinct room/microphone/RIR-pair condition with two
independently voiced mixtures. Room positives share that RIR pair; this does
not enforce invariance to source positions. Path positives share an ordered
physical RIR, after separate CTF PIT assignment for each view. Path negatives
are only the OTHER path in the SAME item, not paths from unrelated rooms.

Padding, CTF supervision and ordinary sweep/PIM inference follow the baseline.
Projection heads are auxiliary and are never used by the CTF decoder.
"""

import math
from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F

from espnet2.rir.rec_rir.pooled_bimamba_sweep_v2_pit import (
    ESPnetPooledBiMambaSweepV2PITModel,
)
from espnet2.torch_utils.device_funcs import force_gatherable


def symmetric_infonce(first, second, temperature):
    """NT-Xent over matching views ``[groups, identities, features]``.

    Identities interact only within a group. For room loss there is one group
    and each batch item is an identity. For path loss each physical room item
    is a separate group and the two paths are its identities. The same path
    in the other view is always the sole positive, never a negative.
    """
    if first.shape != second.shape or first.ndim != 3:
        raise ValueError("Expected matching [groups, identities, features] codes")
    if any(size <= 0 for size in first.shape):
        raise ValueError("Contrastive codes must have nonempty axes")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("contrastive_temperature must be positive and finite")
    identities = first.shape[1]
    embeddings = F.normalize(torch.cat((first, second), dim=1).float(), dim=-1)
    logits = embeddings @ embeddings.transpose(-2, -1) / temperature
    count = 2 * identities
    self_mask = torch.eye(count, dtype=torch.bool, device=logits.device)
    logits = logits.masked_fill(self_mask[None], -torch.inf)
    # The labels point to the same identity in the opposite view.
    labels = (torch.arange(count, device=logits.device) + identities) % count
    labels = labels[None].expand(first.shape[0], -1)
    return F.cross_entropy(logits.reshape(-1, count), labels.reshape(-1))


class FrequencyCodeProjection(nn.Module):
    """Project feature channels and retain the frequency ordering in cosine similarity."""

    def __init__(self, dim, projection_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim), nn.SiLU(),
            nn.Linear(dim, projection_dim),
        )

    def forward(self, code):
        return self.layers(code).flatten(-2)


class ESPnetPooledBiMambaRoomPathContrastiveModel(ESPnetPooledBiMambaSweepV2PITModel):
    def __init__(self, predictor_conf=None, lambda_room=0.01, lambda_path=0.01,
                 contrastive_temperature=0.1, projection_dim=32,
                 validation_contrastive=False, **kwargs):
        if (not all(math.isfinite(value) for value in
                    (lambda_room, lambda_path, contrastive_temperature))
                or min(lambda_room, lambda_path) < 0 or contrastive_temperature <= 0):
            raise ValueError("Use finite nonnegative contrastive weights and positive temperature")
        if (isinstance(projection_dim, bool) or not isinstance(projection_dim, int)
                or projection_dim <= 0):
            raise ValueError("projection_dim must be a positive integer")
        if not isinstance(validation_contrastive, bool):
            raise ValueError("validation_contrastive must be a boolean")
        conf = dict(predictor_conf or {})
        if conf.get("room_path_factorized", True) is not True:
            raise ValueError("Contrastive room/path training requires room_path_factorized=true")
        conf["room_path_factorized"] = True
        super().__init__(predictor_conf=conf, **kwargs)
        if self.drr_loss_weight:
            raise ValueError("Use the basic Sweep v2 loss for this contrastive comparison")
        self.lambda_room = float(lambda_room)
        self.lambda_path = float(lambda_path)
        self.contrastive_temperature = float(contrastive_temperature)
        self.validation_contrastive = validation_contrastive
        dim = self.ctf_predictor.emb_dim
        # Initialize these after all baseline modules to preserve seeded
        # predictor weights, including the zero-weight paired-data control.
        self.room_projection = FrequencyCodeProjection(dim, projection_dim)
        self.path_projection = FrequencyCodeProjection(dim, projection_dim)

    def _estimate_ctf_and_aux_from_complex(self, input_complex):
        input_tf = input_complex.squeeze(1).transpose(1, 2).contiguous()
        ctf, _, auxiliary = self.ctf_predictor(
            input_tf, additional={"return_pre_slot_codes": True})
        return ctf.to(dtype=torch.complex64), auxiliary

    def align_path_codes(self, codes, best):
        """Reorder predicted slots into teacher RIR order using CTF-only PIT."""
        permutations = torch.as_tensor(self.permutations, device=codes.device)
        # Each permutation maps prediction -> reference; invert for gathering.
        reference_to_prediction = permutations[best].argsort(dim=-1)
        index = reference_to_prediction[..., None, None].expand_as(codes)
        return codes.gather(1, index)

    def contrastive_losses(self, first, second, best_first, best_second):
        room_first = self.room_projection(first["room_code"])
        room_second = self.room_projection(second["room_code"])
        path_first = self.path_projection(self.align_path_codes(
            first["pre_slot_path_codes"], best_first))
        path_second = self.path_projection(self.align_path_codes(
            second["pre_slot_path_codes"], best_second))
        room_loss = symmetric_infonce(room_first[None], room_second[None],
                                     self.contrastive_temperature)
        path_loss = symmetric_infonce(path_first, path_second,
                                     self.contrastive_temperature)
        return room_loss, path_loss

    def _prepare_view(self, speech, lengths):
        speech = self._to_mono(speech)
        if (lengths is None or lengths.shape != speech.shape[:1]
                or torch.any(lengths <= 0) or torch.any(lengths > speech.shape[1])):
            raise ValueError("Each speech view requires valid positive waveform lengths")
        speech = speech[:, :int(lengths.max())]
        if self.normalize_by_mix:
            speech = speech / speech.abs().amax(1, keepdim=True).clamp_min(1e-8)
        return speech

    def forward(self, speech_mix, speech_mix_lengths, rir_ref1, rir_ref2,
                rir_direct1, rir_direct2, speech_mix_view2=None,
                speech_mix_view2_lengths=None, **kwargs):
        paired = self.training or self.validation_contrastive
        if paired and speech_mix_view2 is None:
            raise ValueError("Paired training/validation requires speech_mix_view2")
        first = self._prepare_view(speech_mix, speech_mix_lengths)
        batch = first.shape[0]
        direct = torch.stack([self._to_mono(x) for x in (rir_direct1, rir_direct2)], 1)
        reverb = torch.stack([self._to_mono(x) for x in (rir_ref1, rir_ref2)], 1)
        if direct.shape != reverb.shape or direct.shape[0] != batch:
            raise ValueError("Expected matching fixed-length direct/reverb RIR pairs")
        for name in ("rir_ref1", "rir_ref2", "rir_direct1", "rir_direct2"):
            lengths = kwargs.get(name + "_lengths")
            if lengths is not None and not torch.all(lengths == direct.shape[-1]):
                raise ValueError("Paired contrastive training requires fixed-length RIRs")
        speech = first
        if paired:
            second = self._prepare_view(speech_mix_view2, speech_mix_view2_lengths)
            if second.shape[0] != batch:
                raise ValueError("Both speech views must have the same batch size")
            length = max(first.shape[1], second.shape[1])
            # Same padding semantics as the baseline. Paired examples from the
            # recipe have equal lengths; full-time-axis pooling is unchanged.
            speech = torch.cat((F.pad(first, (0, length - first.shape[1])),
                                F.pad(second, (0, length - second.shape[1]))), 0)
        spectrum = self.transforms.stft(speech[:, None], "complex")
        ctf, auxiliary = self._estimate_ctf_and_aux_from_complex(spectrum)
        primary, best_first, _ = self.paired_pit_loss_with_stats(ctf[:batch], direct, reverb)
        loss = primary
        stats = dict(loss_sweep=primary.detach(),
                     pit_perm0_ratio=(best_first == 0).float().mean())
        if paired:
            secondary, best_second, _ = self.paired_pit_loss_with_stats(
                ctf[batch:], direct, reverb)
            first_codes = OrderedDict((key, value[:batch]) for key, value in auxiliary.items())
            second_codes = OrderedDict((key, value[batch:]) for key, value in auxiliary.items())
            room_loss, path_loss = self.contrastive_losses(
                first_codes, second_codes, best_first, best_second)
            sweep_loss = (primary + secondary) / 2
            paired_total = (sweep_loss + self.lambda_room * room_loss
                            + self.lambda_path * path_loss)
            stats.update(loss_sweep_pair=sweep_loss.detach(),
                         loss_sweep_view1=primary.detach(),
                         loss_sweep_view2=secondary.detach(),
                         loss_room=room_loss.detach(), loss_path=path_loss.detach(),
                         loss_paired_total=paired_total.detach(),
                         room_negative_count=2 * max(0, batch - 1))
            if self.training:
                loss = paired_total
                stats["loss_sweep"] = sweep_loss.detach()
                stats["pit_perm0_ratio"] = torch.cat((best_first, best_second)).eq(0).float().mean()
            # Even optional paired validation retains primary-view loss and
            # loss_sweep for checkpoint selection against existing methods.
        stats["loss"] = loss.detach()
        return force_gatherable((loss, stats, batch), loss.device)

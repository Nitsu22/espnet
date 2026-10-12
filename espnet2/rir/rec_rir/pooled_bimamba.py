"""Time encoders, two-slot pooling and shared frequency refinement for CTFs.

Lengths are intentionally not masked, matching the current TF-Locoformer
Sweep v2 baseline. The returned taps use that baseline's convolution ordering.
"""
from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F
from rotary_embedding_torch import RotaryEmbedding

from espnet2.rir.rec_rir.model import _build_mamba


class TimeLocalAttention(nn.Module):
    """Residual temporal attention within an exact centered frame window.

    Every frequency bin is an independent sequence. Query and sequence
    batches are bounded separately, avoiding a full ``T x T`` attention
    matrix or mask even for long validation utterances. RoPE positions are
    absolute frame indices, independent of query chunk boundaries.

    Only nonexistent positions beyond the sequence edges are excluded.
    Batch padding remains visible, matching the existing predictor baseline.
    """

    def __init__(self, dim, heads, window, eps, query_chunk_size=64,
                 sequence_batch_size=64):
        super().__init__()
        dimensions = (dim, heads, window, query_chunk_size, sequence_batch_size)
        if any(isinstance(value, bool) or not isinstance(value, int)
               or value <= 0 for value in dimensions):
            raise ValueError("Local attention dimensions must be positive integers")
        if window % 2 != 1:
            raise ValueError("Local attention window must be odd")
        if dim % heads or (dim // heads) % 2:
            raise ValueError("Temporal RoPE requires an even head dimension and dim divisible by heads")
        self.dim = dim
        self.heads = heads
        self.window = window
        self.radius = window // 2
        self.query_chunk_size = query_chunk_size
        self.sequence_batch_size = sequence_batch_size
        self.norm_attention = nn.LayerNorm(dim, eps=eps)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.projection = nn.Linear(dim, dim)
        head_dim = dim // heads
        frequencies = 1.0 / (
            10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("rope_frequencies", frequencies, persistent=False)

    def _rotate_time_positions(self, tensor, positions):
        angles = (positions.float()[:, None]
                  * self.rope_frequencies.float()[None, :])
        angles = angles.repeat_interleave(2, dim=-1).to(tensor)
        paired = tensor.reshape(*tensor.shape[:-1], -1, 2)
        rotated_half = torch.stack((-paired[..., 1], paired[..., 0]), -1)
        rotated_half = rotated_half.flatten(-2)
        return tensor * angles.cos() + rotated_half * angles.sin()

    def forward(self, x):
        if x.ndim != 4 or x.shape[-1] != self.dim:
            raise ValueError("Expected features [B, T, F, dim]")
        batch, frames, freqs, dim = x.shape
        if min(batch, frames, freqs) <= 0:
            raise ValueError("Local attention requires nonempty feature axes")
        sequence = x.permute(0, 2, 1, 3).reshape(batch * freqs, frames, dim)
        positions = torch.arange(frames, device=x.device)
        updates = []
        for sequence_start in range(0, batch * freqs, self.sequence_batch_size):
            chunk = sequence[sequence_start:sequence_start + self.sequence_batch_size]
            count = chunk.shape[0]
            qkv = self.qkv(self.norm_attention(chunk)).reshape(
                count, frames, 3, self.heads, dim // self.heads)
            query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
            query = self._rotate_time_positions(query, positions)
            key = self._rotate_time_positions(key, positions)
            query_updates = []
            for query_start in range(0, frames, self.query_chunk_size):
                query_end = min(frames, query_start + self.query_chunk_size)
                key_start = max(0, query_start - self.radius)
                key_end = min(frames, query_end + self.radius)
                # SDPA's boolean mask uses True for allowed key positions.
                distance = (positions[query_start:query_end, None]
                            - positions[None, key_start:key_end]).abs()
                allowed = distance <= self.radius
                update = F.scaled_dot_product_attention(
                    query[:, :, query_start:query_end],
                    key[:, :, key_start:key_end],
                    value[:, :, key_start:key_end],
                    attn_mask=allowed, dropout_p=0.0)
                query_updates.append(update)
            update = torch.cat(query_updates, dim=2)
            update = update.transpose(1, 2).reshape(count, frames, dim)
            updates.append(self.projection(update))
        update = torch.cat(updates, dim=0)
        return x + update.reshape(batch, freqs, frames, dim).permute(0, 2, 1, 3)


class TimeBiMamba(nn.Module):
    def __init__(self, dim, d_state, d_conv, eps):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)
        specification = f"mamba({d_state},{d_conv})"
        self.forward_mamba = _build_mamba(dim, specification)
        self.backward_mamba = _build_mamba(dim, specification)
        self.projection = nn.Linear(2 * dim, dim)

    def forward(self, x):
        batch, frames, freqs, dim = x.shape
        sequence = self.norm(x).permute(0, 2, 1, 3).reshape(batch * freqs, frames, dim)
        forward = self.forward_mamba(sequence.contiguous())
        backward = self.backward_mamba(sequence.flip(1).contiguous()).flip(1)
        update = self.projection(torch.cat((forward, backward), -1))
        return x + update.reshape(batch, freqs, frames, dim).permute(0, 2, 1, 3)


class TimeBiLSTM(nn.Module):
    """Use the same time layout and residual as BiMamba with a BiLSTM."""

    def __init__(self, dim, eps):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)
        self.lstm = nn.LSTM(dim, dim, batch_first=True, bidirectional=True)
        self.projection = nn.Linear(2 * dim, dim)

    def forward(self, x):
        batch, frames, freqs, dim = x.shape
        sequence = self.norm(x).permute(0, 2, 1, 3).reshape(batch * freqs, frames, dim)
        update, _ = self.lstm(sequence.contiguous())
        update = self.projection(update)
        return x + update.reshape(batch, freqs, frames, dim).permute(0, 2, 1, 3)


class LightFrequencyBlock(nn.Module):
    def __init__(self, dim, freqs, bottleneck, rank, kernel, eps, global_mlp=True):
        super().__init__()
        if not isinstance(global_mlp, bool):
            raise ValueError("global_mlp must be a boolean")
        self.global_mlp = global_mlp
        self.local_norm = nn.LayerNorm(dim, eps=eps)
        self.local = nn.Conv1d(dim, dim, kernel, padding=kernel // 2, groups=dim)
        if global_mlp:
            self.global_norm = nn.LayerNorm(dim, eps=eps)
            self.compress = nn.Linear(dim, bottleneck)
            # One frequency MLP shared by all compressed feature channels.
            self.frequency_mlp = nn.Sequential(nn.Linear(freqs, rank), nn.GELU(), nn.Linear(rank, freqs))
            self.expand = nn.Linear(bottleneck, dim)

    def forward(self, x):
        batch, frames, freqs, dim = x.shape
        sequence = x.reshape(batch * frames, freqs, dim)
        update = self.local(self.local_norm(sequence).transpose(1, 2)).transpose(1, 2)
        sequence = sequence + F.gelu(update)
        if self.global_mlp:
            update = self.compress(self.global_norm(sequence)).transpose(1, 2)
            update = self.frequency_mlp(update).transpose(1, 2)
            sequence = sequence + self.expand(update)
        return sequence.reshape(batch, frames, freqs, dim)


class FrequencyAttentionBlock(nn.Module):
    def __init__(self, dim, heads, ffn_dim, kernel, eps, attention=True):
        super().__init__()
        if not isinstance(attention, bool):
            raise ValueError("attention must be a boolean")
        self.attention = attention
        self.heads = heads
        if attention:
            if dim % heads or (dim // heads) % 2:
                raise ValueError("RoPE requires an even head dimension and dim divisible by heads")
            self.norm_attention = nn.LayerNorm(dim, eps=eps)
            self.qkv = nn.Linear(dim, 3 * dim)
            self.rope = RotaryEmbedding(dim // heads)
            self.projection = nn.Linear(dim, dim)
        self.norm_ffn = nn.LayerNorm(dim, eps=eps)
        self.ffn_in = nn.Linear(dim, 2 * ffn_dim)
        self.local = nn.Conv1d(ffn_dim, ffn_dim, kernel, padding=kernel // 2, groups=ffn_dim)
        self.ffn_out = nn.Linear(ffn_dim, dim)

    def forward(self, x):
        if self.attention:
            batch, freqs, dim = x.shape
            qkv = self.qkv(self.norm_attention(x)).reshape(batch, freqs, 3, self.heads, dim // self.heads)
            query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
            query = self.rope.rotate_queries_or_keys(query)
            key = self.rope.rotate_queries_or_keys(key)
            update = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0)
            x = x + self.projection(update.transpose(1, 2).reshape(batch, freqs, dim))
        update = F.glu(self.ffn_in(self.norm_ffn(x)), dim=-1)
        update = F.silu(self.local(update.transpose(1, 2)).transpose(1, 2))
        return x + self.ffn_out(update)


class FrequencyBiMambaBlock(nn.Module):
    """Bidirectional frequency Mamba with the attention block's Conv-FFN.

    Each input is an independent frequency sequence ``[N, F, D]``. The
    backward branch reverses frequency bins, never frames or speaker slots.
    The FFN modules are reused from a seeded attention block so comparisons
    change the sequence mixer while keeping the FFN initialization fixed.
    """

    def __init__(self, dim, d_state, d_conv, eps, ffn_source):
        super().__init__()
        if ffn_source.ffn_in.in_features != dim:
            raise ValueError("Frequency Mamba and FFN dimensions differ")
        self.dim = dim
        self.norm_mamba = nn.LayerNorm(dim, eps=eps)
        specification = f"mamba({d_state},{d_conv})"
        self.forward_mamba = _build_mamba(dim, specification)
        self.backward_mamba = _build_mamba(dim, specification)
        self.projection = nn.Linear(2 * dim, dim)
        self.norm_ffn = ffn_source.norm_ffn
        self.ffn_in = ffn_source.ffn_in
        self.local = ffn_source.local
        self.ffn_out = ffn_source.ffn_out

    def forward(self, x):
        if x.ndim != 3 or x.shape[-1] != self.dim:
            raise ValueError("Expected frequency sequences [N, F, dim]")
        sequence = self.norm_mamba(x)
        forward = self.forward_mamba(sequence.contiguous())
        backward = self.backward_mamba(sequence.flip(1).contiguous()).flip(1)
        x = x + self.projection(torch.cat((forward, backward), -1))
        update = F.glu(self.ffn_in(self.norm_ffn(x)), dim=-1)
        update = F.silu(self.local(update.transpose(1, 2)).transpose(1, 2))
        return x + self.ffn_out(update)


class FramewiseFrequencyBlock(nn.Module):
    """Apply a frequency block independently to every observation frame."""

    def __init__(self, sequence_block):
        super().__init__()
        self.sequence_block = sequence_block

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError("Expected features [B, T, F, D]")
        batch, frames, freqs, dim = x.shape
        output = self.sequence_block(x.reshape(batch * frames, freqs, dim))
        return output.reshape(batch, frames, freqs, dim)


class PostPoolingAttention(nn.Module):
    """Attention-only residual over speakers, frequencies, or their union.

    Input and output are ``[batch, speakers, frequencies, channels]``. All
    projections are shared across speakers and frequencies. Joint attention
    uses the same frequency positions for every speaker, so flattening the
    speaker and frequency axes does not introduce a fictitious frequency
    offset between speakers.
    """

    def __init__(self, dim, heads, eps, axis):
        super().__init__()
        if axis not in ("speaker", "joint", "frequency"):
            raise ValueError("axis must be 'speaker', 'joint', or 'frequency'")
        if (isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0
                or isinstance(heads, bool) or not isinstance(heads, int)
                or heads <= 0 or dim % heads):
            raise ValueError("Use positive integer dimensions with dim divisible by heads")
        if axis != "speaker" and (dim // heads) % 2:
            raise ValueError("Frequency RoPE requires an even head dimension")
        self.axis = axis
        self.dim = dim
        self.heads = heads
        self.norm_attention = nn.LayerNorm(dim, eps=eps)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.projection = nn.Linear(dim, dim)
        # Unlike RotaryEmbedding's frozen Parameter, these deterministic
        # frequencies are a buffer: every interaction has identical counts.
        inverse_frequencies = None
        if axis != "speaker":
            head_dim = dim // heads
            inverse_frequencies = 1.0 / (
                10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("rope_frequencies", inverse_frequencies,
                             persistent=False)

    def _rotate_frequency_positions(self, tensor, positions):
        angles = (positions.float()[:, None]
                  * self.rope_frequencies.float()[None, :])
        angles = angles.repeat_interleave(2, dim=-1).to(tensor)
        paired = tensor.reshape(*tensor.shape[:-1], -1, 2)
        rotated_half = torch.stack((-paired[..., 1], paired[..., 0]), -1)
        rotated_half = rotated_half.flatten(-2)
        return tensor * angles.cos() + rotated_half * angles.sin()

    def forward(self, x):
        if x.ndim != 4 or x.shape[-1] != self.dim:
            raise ValueError("Expected pooled features [B, S, F, dim]")
        batch, speakers, freqs, dim = x.shape
        if self.axis == "speaker":
            sequence = x.permute(0, 2, 1, 3).reshape(batch * freqs, speakers, dim)
        elif self.axis == "frequency":
            sequence = x.reshape(batch * speakers, freqs, dim)
        else:
            sequence = x.reshape(batch, speakers * freqs, dim)
        sequence_batch, sequence_length, _ = sequence.shape
        qkv = self.qkv(self.norm_attention(sequence)).reshape(
            sequence_batch, sequence_length, 3, self.heads, dim // self.heads)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        if self.axis != "speaker":
            positions = torch.arange(freqs, device=x.device)
            if self.axis == "joint":
                positions = positions.repeat(speakers)
            query = self._rotate_frequency_positions(query, positions)
            key = self._rotate_frequency_positions(key, positions)
        update = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0)
        update = self.projection(
            update.transpose(1, 2).reshape(sequence_batch, sequence_length, dim))
        if self.axis == "speaker":
            update = update.reshape(batch, freqs, speakers, dim).permute(0, 2, 1, 3)
        else:
            update = update.reshape(batch, speakers, freqs, dim)
        return x + update


class PooledBiMambaCTFPredictor(nn.Module):
    def __init__(self, input_dim=257, num_spk=2, ctf_taps=60, emb_dim=64,
                 pre_layers=4, post_layers=2, d_state=16, d_conv=4,
                 freq_bottleneck=16, freq_rank=16, local_kernel=5,
                 pooling_hidden=32, n_heads=4, ffn_dim=128,
                 head_dim=128, slot_embedding_std=0.02, eps=1e-5,
                 post_attention=True, frequency_refinement_position="after_pool",
                 time_module="bimamba", time_block_indices=None,
                 frequency_block_indices=None, pooling_channelwise=True,
                 pre_frequency_global=True, post_interaction="none",
                 post_interaction_order="after_frequency",
                 pre_frequency_module="light",
                 post_frequency_module="attention",
                 time_local_attention_window=0,
                 time_local_attention_query_chunk_size=64,
                 time_local_attention_sequence_batch_size=64):
        super().__init__()
        positive = (input_dim, num_spk, ctf_taps, emb_dim, pre_layers, d_state,
                    d_conv, freq_bottleneck, freq_rank, local_kernel, pooling_hidden,
                    n_heads, ffn_dim, head_dim)
        if min(positive) <= 0 or post_layers < 0 or local_kernel % 2 != 1:
            raise ValueError("Use positive dimensions, nonnegative post_layers and an odd local kernel")
        if num_spk != 2:
            raise ValueError("The sweep PIT wrapper requires two speakers")
        if not isinstance(post_attention, bool):
            raise ValueError("post_attention must be a boolean")
        if frequency_refinement_position not in ("after_pool", "before_pool"):
            raise ValueError("frequency_refinement_position must be 'after_pool' or 'before_pool'")
        if time_module not in ("bimamba", "bilstm"):
            raise ValueError("time_module must be 'bimamba' or 'bilstm'")
        if (isinstance(time_local_attention_window, bool)
                or not isinstance(time_local_attention_window, int)
                or time_local_attention_window < 0
                or (time_local_attention_window > 0
                    and time_local_attention_window % 2 != 1)):
            raise ValueError("time_local_attention_window must be zero or a positive odd integer")
        local_chunks = (time_local_attention_query_chunk_size,
                        time_local_attention_sequence_batch_size)
        if any(isinstance(value, bool) or not isinstance(value, int)
               or value <= 0 for value in local_chunks):
            raise ValueError("Local attention chunk sizes must be positive integers")
        if not isinstance(pooling_channelwise, bool):
            raise ValueError("pooling_channelwise must be a boolean")
        if not isinstance(pre_frequency_global, bool):
            raise ValueError("pre_frequency_global must be a boolean")
        if pre_frequency_module not in ("light", "attention", "bimamba"):
            raise ValueError("pre_frequency_module must be 'light', 'attention', or 'bimamba'")
        if post_frequency_module not in ("attention", "bimamba"):
            raise ValueError("post_frequency_module must be 'attention' or 'bimamba'")
        if pre_frequency_module != "light" and not pre_frequency_global:
            raise ValueError("pre_frequency_global selects a branch of the light module only")
        if post_frequency_module == "bimamba":
            if not post_attention:
                raise ValueError("post_attention=False cannot select a BiMamba replacement")
            if post_layers == 0:
                raise ValueError("A post-frequency replacement requires positive post_layers")
        if post_interaction not in ("none", "speaker", "joint", "frequency"):
            raise ValueError("post_interaction must be 'none', 'speaker', 'joint', or 'frequency'")
        if post_interaction_order not in ("after_frequency", "before_frequency"):
            raise ValueError("post_interaction_order must be 'after_frequency' or 'before_frequency'")
        if post_interaction != "none":
            if frequency_refinement_position != "after_pool":
                raise ValueError("Post-pooling interactions require frequency_refinement_position='after_pool'")
            if post_layers == 0:
                raise ValueError("Enabled post_interaction requires positive post_layers")
            if (isinstance(n_heads, bool) or not isinstance(n_heads, int)
                    or n_heads <= 0 or emb_dim % n_heads):
                raise ValueError("Use a positive integer n_heads dividing emb_dim")
            if post_interaction != "speaker" and (emb_dim // n_heads) % 2:
                raise ValueError("Frequency RoPE requires an even head dimension")
        time_indices = self._active_indices(time_block_indices, pre_layers, "time_block_indices")
        frequency_indices = self._active_indices(frequency_block_indices, pre_layers, "frequency_block_indices")
        if time_local_attention_window and not time_indices:
            raise ValueError("Local attention requires at least one active time stage")
        self.input_dim = input_dim
        self._num_spk = num_spk
        self.ctf_taps = ctf_taps
        self.frequency_refinement_position = frequency_refinement_position
        self.post_interaction = post_interaction
        self.post_interaction_order = post_interaction_order
        self.pre_frequency_module = pre_frequency_module
        self.post_frequency_module = post_frequency_module
        self.time_local_attention_window = time_local_attention_window
        self.time_local_attention_query_chunk_size = time_local_attention_query_chunk_size
        self.time_local_attention_sequence_batch_size = time_local_attention_sequence_batch_size
        self.pooling_channels = emb_dim if pooling_channelwise else 1
        self.encoder = nn.Conv2d(2, emb_dim, kernel_size=3, padding=1)
        if time_module == "bimamba":
            self.time_blocks = nn.ModuleList(
                TimeBiMamba(emb_dim, d_state, d_conv, eps) if i in time_indices else nn.Identity()
                for i in range(pre_layers))
        else:
            self.time_blocks = nn.ModuleList(
                TimeBiLSTM(emb_dim, eps) if i in time_indices else nn.Identity()
                for i in range(pre_layers))
        self.frequency_blocks = nn.ModuleList(
            LightFrequencyBlock(emb_dim, input_dim, freq_bottleneck, freq_rank, local_kernel, eps,
                                global_mlp=pre_frequency_global) if i in frequency_indices else nn.Identity()
            for i in range(pre_layers))
        self.pooling = nn.Sequential(nn.Linear(emb_dim, pooling_hidden), nn.GELU(),
                                     nn.Linear(pooling_hidden, num_spk * self.pooling_channels))
        self.slot_embeddings = nn.Parameter(torch.empty(num_spk, emb_dim))
        nn.init.normal_(self.slot_embeddings, std=slot_embedding_std)
        self.post_blocks = nn.ModuleList(
            FrequencyAttentionBlock(emb_dim, n_heads, ffn_dim, local_kernel, eps, attention=post_attention)
            for _ in range(post_layers))
        self.head = nn.Sequential(nn.LayerNorm(emb_dim, eps=eps), nn.Linear(emb_dim, head_dim),
                                  nn.SiLU(), nn.Linear(head_dim, 2 * ctf_taps))
        # Initialize additions last so the seed still gives every common
        # baseline parameter (including the CTF head) the exact same values.
        self.post_interaction_blocks = nn.ModuleList(
            PostPoolingAttention(emb_dim, n_heads, eps, post_interaction)
            for _ in range(post_layers) if post_interaction != "none")
        # Build replacements after all legacy modules. Independent CPU RNG
        # streams preserve common baseline weights and make the pre-only /
        # post-only mixers identical to their counterparts in the combined
        # experiment. Discarded placeholders are not registered parameters.
        seed = torch.initial_seed()

        def seeded_module(offset, factory):
            with torch.random.fork_rng(devices=[]):
                torch.default_generator.manual_seed((seed + offset) % (2**63 - 1))
                return factory()

        if pre_frequency_module != "light":
            for index in frequency_indices:
                attention = seeded_module(
                    100003 + 97 * index,
                    lambda: FrequencyAttentionBlock(
                        emb_dim, n_heads, ffn_dim, local_kernel, eps),
                )
                sequence = attention
                if pre_frequency_module == "bimamba":
                    sequence = seeded_module(
                        200003 + 97 * index,
                        lambda: FrequencyBiMambaBlock(
                            emb_dim, d_state, d_conv, eps, attention),
                    )
                self.frequency_blocks[index] = FramewiseFrequencyBlock(sequence)
        if post_frequency_module == "bimamba":
            for index, attention in enumerate(self.post_blocks):
                self.post_blocks[index] = seeded_module(
                    300007 + 97 * index,
                    lambda: FrequencyBiMambaBlock(
                        emb_dim, d_state, d_conv, eps, attention),
                )
        # Initialize the additive time modules last, with independent RNG
        # streams. Enabling them leaves every shared weight and RNG state
        # identical to the baseline; disabling adds no state-dict entries.
        self.time_local_attention_blocks = nn.ModuleList()
        if time_local_attention_window:
            for index in range(pre_layers):
                block = nn.Identity()
                if index in time_indices:
                    block = seeded_module(
                        400009 + 97 * index,
                        lambda: TimeLocalAttention(
                            emb_dim, n_heads, time_local_attention_window, eps,
                            time_local_attention_query_chunk_size,
                            time_local_attention_sequence_batch_size),
                    )
                self.time_local_attention_blocks.append(block)

    @staticmethod
    def _active_indices(indices, layers, name):
        if indices is None:
            return tuple(range(layers))
        if not isinstance(indices, (list, tuple)) or any(
                isinstance(i, bool) or not isinstance(i, int) for i in indices):
            raise ValueError(f"{name} must be a list of integer stage indices")
        if len(set(indices)) != len(indices) or any(i < 0 or i >= layers for i in indices):
            raise ValueError(f"{name} must contain unique indices in [0, pre_layers)")
        return tuple(indices)

    def forward(self, input, ilens=None, additional=None):
        if input.ndim != 3 or not torch.is_complex(input) or input.shape[-1] != self.input_dim:
            raise ValueError("Expected complex STFT [B, T, input_dim]")
        x = self.encoder(torch.stack((input.real, input.imag), 1).float()).permute(0, 2, 3, 1)
        if self.time_local_attention_window:
            for local_block, time_block, frequency_block in zip(
                    self.time_local_attention_blocks, self.time_blocks,
                    self.frequency_blocks):
                x = frequency_block(time_block(local_block(x)))
        else:
            for time_block, frequency_block in zip(self.time_blocks, self.frequency_blocks):
                x = frequency_block(time_block(x))
        batch, frames, freqs, dim = x.shape
        if self.frequency_refinement_position == "before_pool":
            sequence = x.reshape(batch * frames, freqs, dim)
            for block in self.post_blocks:
                sequence = block(sequence)
            x = sequence.reshape(batch, frames, freqs, dim)
        scores = self.pooling(x).reshape(batch, frames, freqs, self.num_spk, self.pooling_channels)
        # Match baseline padding behavior: softmax over the full batch time axis.
        weights = scores.softmax(dim=1)
        pooled = (weights * x.unsqueeze(3)).sum(1).permute(0, 2, 1, 3)
        pooled = pooled + self.slot_embeddings[None, :, None, :]
        if self.post_interaction == "none":
            # Keep the legacy layout and operation ordering exactly intact.
            pooled = pooled.reshape(batch * self.num_spk, freqs, dim)
            if self.frequency_refinement_position == "after_pool":
                for block in self.post_blocks:
                    pooled = block(pooled)
        else:
            for block, interaction in zip(self.post_blocks, self.post_interaction_blocks):
                if self.post_interaction_order == "before_frequency":
                    pooled = interaction(pooled)
                pooled = block(pooled.reshape(batch * self.num_spk, freqs, dim))
                pooled = pooled.reshape(batch, self.num_spk, freqs, dim)
                if self.post_interaction_order == "after_frequency":
                    pooled = interaction(pooled)
            pooled = pooled.reshape(batch * self.num_spk, freqs, dim)
        output = self.head(pooled).reshape(batch, self.num_spk, freqs, self.ctf_taps, 2).float()
        ctf = torch.complex(output[..., 0], output[..., 1]).contiguous()
        return ctf, ilens, OrderedDict()

    @property
    def num_spk(self):
        return self._num_spk

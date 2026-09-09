"""Header-based capacity planning for FLOAT WHAMR audio and RIRs."""

import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf


def estimate(wsj_root, noise_root, script_dir, rates, lengths, splits, mono, limit):
    cache = {}

    def frames(path, rate):
        path = str(path)
        if path not in cache:
            info = sf.info(path)
            if info.samplerate != 16000:
                raise ValueError('Expected 16 kHz source: ' + path)
            cache[path] = info.frames
        return math.ceil(cache[path] * rate / 16000)

    budgets = {}
    channels = 1 if mono else 2
    for split in splits:
        speech = pd.read_csv(Path(script_dir, 'data', 'mix_2_spk_filenames_{}.csv'.format(split))).set_index('output_filename')
        rooms = pd.read_csv(Path(script_dir, 'data', 'reverb_params_{}.csv'.format(split))).set_index('utterance_id')
        if not speech.index.is_unique or not rooms.index.is_unique:
            raise ValueError('Duplicate metadata IDs')
        with np.load(Path(noise_root, 'metadata', 'scaling_{}.npz'.format(split)), allow_pickle=True) as scaling:
            ids = scaling['utterance_id'][:limit]
            for rate in rates:
                for length in lengths:
                    for prefix in ('scaling_wsjmix', 'scaling_wham_speech', 'scaling_wham_noise'):
                        if '{}_{}_{}'.format(prefix, rate, length) not in scaling:
                            raise ValueError('Missing scaling metadata')
            for uid in ids:
                row = speech.loc[uid]
                amount = 0
                for rate_name in rates:
                    rate = int(rate_name.replace('k', '000'))
                    a = frames(Path(wsj_root, row.s1_path), rate)
                    b = frames(Path(wsj_root, row.s2_path), rate)
                    noise = frames(Path(noise_root, split, uid), rate)
                    for length in lengths:
                        size = min(a, b) if length == 'min' else max(noise, a, b)
                        # Four RIRs bounded by WhamRoom.max_rir_len, plus 11 audios.
                        rir_size = math.ceil(float(rooms.loc[uid].T60) * rate) + 2
                        amount += (11 * size + 4 * rir_size) * channels * 4
                        amount += 15 * 4096 + 8192  # WAV headers, allocation, metadata
                budgets[(split, uid)] = math.ceil(amount * 1.10)
        print('Capacity scan: {} {} utterances'.format(split, len(ids)), flush=True)
    return budgets


def free_bytes(path):
    path = Path(path).resolve()
    while not path.exists():
        path = path.parent
    return shutil.disk_usage(str(path)).free


def require_space(path, reserve, next_bytes):
    available = free_bytes(path)
    if available < reserve + next_bytes:
        raise RuntimeError('Insufficient free space: free={:.2f} GiB, reserve={:.2f} GiB, required={:.2f} GiB'.format(
            available / 2**30, reserve / 2**30, next_bytes / 2**30))

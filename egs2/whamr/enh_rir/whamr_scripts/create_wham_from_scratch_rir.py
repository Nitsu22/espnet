"""Generate all WHAMR variants and matching physical RIR WAVs."""

import os
import json
import hashlib
from pathlib import Path
import numpy as np
import soundfile as sf
import pandas as pd
from constants import SAMPLERATE
import argparse
from scipy.signal import resample_poly
from utils import read_scaled_wav, quantize, fix_length, create_wham_mixes, append_or_truncate


SCRIPT_DIR = Path(__file__).resolve().parent
FILELIST_STUB = str(SCRIPT_DIR / 'data' / 'mix_2_spk_filenames_{}.csv')

SINGLE_DIR = 'mix_single'
BOTH_DIR = 'mix_both'
CLEAN_DIR = 'mix_clean'
S1_DIR = 's1'
S2_DIR = 's2'
NOISE_DIR = 'noise'
RIR1_DIR = 'rir1'
RIR2_DIR = 'rir2'
SUFFIXES = ['_anechoic', '_reverb']

MONO = False
SPLITS = ['tr', 'cv', 'tt']
SAMPLE_RATES = ['8k', '16k']
DATA_LEN = ['min', 'max']


def _stack_source_rir(rir_list, source_index):
    # rir_list is indexed as [mic][source].
    max_len = max(len(mic_rirs[source_index]) for mic_rirs in rir_list)
    rir = np.zeros((max_len, len(rir_list)), dtype=np.float64)
    for mic_index, mic_rirs in enumerate(rir_list):
        mic_rir = np.asarray(mic_rirs[source_index], dtype=np.float64)
        rir[:len(mic_rir), mic_index] = mic_rir
    return rir


def _resample_rir(rir, sample_rate):
    if sample_rate == SAMPLERATE:
        return rir
    if SAMPLERATE % sample_rate != 0:
        raise ValueError('RIR sample rate must divide {}: {}'.format(SAMPLERATE, sample_rate))
    return resample_poly(rir, sample_rate, SAMPLERATE, axis=0)


def create_wham(wsj_root, wham_noise_path, output_root, sample_rates=None,
                data_lengths=None, splits=None, mono=None, limit=None,
                min_free_gib=100.0, estimate_only=False):
    from wham_room import WhamRoom
    import pyroomacoustics

    sample_rates = list(SAMPLE_RATES if sample_rates is None else sample_rates)
    data_lengths = list(DATA_LEN if data_lengths is None else data_lengths)
    splits = list(SPLITS if splits is None else splits)
    mono = MONO if mono is None else mono
    for values, allowed in ((sample_rates, {'8k', '16k'}),
                            (data_lengths, {'min', 'max'}),
                            (splits, {'tr', 'cv', 'tt'})):
        if not values or len(set(values)) != len(values) or not set(values) <= allowed:
            raise ValueError('Invalid or duplicate selection: {}'.format(values))
    if limit is not None and limit < 1:
        raise ValueError('limit must be positive')
    output_root = str(Path(output_root).resolve())
    from rir_plus_capacity import estimate, free_bytes, require_space
    if not np.isfinite(min_free_gib) or min_free_gib < 0:
        raise ValueError('min_free_gib must be finite and nonnegative')
    if not estimate_only and Path(output_root).exists():
        raise FileExistsError(output_root)
    budgets = estimate(wsj_root, wham_noise_path, SCRIPT_DIR, sample_rates,
                       data_lengths, splits, mono, limit)
    capacity = dict(estimated_bytes_with_margin=sum(budgets.values()),
                    free_bytes=free_bytes(output_root), reserve_bytes=int(min_free_gib * 2**30),
                    utterances=len(budgets), margin=1.10)
    print(json.dumps(capacity, indent=2), flush=True)
    if estimate_only:
        return capacity
    require_space(output_root, capacity['reserve_bytes'], sum(budgets.values()))
    # Never mix a new run with existing audio or silently overwrite another run.
    Path(output_root).mkdir(parents=True, exist_ok=False)
    config = dict(sample_rates=sample_rates, data_lengths=data_lengths, splits=splits,
                  mono=mono, limit=limit, wsj_root=str(Path(wsj_root).resolve()),
                  wham_noise_root=str(Path(wham_noise_path).resolve()),
                  rir_definition='unscaled physical RIR; full delay and tail; FLOAT WAV',
                  channels='left' if mono else 'left,right',
                  pyroomacoustics_version=pyroomacoustics.__version__,
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    Path(output_root, 'generation_config.json').write_text(json.dumps(config, indent=2))
    Path(output_root, 'capacity.json').write_text(json.dumps(capacity, indent=2))
    LEFT_CH_IND = 0
    if mono:
        ch_ind = LEFT_CH_IND
    else:
        ch_ind = [0, 1]

    scaling_npz_stub = os.path.join(wham_noise_path, 'metadata', 'scaling_{}.npz')
    reverb_param_stub = str(SCRIPT_DIR / 'data' / 'reverb_params_{}.csv')

    for splt in splits:

        wsjmix_path = FILELIST_STUB.format(splt)
        wsjmix_df = pd.read_csv(wsjmix_path)

        scaling_npz_path = scaling_npz_stub.format(splt)
        scaling_npz = np.load(scaling_npz_path, allow_pickle=True)

        noise_path = os.path.join(wham_noise_path, splt)

        reverb_param_path = reverb_param_stub.format(splt)
        reverb_param_df = pd.read_csv(reverb_param_path)

        for wav_dir in ['wav' + sr for sr in sample_rates]:
            for datalen_dir in data_lengths:
                output_path = os.path.join(output_root, wav_dir, datalen_dir, splt)
                for sfx in SUFFIXES:
                    os.makedirs(os.path.join(output_path, CLEAN_DIR+sfx), exist_ok=True)
                    os.makedirs(os.path.join(output_path, SINGLE_DIR+sfx), exist_ok=True)
                    os.makedirs(os.path.join(output_path, BOTH_DIR+sfx), exist_ok=True)
                    os.makedirs(os.path.join(output_path, S1_DIR+sfx), exist_ok=True)
                    os.makedirs(os.path.join(output_path, S2_DIR+sfx), exist_ok=True)
                for source in (RIR1_DIR, RIR2_DIR):
                    for suffix in SUFFIXES:
                        os.makedirs(os.path.join(output_path, source + suffix), exist_ok=True)
                os.makedirs(os.path.join(output_path, NOISE_DIR), exist_ok=True)

        utt_ids = scaling_npz['utterance_id']
        start_samp_16k = scaling_npz['speech_start_sample_16k']

        if len(set(utt_ids)) != len(utt_ids):
            raise ValueError('Duplicate utterance IDs')
        for i_utt, output_name in enumerate(utt_ids[:limit]):
            require_space(output_root, capacity['reserve_bytes'], budgets[(splt, output_name)])
            if Path(output_name).name != output_name:
                raise ValueError('Unsafe utterance filename: {}'.format(output_name))
            utt_row = reverb_param_df[reverb_param_df['utterance_id'] == output_name]
            room = WhamRoom([utt_row['room_x'].iloc[0], utt_row['room_y'].iloc[0], utt_row['room_z'].iloc[0]],
                            [[utt_row['micL_x'].iloc[0], utt_row['micL_y'].iloc[0], utt_row['mic_z'].iloc[0]],
                             [utt_row['micR_x'].iloc[0], utt_row['micR_y'].iloc[0], utt_row['mic_z'].iloc[0]]],
                            [utt_row['s1_x'].iloc[0], utt_row['s1_y'].iloc[0], utt_row['s1_z'].iloc[0]],
                            [utt_row['s2_x'].iloc[0], utt_row['s2_y'].iloc[0], utt_row['s2_z'].iloc[0]],
                            utt_row['T60'].iloc[0])
            room.generate_rirs()
            physical_rirs = {
                'rir{}_{}'.format(source + 1, condition): _stack_source_rir(rirs, source)
                for condition, rirs in (('anechoic', room.rir_anechoic),
                                        ('reverb', room.rir_reverberant))
                for source in (0, 1)
            }

            # read the 16kHz unscaled speech files, but make sure to add all 'max' padding to end of utterances
            # for synthesizing all the reverb tails
            utt_row = wsjmix_df[wsjmix_df['output_filename'] == output_name]
            s1_path = os.path.join(wsj_root, utt_row['s1_path'].iloc[0])
            s2_path = os.path.join(wsj_root, utt_row['s2_path'].iloc[0])
            s1_temp = quantize(read_scaled_wav(s1_path, 1))
            s2_temp = quantize(read_scaled_wav(s2_path, 1))
            s1_temp, s2_temp = fix_length(s1_temp, s2_temp, 'max')
            noise_samples_temp = read_scaled_wav(os.path.join(noise_path, output_name), 1)
            s1_temp, s2_temp, noise_samples_temp = append_or_truncate(s1_temp, s2_temp,
                                                                      noise_samples_temp, 'max',
                                                                      start_samp_16k=0)  # don't pad beginning yet

            room.add_audio(s1_temp, s2_temp)

            anechoic = room.generate_audio(anechoic=True, fs=sample_rates)
            reverberant = room.generate_audio(fs=sample_rates)

            for sr_i, sr_dir in enumerate(sample_rates):
                wav_dir = 'wav' + sr_dir
                if sr_dir == '8k':
                    sr = 8000
                    downsample = True
                else:
                    sr = SAMPLERATE
                    downsample = False

                output_rirs = {name: _resample_rir(h, sr)[:, ch_ind]
                               for name, h in physical_rirs.items()}

                for datalen_dir in data_lengths:
                    output_path = os.path.join(output_root, wav_dir, datalen_dir, splt)
                    for name, h in output_rirs.items():
                        sf.write(os.path.join(output_path, name, output_name),
                                 h, sr, subtype='FLOAT')

                    wsjmix_key = 'scaling_wsjmix_{}_{}'.format(sr_dir, datalen_dir)
                    wham_speech_key = 'scaling_wham_speech_{}_{}'.format(sr_dir, datalen_dir)
                    wham_noise_key = 'scaling_wham_noise_{}_{}'.format(sr_dir, datalen_dir)

                    utt_row = wsjmix_df[wsjmix_df['output_filename'] == output_name]
                    s1_path = os.path.join(wsj_root, utt_row['s1_path'].iloc[0])
                    s2_path = os.path.join(wsj_root, utt_row['s2_path'].iloc[0])

                    s1 = read_scaled_wav(s1_path, scaling_npz[wsjmix_key][i_utt][0], downsample)
                    s1 = quantize(s1) * scaling_npz[wham_speech_key][i_utt]
                    s2 = read_scaled_wav(s2_path, scaling_npz[wsjmix_key][i_utt][1], downsample)
                    s2 = quantize(s2) * scaling_npz[wham_speech_key][i_utt]

                    # Make relative source energy of anechoic sources same with original in mono (left channel) case
                    s1_spatial_scaling = np.sqrt(np.sum(s1 ** 2) / np.sum(anechoic[sr_i][0, LEFT_CH_IND, :] ** 2))
                    s2_spatial_scaling = np.sqrt(np.sum(s2 ** 2) / np.sum(anechoic[sr_i][1, LEFT_CH_IND, :] ** 2))

                    # Physical RIRs intentionally exclude these utterance gains.
                    # Retain the gains and source paths for reproducible reuse.
                    with open(os.path.join(output_path, 'utterances.jsonl'), 'a') as metadata:
                        metadata.write(json.dumps(dict(
                            utterance_id=Path(output_name).stem,
                            source1=str(Path(s1_path).resolve()),
                            source2=str(Path(s2_path).resolve()),
                            s1_spatial_scaling=float(s1_spatial_scaling),
                            s2_spatial_scaling=float(s2_spatial_scaling),
                            noise_scaling=float(scaling_npz[wham_noise_key][i_utt]),
                            speech_start_sample_16k=int(start_samp_16k[i_utt]),
                            target_t60=float(room.T60),
                        )) + '\n')

                    noise_samples_full = read_scaled_wav(os.path.join(noise_path, output_name),
                                                         scaling_npz[wham_noise_key][i_utt],
                                                         downsample_8K=downsample, mono=mono)
                    if datalen_dir == 'max':
                        out_len = len(noise_samples_full)
                    else:
                        out_len = np.minimum(len(s1), len(s2))

                    s1_anechoic, s2_anechoic = fix_length(anechoic[sr_i][0, ch_ind, :out_len].T * s1_spatial_scaling,
                                                          anechoic[sr_i][1, ch_ind, :out_len].T * s2_spatial_scaling,
                                                          datalen_dir)
                    s1_reverb, s2_reverb = fix_length(reverberant[sr_i][0, ch_ind, :out_len].T * s1_spatial_scaling,
                                                      reverberant[sr_i][1, ch_ind, :out_len].T * s2_spatial_scaling,
                                                      datalen_dir)

                    sources = [(s1_anechoic, s2_anechoic), (s1_reverb, s2_reverb)]
                    for i_sfx, (sfx, source_pair) in enumerate(zip(SUFFIXES, sources)):
                        s1_samples, s2_samples, noise_samples = append_or_truncate(source_pair[0], source_pair[1],
                                                                                   noise_samples_full, datalen_dir,
                                                                                   start_samp_16k[i_utt], downsample)

                        mix_clean, mix_single, mix_both = create_wham_mixes(s1_samples, s2_samples, noise_samples)

                        # write audio
                        samps = [mix_clean, mix_single, mix_both, s1_samples, s2_samples]
                        dirs = [CLEAN_DIR, SINGLE_DIR, BOTH_DIR, S1_DIR, S2_DIR]
                        for dir, samp in zip(dirs, samps):
                            sf.write(os.path.join(output_path, dir+sfx, output_name), samp,
                                     sr, subtype='FLOAT')

                        if i_sfx == 0: # only write noise once as it doesn't change between anechoic and reverberant
                            sf.write(os.path.join(output_path, NOISE_DIR, output_name), noise_samples,
                                     sr, subtype='FLOAT')

            if (i_utt + 1) % 500 == 0:
                progress = dict(split=splt, completed=i_utt + 1,
                                total=len(utt_ids[:limit]), free_bytes=free_bytes(output_root))
                Path(output_root, 'progress.json').write_text(json.dumps(progress, indent=2))
                print(json.dumps(progress), flush=True)

        scaling_npz.close()

    # Each selection has one SCP per signal type; identical keys join all variants.
    for split in splits:
        for rate in sample_rates:
            for length in data_lengths:
                root = Path(output_root, 'wav' + rate, length, split)
                manifest = Path(output_root, 'manifests', '{}_{}_{}'.format(split, length, rate))
                manifest.mkdir(parents=True)
                counts = {}
                for signal_dir in sorted(root.iterdir()):
                    if not signal_dir.is_dir():
                        continue
                    files = sorted(signal_dir.glob('*.wav'))
                    with (manifest / (signal_dir.name + '.scp')).open('w') as stream:
                        for wav in files:
                            stream.write('{} {}\n'.format(wav.stem, wav))
                    counts[signal_dir.name] = len(files)
                (manifest / 'counts.json').write_text(json.dumps(counts, indent=2))
    Path(output_root, 'generation_complete.json').write_text(json.dumps(config, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', default=str(SCRIPT_DIR.parent / 'data_rir_plus'),
                        help='New output root; refuses existing directories.')
    parser.add_argument('--wsj0-root', required=True)
    parser.add_argument('--wham-noise-root', required=True)
    parser.add_argument('--sample-rates', nargs='+', choices=['8k', '16k'], default=SAMPLE_RATES)
    parser.add_argument('--data-lengths', nargs='+', choices=['min', 'max'], default=DATA_LEN)
    parser.add_argument('--splits', nargs='+', choices=['tr', 'cv', 'tt'], default=SPLITS)
    parser.add_argument('--mono', action='store_true', default=MONO,
                        help='Save left microphone only; default saves both microphones.')
    parser.add_argument('--limit', type=int, help='Smoke test: first N utterances per split.')
    parser.add_argument('--min-free-gib', type=float, default=100.0)
    parser.add_argument('--estimate-only', action='store_true')
    args = parser.parse_args()
    create_wham(args.wsj0_root, args.wham_noise_root, args.output_dir,
                args.sample_rates, args.data_lengths, args.splits, args.mono, args.limit,
                args.min_free_gib, args.estimate_only)

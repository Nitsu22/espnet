import os
import numpy as np
import pandas as pd
from constants import SAMPLERATE
import argparse
from utils import read_scaled_wav, quantize, fix_length, append_or_truncate
from wham_room import WhamRoom


FILELIST_STUB = os.path.join('data', 'mix_2_spk_filenames_{}.csv')

MONO = False
SPLITS = ['tr', 'cv', 'tt']
SAMPLE_RATES = ['8k']
DATA_LEN = ['min']

def _pad_rir(rir_list, target_len):
    # rir_list: list (n_mics) of list (n_src) of 1D arrays
    n_mics = len(rir_list)
    if n_mics == 0:
        return np.zeros((0, 0, target_len), dtype=np.float64)
    n_src = len(rir_list[0])
    rir_out = np.zeros((n_mics, n_src, target_len), dtype=np.float64)
    for m, mic_rirs in enumerate(rir_list):
        if len(mic_rirs) != n_src:
            raise ValueError("Inconsistent number of sources across microphones in RIR list")
        for s, rir in enumerate(mic_rirs):
            rir_out[m, s, :len(rir)] = rir
    return rir_out


def _path_in_split(npz_root, wav_dir, datalen_dir, split, subdir, utt_id, ext):
    utt_base = os.path.splitext(str(utt_id))[0]
    return os.path.join(npz_root, wav_dir, datalen_dir, split, subdir, utt_base + ext)


def create_wham(wsj_root, wham_noise_path, output_root, npz_root=None, write_audio=True):
    if npz_root is None:
        npz_root = output_root

    if write_audio:
        raise RuntimeError("Audio output is disabled in create_wham_from_scratch_npz.py; use --skip-audio.")

    scaling_npz_stub = os.path.join(wham_noise_path, 'metadata', 'scaling_{}.npz')
    reverb_param_stub = os.path.join('data', 'reverb_params_{}.csv')

    for splt in SPLITS:

        wsjmix_path = FILELIST_STUB.format(splt)
        wsjmix_df = pd.read_csv(wsjmix_path)

        scaling_npz_path = scaling_npz_stub.format(splt)
        scaling_npz = np.load(scaling_npz_path, allow_pickle=True)

        noise_path = os.path.join(wham_noise_path, splt)

        reverb_param_path = reverb_param_stub.format(splt)
        reverb_param_df = pd.read_csv(reverb_param_path)

        for wav_dir in ['wav' + sr for sr in SAMPLE_RATES]:
            for datalen_dir in DATA_LEN:
                for subdir in [
                    "npz",
                    "s1_base_npz",
                    "s2_base_npz",
                    "s1_temp_npz",
                    "s2_temp_npz",
                    "noise_base_npz",
                    "rir_npz",
                    "room_param_npz",
                ]:
                    os.makedirs(os.path.join(npz_root, wav_dir, datalen_dir, splt, subdir), exist_ok=True)

        utt_ids = scaling_npz['utterance_id']
        start_samp_16k = scaling_npz['speech_start_sample_16k']

        for i_utt, output_name in enumerate(utt_ids):
            output_name = str(output_name)
            utt_row = reverb_param_df[reverb_param_df['utterance_id'] == output_name]
            room = WhamRoom([utt_row['room_x'].iloc[0], utt_row['room_y'].iloc[0], utt_row['room_z'].iloc[0]],
                            [[utt_row['micL_x'].iloc[0], utt_row['micL_y'].iloc[0], utt_row['mic_z'].iloc[0]],
                             [utt_row['micR_x'].iloc[0], utt_row['micR_y'].iloc[0], utt_row['mic_z'].iloc[0]]],
                            [utt_row['s1_x'].iloc[0], utt_row['s1_y'].iloc[0], utt_row['s1_z'].iloc[0]],
                            [utt_row['s2_x'].iloc[0], utt_row['s2_y'].iloc[0], utt_row['s2_z'].iloc[0]],
                            utt_row['T60'].iloc[0])
            room.generate_rirs()
            room_dim = np.array([utt_row['room_x'].iloc[0], utt_row['room_y'].iloc[0], utt_row['room_z'].iloc[0]],
                                dtype=np.float64)
            mic_pos = np.array([[utt_row['micL_x'].iloc[0], utt_row['micL_y'].iloc[0], utt_row['mic_z'].iloc[0]],
                                [utt_row['micR_x'].iloc[0], utt_row['micR_y'].iloc[0], utt_row['mic_z'].iloc[0]]],
                               dtype=np.float64)
            s1_pos = np.array([utt_row['s1_x'].iloc[0], utt_row['s1_y'].iloc[0], utt_row['s1_z'].iloc[0]],
                              dtype=np.float64)
            s2_pos = np.array([utt_row['s2_x'].iloc[0], utt_row['s2_y'].iloc[0], utt_row['s2_z'].iloc[0]],
                              dtype=np.float64)
            t60 = float(utt_row['T60'].iloc[0])

            # read the 16kHz unscaled speech files, but make sure to add all 'max' padding to end of utterances
            # for synthesizing all the reverb tails
            utt_row = wsjmix_df[wsjmix_df['output_filename'] == output_name]
            s1_path = os.path.join(wsj_root, utt_row['s1_path'].iloc[0])
            s2_path = os.path.join(wsj_root, utt_row['s2_path'].iloc[0])
            s1_temp = quantize(read_scaled_wav(s1_path, 1))
            s2_temp = quantize(read_scaled_wav(s2_path, 1))
            s1_temp, s2_temp = fix_length(s1_temp, s2_temp, 'max')
            noise_samples_temp = read_scaled_wav(os.path.join(noise_path, output_name), 1)
            s1_temp, s2_temp, _ = append_or_truncate(s1_temp, s2_temp,
                                                     noise_samples_temp, 'max',
                                                     start_samp_16k=0)  # don't pad beginning yet

            for sr_i, sr_dir in enumerate(SAMPLE_RATES):
                wav_dir = 'wav' + sr_dir
                if sr_dir == '8k':
                    sr = 8000
                    downsample = True
                else:
                    sr = SAMPLERATE
                    downsample = False

                for datalen_dir in DATA_LEN:
                    wsjmix_key = 'scaling_wsjmix_{}_{}'.format(sr_dir, datalen_dir)
                    wham_speech_key = 'scaling_wham_speech_{}_{}'.format(sr_dir, datalen_dir)
                    wham_noise_key = 'scaling_wham_noise_{}_{}'.format(sr_dir, datalen_dir)

                    utt_row = wsjmix_df[wsjmix_df['output_filename'] == output_name]
                    s1_path = os.path.join(wsj_root, utt_row['s1_path'].iloc[0])
                    s2_path = os.path.join(wsj_root, utt_row['s2_path'].iloc[0])

                    s1_base = quantize(read_scaled_wav(s1_path, scaling_npz[wsjmix_key][i_utt][0], downsample))
                    s2_base = quantize(read_scaled_wav(s2_path, scaling_npz[wsjmix_key][i_utt][1], downsample))

                    noise_base = read_scaled_wav(os.path.join(noise_path, output_name),
                                                 1,
                                                 downsample_8K=downsample, mono=MONO)

                    s1_base_path = _path_in_split(npz_root, wav_dir, datalen_dir, splt, "s1_base_npz", output_name, ".npy")
                    s2_base_path = _path_in_split(npz_root, wav_dir, datalen_dir, splt, "s2_base_npz", output_name, ".npy")
                    s1_temp_path = _path_in_split(npz_root, wav_dir, datalen_dir, splt, "s1_temp_npz", output_name, ".npy")
                    s2_temp_path = _path_in_split(npz_root, wav_dir, datalen_dir, splt, "s2_temp_npz", output_name, ".npy")
                    noise_base_path = _path_in_split(npz_root, wav_dir, datalen_dir, splt, "noise_base_npz", output_name, ".npy")
                    rir_path = _path_in_split(npz_root, wav_dir, datalen_dir, splt, "rir_npz", output_name, ".npz")
                    room_param_path = _path_in_split(npz_root, wav_dir, datalen_dir, splt, "room_param_npz", output_name, ".npz")
                    npz_path = _path_in_split(npz_root, wav_dir, datalen_dir, splt, "npz", output_name, ".npz")

                    np.save(s1_base_path, np.asarray(s1_base, dtype=np.float64))
                    np.save(s2_base_path, np.asarray(s2_base, dtype=np.float64))
                    np.save(s1_temp_path, np.asarray(s1_temp, dtype=np.float64))
                    np.save(s2_temp_path, np.asarray(s2_temp, dtype=np.float64))
                    np.save(noise_base_path, np.asarray(noise_base, dtype=np.float64))

                    rir_len = int(room.max_rir_len)
                    rir_anechoic = _pad_rir(room.rir_anechoic, rir_len)
                    rir_reverberant = _pad_rir(room.rir_reverberant, rir_len)
                    np.savez(
                        rir_path,
                        rir_anechoic=rir_anechoic,
                        rir_reverberant=rir_reverberant,
                        fs=np.array(room.fs, dtype=np.int64),
                        rir_len=np.array(rir_len, dtype=np.int64),
                    )
                    np.savez(
                        room_param_path,
                        room_dim=room_dim,
                        mic_pos=mic_pos,
                        s1_pos=s1_pos,
                        s2_pos=s2_pos,
                        T60=np.array(t60, dtype=np.float64),
                        max_rir_len=np.array(room.max_rir_len, dtype=np.int64),
                        fs=np.array(room.fs, dtype=np.int64),
                    )

                    s1_base_rel = os.path.join("s1_base_npz", os.path.basename(s1_base_path))
                    s2_base_rel = os.path.join("s2_base_npz", os.path.basename(s2_base_path))
                    s1_temp_rel = os.path.join("s1_temp_npz", os.path.basename(s1_temp_path))
                    s2_temp_rel = os.path.join("s2_temp_npz", os.path.basename(s2_temp_path))
                    noise_base_rel = os.path.join("noise_base_npz", os.path.basename(noise_base_path))
                    rir_rel = os.path.join("rir_npz", os.path.basename(rir_path))
                    room_param_rel = os.path.join("room_param_npz", os.path.basename(room_param_path))
                    np.savez(
                        npz_path,
                        format_version=np.array("whamr_npz_v2"),
                        utt_id=np.array(output_name),
                        split=np.array(splt),
                        sample_rate=np.array(sr, dtype=np.int64),
                        data_len=np.array(datalen_dir),
                        mono=np.array(MONO),
                        start_samp_16k=np.array(start_samp_16k[i_utt], dtype=np.int64),
                        wsjmix_scale=np.array(scaling_npz[wsjmix_key][i_utt], dtype=np.float64),
                        wham_speech_scale=np.array(scaling_npz[wham_speech_key][i_utt], dtype=np.float64),
                        wham_noise_scale=np.array(scaling_npz[wham_noise_key][i_utt], dtype=np.float64),
                        s1_base_relpath=np.array(s1_base_rel),
                        s2_base_relpath=np.array(s2_base_rel),
                        s1_temp_relpath=np.array(s1_temp_rel),
                        s2_temp_relpath=np.array(s2_temp_rel),
                        noise_base_relpath=np.array(noise_base_rel),
                        rir_relpath=np.array(rir_rel),
                        room_param_relpath=np.array(room_param_rel),
                        s1_path=np.array(s1_path),
                        s2_path=np.array(s2_path),
                        noise_path=np.array(os.path.join(noise_path, output_name)),
                    )

            if (i_utt + 1) % 500 == 0:
                print('Completed {} of {} utterances'.format(i_utt + 1, len(wsjmix_df)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for writing WHAMR npz metadata.')
    parser.add_argument('--wsj0-root', type=str, required=True,
                        help='Path to the folder containing wsj0/')
    parser.add_argument('--wham-noise-root', type=str, required=True,
                        help='Path to the downloaded and unzipped wham folder containing metadata/')
    parser.add_argument('--npz-dir', type=str, default=None,
                        help='Base directory for npz metadata (default: <output-dir>).')
    parser.add_argument('--skip-audio', action='store_true',
                        help='Skip writing wav outputs and only save npz.')
    args = parser.parse_args()
    create_wham(args.wsj0_root, args.wham_noise_root, args.output_dir,
                npz_root=args.npz_dir, write_audio=not args.skip_audio)

#!/usr/bin/env python3
"""t-SNE visualization for fixed-RIR mixture embeddings.

This script selects the first N utterance pairs from a Kaldi-style data dir,
re-synthesizes each mixture with the first K fixed RIRs, extracts spatial
embeddings from a trained NPZ spatial-encoder model, and saves t-SNE outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

# Allow `import espnet2` regardless of current working directory.
def _find_repo_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "espnet2").is_dir():
            return candidate
    raise RuntimeError("Failed to locate repo root containing espnet2/")


REPO_ROOT = _find_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from espnet2.tasks.enh_se_npz import SpatialEncoderTask
from espnet2.train.preprocessor_npz import NpzSwapRirPreprocessor


def _read_scp_with_order(path: Path) -> Tuple[List[str], Dict[str, str]]:
    order: List[str] = []
    mapping: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            utt_id, value = line.split(maxsplit=1)
            order.append(utt_id)
            mapping[utt_id] = value
    return order, mapping


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return path


def _select_pair_uids(
    all_uids: List[str],
    num_pairs: int,
    mode: str,
    pair_seed: Optional[int],
) -> List[str]:
    if num_pairs > len(all_uids):
        raise ValueError(
            f"Requested {num_pairs} pairs, but only {len(all_uids)} available"
        )
    if mode == "head":
        return all_uids[:num_pairs]
    if mode == "random":
        rng = np.random.default_rng(pair_seed)
        indices = rng.choice(len(all_uids), size=num_pairs, replace=False)
        return [all_uids[int(i)] for i in indices]
    raise ValueError(f"Unsupported pair-selection mode: {mode}")


def _to_tensor_batch_mixture(wav: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    x = np.asarray(wav, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]  # [B=1, T]
        lengths = np.array([x.shape[1]], dtype=np.int64)
        return torch.from_numpy(x), torch.from_numpy(lengths)

    if x.ndim != 2:
        raise ValueError(f"Expected 1D/2D waveform, got shape={x.shape}")

    # Canonical shape for model input is [B, T, C].
    if x.shape[0] >= x.shape[1]:
        # likely [T, C]
        t_c = x
    else:
        # likely [C, T]
        t_c = x.T
    t_c = t_c[None, :, :]
    lengths = np.array([t_c.shape[1]], dtype=np.int64)
    return torch.from_numpy(t_c), torch.from_numpy(lengths)


def _extract_embedding(
    model: torch.nn.Module,
    speech_mix: np.ndarray,
    sample_rate: int,
    device: str,
) -> np.ndarray:
    speech, lengths = _to_tensor_batch_mixture(speech_mix)
    speech = speech.to(device)
    lengths = lengths.to(device)

    with torch.no_grad():
        features, flens = model.encoder(speech, lengths, fs=sample_rate)
        if speech.dim() == 2:
            num_channels = 1
        else:
            num_channels = speech.shape[2]
        emb = model.spatial_encoder(features, flens, num_channels=num_channels)

    vec = emb.squeeze(0).detach().cpu().numpy().astype(np.float32)
    return vec


def _save_meta_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    if not rows:
        raise ValueError("No metadata rows to save")
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_tsne(path: Path, tsne_xy: np.ndarray, rows: List[Dict[str, str]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [r["rir_label"] for r in rows]
    unique_labels = sorted(set(labels))

    palette = plt.get_cmap("tab20")
    cmap = {label: palette(i % palette.N) for i, label in enumerate(unique_labels)}

    plt.figure(figsize=(8, 6), dpi=150)
    for label in unique_labels:
        idx = [i for i, v in enumerate(labels) if v == label]
        pts = tsne_xy[idx]
        plt.scatter(pts[:, 0], pts[:, 1], s=45, alpha=0.85, color=cmap[label], label=label)

    # Annotate with pair index for scanability.
    for i, row in enumerate(rows):
        plt.text(
            tsne_xy[i, 0],
            tsne_xy[i, 1],
            row["pair_idx"],
            fontsize=7,
            alpha=0.75,
            ha="left",
            va="bottom",
        )

    plt.title(f"t-SNE of Mixture Embeddings (Fixed {len(unique_labels)} RIR)")
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=str, default="valid.loss.best.pth")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--num-pairs", type=int, default=5)
    parser.add_argument(
        "--pair-selection",
        type=str,
        default="head",
        choices=["random", "head"],
        help="How to choose anchor-like pair UIDs from npz.scp.",
    )
    parser.add_argument(
        "--pair-seed",
        type=int,
        default=None,
        help="Seed for pair-selection=random. If omitted, pairs change each run.",
    )
    parser.add_argument("--fixed-rir-count", type=int, default=2)
    parser.add_argument("--sample-rate", type=int, default=8000)
    parser.add_argument("--mix-type", type=str, default="both", choices=["both", "single", "clean"])
    parser.add_argument("--ref-condition", type=str, default="reverb", choices=["reverb", "anechoic"])
    parser.add_argument(
        "--anchor-single-channel",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, mimic preprocessor_npz anchor path and use speech[:, 0] for multi-channel inputs.",
    )
    parser.add_argument("--perplexity", type=float, default=5.0)
    parser.add_argument("--learning-rate", default="auto")
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.num_pairs <= 0:
        raise ValueError("--num-pairs must be > 0")
    if args.fixed_rir_count <= 0:
        raise ValueError("--fixed-rir-count must be > 0")

    checkpoint_path = _require(args.exp_dir / args.checkpoint)
    config_path = _require(args.exp_dir / "config.yaml")

    scp_names = [
        "npz.scp",
        "spk1_base_npz.scp",
        "spk2_base_npz.scp",
        "spk1_temp_npz.scp",
        "spk2_temp_npz.scp",
        "noise_base_npz.scp",
        "rir_npz.scp",
        "room_param_npz.scp",
    ]
    scp_paths = {name: _require(args.data_dir / name) for name in scp_names}

    npz_order, npz_map = _read_scp_with_order(scp_paths["npz.scp"])
    s1_map = _read_scp_with_order(scp_paths["spk1_base_npz.scp"])[1]
    s2_map = _read_scp_with_order(scp_paths["spk2_base_npz.scp"])[1]
    s1t_map = _read_scp_with_order(scp_paths["spk1_temp_npz.scp"])[1]
    s2t_map = _read_scp_with_order(scp_paths["spk2_temp_npz.scp"])[1]
    noise_map = _read_scp_with_order(scp_paths["noise_base_npz.scp"])[1]
    rir_order, rir_map = _read_scp_with_order(scp_paths["rir_npz.scp"])
    room_map = _read_scp_with_order(scp_paths["room_param_npz.scp"])[1]

    pair_uids = _select_pair_uids(
        all_uids=npz_order,
        num_pairs=args.num_pairs,
        mode=args.pair_selection,
        pair_seed=args.pair_seed,
    )

    fixed_rir_uids = rir_order[: args.fixed_rir_count]
    if len(fixed_rir_uids) < args.fixed_rir_count:
        raise ValueError(
            f"Requested {args.fixed_rir_count} fixed RIRs, but only {len(fixed_rir_uids)} available"
        )

    for rir_uid in fixed_rir_uids:
        if rir_uid not in room_map:
            raise KeyError(f"RIR uid {rir_uid} not found in room_param_npz.scp")

    model, _ = SpatialEncoderTask.build_model_from_file(
        config_file=config_path,
        model_file=checkpoint_path,
        device=args.device,
    )
    model.eval()

    preproc = NpzSwapRirPreprocessor(
        train=False,
        mix_type=args.mix_type,
        ref_condition=args.ref_condition,
        sample_rate=args.sample_rate,
        force_single_channel=False,
        channel_reordering=False,
        speech_segment=None,
        contrastive_enable=False,
        anchor_single_channel=args.anchor_single_channel,
        contrastive_num_neg=1,
    )

    embeddings: List[np.ndarray] = []
    meta_rows: List[Dict[str, str]] = []

    for pair_idx, uid in enumerate(pair_uids, start=1):
        required_maps = {
            "npz": npz_map,
            "s1": s1_map,
            "s2": s2_map,
            "s1t": s1t_map,
            "s2t": s2t_map,
            "noise": noise_map,
        }
        missing = [name for name, mp in required_maps.items() if uid not in mp]
        if missing:
            raise KeyError(f"UID {uid} missing entries: {', '.join(missing)}")

        bundle = preproc._load_npz_bundle(npz_map[uid])

        for rir_idx, rir_uid in enumerate(fixed_rir_uids, start=1):
            rir_npz, room_dim, mic_pos, s1_pos, s2_pos, t60, room_fs = preproc._load_room_and_rir(
                rir_map[rir_uid], room_map[rir_uid]
            )

            speech_mix, s1_samples, s2_samples = preproc._synthesize_mix(
                sample_rate=bundle["sample_rate"],
                data_len=bundle["data_len"],
                start_samp_16k=bundle["start_samp_16k"],
                wsjmix_scale=bundle["wsjmix_scale"],
                wham_speech_scale=bundle["wham_speech_scale"],
                wham_noise_scale=bundle["wham_noise_scale"],
                mono=bundle["mono"],
                s1_base=bundle["s1_base"],
                s2_base=bundle["s2_base"],
                s1_temp=bundle["s1_temp"],
                s2_temp=bundle["s2_temp"],
                noise_base=bundle["noise_base"],
                rir_npz=rir_npz,
                room_dim=room_dim,
                mic_pos=mic_pos,
                s1_pos=s1_pos,
                s2_pos=s2_pos,
                t60=t60,
                room_fs=room_fs,
                s1_scale_factor=1.0,
                s2_scale_factor=1.0,
            )
            speech_mix = preproc._apply_postprocess(
                uid=uid,
                speech_mix=speech_mix,
                s1_samples=s1_samples,
                s2_samples=s2_samples,
                crop=None,
            )
            # Keep the same order as preprocessor_npz._speech_process_contrastive:
            # apply postprocess first, then optionally convert anchor to single-channel.
            if args.anchor_single_channel and np.asarray(speech_mix).ndim > 1:
                speech_mix = np.asarray(speech_mix)[:, 0]

            emb = _extract_embedding(
                model=model,
                speech_mix=speech_mix,
                sample_rate=bundle["sample_rate"],
                device=args.device,
            )
            embeddings.append(emb)

            meta_rows.append(
                {
                    "index": str(len(meta_rows)),
                    "pair_idx": str(pair_idx),
                    "pair_uid": uid,
                    "rir_label": f"RIR-{rir_idx}",
                    "rir_uid": rir_uid,
                    "rir_path": rir_map[rir_uid],
                    "room_param_path": room_map[rir_uid],
                }
            )

    emb_arr = np.stack(embeddings, axis=0)
    n_samples = emb_arr.shape[0]
    if n_samples < 3:
        raise ValueError(f"Need at least 3 points for t-SNE, got {n_samples}")

    perplexity = min(args.perplexity, float(n_samples - 1))
    from sklearn.manifold import TSNE

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=args.learning_rate,
        random_state=args.random_seed,
        init="pca",
    )
    tsne_xy = tsne.fit_transform(emb_arr)

    args.output_dir.mkdir(parents=True, exist_ok=False)

    np.savez(
        args.output_dir / "embeddings.npz",
        embeddings=emb_arr,
        tsne=tsne_xy,
    )
    _save_meta_csv(args.output_dir / "meta.csv", meta_rows)
    _plot_tsne(args.output_dir / "tsne.png", tsne_xy, meta_rows)

    used = {
        "exp_dir": str(args.exp_dir),
        "checkpoint": args.checkpoint,
        "data_dir": str(args.data_dir),
        "pair_uids": pair_uids,
        "pair_selection": args.pair_selection,
        "pair_seed": args.pair_seed,
        "fixed_rir_uids": fixed_rir_uids,
        "num_pairs": args.num_pairs,
        "fixed_rir_count": args.fixed_rir_count,
        "sample_rate": args.sample_rate,
        "mix_type": args.mix_type,
        "ref_condition": args.ref_condition,
        "anchor_single_channel": bool(args.anchor_single_channel),
        "perplexity_input": args.perplexity,
        "perplexity_effective": perplexity,
        "learning_rate": args.learning_rate,
        "random_seed": args.random_seed,
        "device": args.device,
        "n_points": int(n_samples),
        "embedding_dim": int(emb_arr.shape[1]),
    }
    with (args.output_dir / "config_used.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(used, f, sort_keys=False)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(used, f, indent=2)

    print(f"Saved outputs to: {args.output_dir}")
    print(f"Points: {n_samples}, embedding_dim={emb_arr.shape[1]}, perplexity={perplexity}")


if __name__ == "__main__":
    main()

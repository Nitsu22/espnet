#!/usr/bin/env python3
"""t-SNE visualization with fixed+random RIR and random mixture selection.

The script reads fixed RIR paths from a YAML file, keeps those RIRs, and fills
the remaining slots with randomly selected RIRs from rir_npz.scp.
Mixtures are always selected randomly from npz.scp.
Default behavior: random 100 mixtures, assigned in blocks of 10 mixtures
to RIR-1 .. RIR-10.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
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


def _canonical_path(path: Path) -> str:
    return str(path.expanduser().resolve())


def _load_path_config(path_cfg: Path) -> Dict:
    suffix = path_cfg.suffix.lower()
    text = path_cfg.read_text(encoding="utf-8")
    if suffix == ".json":
        cfg = json.loads(text)
    else:
        cfg = yaml.safe_load(text)
    if not isinstance(cfg, dict):
        raise ValueError(f"{path_cfg} must contain a mapping object")
    return cfg


def _load_fixed_rir_paths(path_cfg: Path) -> List[Tuple[str, Path]]:
    cfg = _load_path_config(path_cfg)
    if not isinstance(cfg, dict):
        raise ValueError(f"{path_cfg} must contain a mapping")
    entries = cfg.get("fixed_rirs")
    if not isinstance(entries, list) or len(entries) == 0:
        raise ValueError(f"{path_cfg} must contain non-empty 'fixed_rirs' list")

    fixed: List[Tuple[str, Path]] = []
    for i, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            raise ValueError(f"fixed_rirs[{i}] must be a mapping")
        raw_path = entry.get("path")
        if not isinstance(raw_path, str) or raw_path.strip() == "":
            raise ValueError(f"fixed_rirs[{i}].path is required")
        label = entry.get("label")
        if not isinstance(label, str) or label.strip() == "":
            label = f"fixed_{i}"

        p = Path(raw_path)
        if not p.is_absolute():
            p = path_cfg.parent / p
        p = p.expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"fixed_rirs[{i}] path does not exist: {p}")
        fixed.append((label, p))
    return fixed


def _select_random_pair_uids(
    all_uids: List[str],
    num_pairs: int,
    pair_seed: Optional[int],
) -> List[str]:
    if num_pairs > len(all_uids):
        raise ValueError(
            f"Requested {num_pairs} pairs, but only {len(all_uids)} available"
        )

    rng = np.random.default_rng(pair_seed)
    indices = rng.choice(len(all_uids), size=num_pairs, replace=False)
    return [all_uids[int(i)] for i in indices]


def _select_rir_uids_with_fixed_paths(
    rir_order: List[str],
    rir_map: Dict[str, str],
    total_count: int,
    fixed_rir_paths: List[Tuple[str, Path]],
    random_seed: Optional[int],
) -> Tuple[List[str], Dict[str, str], List[str]]:
    if total_count <= 0:
        raise ValueError("--fixed-rir-count must be > 0")

    path_to_uid = {
        _canonical_path(Path(path)): uid for uid, path in rir_map.items()
    }

    fixed_uid_to_label: Dict[str, str] = {}
    fixed_uids: List[str] = []
    for label, fixed_path in fixed_rir_paths:
        uid = path_to_uid.get(_canonical_path(fixed_path))
        if uid is None:
            raise ValueError(
                f"Fixed RIR path not found in rir_npz.scp: {fixed_path}"
            )
        if uid in fixed_uid_to_label:
            continue
        fixed_uid_to_label[uid] = label
        fixed_uids.append(uid)

    if len(fixed_uids) > total_count:
        raise ValueError(
            f"fixed-rir-count ({total_count}) is smaller than "
            f"number of fixed RIRs in YAML ({len(fixed_uids)})"
        )

    candidates = [uid for uid in rir_order if uid not in fixed_uid_to_label]
    need_random = total_count - len(fixed_uids)
    if need_random > len(candidates):
        raise ValueError(
            f"Requested {need_random} random RIRs, but only {len(candidates)} available "
            "after excluding fixed RIRs"
        )

    random_uids: List[str] = []
    if need_random > 0:
        rng = np.random.default_rng(random_seed)
        indices = rng.choice(len(candidates), size=need_random, replace=False)
        random_uids = [candidates[int(i)] for i in indices]

    selected = fixed_uids + random_uids
    return selected, fixed_uid_to_label, random_uids


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

    plt.title(f"t-SNE of Mixture Embeddings (Fixed {len(unique_labels)} RIR)")
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    script_dir = Path(__file__).parent
    parser.add_argument("--exp-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=str, default="valid.loss.best.pth")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--rir-path-config",
        type=Path,
        default=script_dir / "path.yaml",
        help="YAML/JSON file that lists fixed RIR paths (fixed_rirs).",
    )
    parser.add_argument("--num-pairs", type=int, default=100)
    parser.add_argument(
        "--pair-selection",
        type=str,
        default="random",
        choices=["random", "head"],
        help="Kept for compatibility. Ignored in this script (mixtures are always random).",
    )
    parser.add_argument(
        "--pair-seed",
        type=int,
        default=None,
        help="Seed for pair-selection=random. If omitted, pairs change each run.",
    )
    parser.add_argument(
        "--fixed-rir-count",
        type=int,
        default=10,
        help="Total number of RIRs to use per mixture. Includes fixed RIRs from YAML plus random fill.",
    )
    parser.add_argument(
        "--rir-selection-seed",
        type=int,
        default=0,
        help="Seed for selecting random (non-fixed) RIRs.",
    )
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
    parser.add_argument(
        "--png-output-dir",
        type=Path,
        default=script_dir / "output",
        help="Directory to save the exported PNG.",
    )
    parser.add_argument(
        "--png-name",
        type=str,
        default=None,
        help="Optional PNG filename in --png-output-dir. Default: output.png",
    )
    args = parser.parse_args()

    if args.num_pairs <= 0:
        raise ValueError("--num-pairs must be > 0")
    if args.fixed_rir_count <= 0:
        raise ValueError("--fixed-rir-count must be > 0")
    if args.num_pairs % args.fixed_rir_count != 0:
        raise ValueError(
            "--num-pairs must be divisible by --fixed-rir-count "
            "(e.g., 100 mixtures and 10 RIRs)"
        )

    checkpoint_path = _require(args.exp_dir / args.checkpoint)
    config_path = _require(args.exp_dir / "config.yaml")
    rir_path_config = _require(args.rir_path_config)

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
    fixed_rir_paths = _load_fixed_rir_paths(rir_path_config)

    pair_uids = _select_random_pair_uids(
        all_uids=npz_order,
        num_pairs=args.num_pairs,
        pair_seed=args.pair_seed,
    )

    fixed_rir_uids, fixed_uid_to_label, random_rir_uids = _select_rir_uids_with_fixed_paths(
        rir_order=rir_order,
        rir_map=rir_map,
        total_count=args.fixed_rir_count,
        fixed_rir_paths=fixed_rir_paths,
        random_seed=args.rir_selection_seed,
    )
    mixtures_per_rir = args.num_pairs // args.fixed_rir_count

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
        rir_idx = ((pair_idx - 1) // mixtures_per_rir) + 1
        rir_uid = fixed_rir_uids[rir_idx - 1]
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
                "pair_source": "random",
                "pair_config_label": "",
                "rir_label": f"RIR-{rir_idx}",
                "rir_source": "fixed" if rir_uid in fixed_uid_to_label else "random",
                "rir_config_label": fixed_uid_to_label.get(rir_uid, ""),
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
    local_png = args.output_dir / "tsne.png"
    _plot_tsne(local_png, tsne_xy, meta_rows)
    args.png_output_dir.mkdir(parents=True, exist_ok=True)
    if args.png_name:
        png_name = args.png_name
    else:
        png_name = "output.png"
    exported_png = args.png_output_dir / png_name
    shutil.copy2(local_png, exported_png)

    used = {
        "exp_dir": str(args.exp_dir),
        "checkpoint": args.checkpoint,
        "data_dir": str(args.data_dir),
        "rir_path_config": str(rir_path_config),
        "pair_uids": pair_uids,
        "pair_selection": "random",
        "pair_seed": args.pair_seed,
        "fixed_mixtures_from_config": [],
        "random_pair_uids": pair_uids,
        "fixed_rir_uids": fixed_rir_uids,
        "fixed_rir_from_yaml": [
            {"label": label, "path": str(path)} for label, path in fixed_rir_paths
        ],
        "random_rir_uids": random_rir_uids,
        "num_pairs": args.num_pairs,
        "fixed_rir_count": args.fixed_rir_count,
        "mixtures_per_rir": mixtures_per_rir,
        "assignment_mode": "block",
        "rir_selection_seed": args.rir_selection_seed,
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
        "png_output_dir": str(args.png_output_dir),
        "png_path": str(exported_png),
    }
    with (args.output_dir / "config_used.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(used, f, sort_keys=False)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(used, f, indent=2)

    print(f"Saved outputs to: {args.output_dir}")
    print(f"Exported PNG to: {exported_png}")
    print(f"Points: {n_samples}, embedding_dim={emb_arr.shape[1]}, perplexity={perplexity}")


if __name__ == "__main__":
    main()

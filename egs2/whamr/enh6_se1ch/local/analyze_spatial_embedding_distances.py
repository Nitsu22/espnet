#!/usr/bin/env python3

import argparse
import csv
import json
import os
from pathlib import Path
from statistics import mean, median
from typing import Dict, Iterable, List

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")

import numpy as np
import torch
import yaml

from espnet2.bin.enh_inference_se_condition_div_ablation import (
    build_model_from_args_and_file,
    get_train_config,
    recursive_dict_update,
)
from espnet2.tasks.enh_se_condition_div import EnhancementTask
from espnet2.torch_utils.device_funcs import to_device
from espnet2.train.preprocessor_npz_ablation import NpzSpatialAblationPreprocessor


def infer_mix_type(npz_data_dir: Path) -> str:
    name = npz_data_dir.name
    if "mix_clean" in name:
        return "clean"
    if "mix_single" in name:
        return "single"
    return "both"


def read_keys(scp_path: Path) -> List[str]:
    keys = []
    for line in scp_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        keys.append(line.split(maxsplit=1)[0])
    return keys


class SpatialEmbeddingExtractor:
    def __init__(
        self,
        train_config: str,
        model_file: str,
        inference_config: str,
        device: str,
        dtype: str,
    ):
        train_config_path = get_train_config(train_config, model_file=model_file).resolve()
        with train_config_path.open("r", encoding="utf-8") as f:
            train_args = yaml.safe_load(f)

        if inference_config is not None:
            with Path(inference_config).expanduser().resolve().open(
                "r", encoding="utf-8"
            ) as f:
                infer_args = yaml.safe_load(f)
            recursive_dict_update(train_args, infer_args, verbose=False)

        train_args["config"] = str(train_config_path)
        enh_train_args = argparse.Namespace(**train_args)
        enh_model = build_model_from_args_and_file(
            EnhancementTask, enh_train_args, model_file, device
        )
        enh_model.to(dtype=getattr(torch, dtype)).eval()

        self.device = device
        self.dtype = dtype
        self.enh_train_args = enh_train_args
        self.enh_model = enh_model

    def compute_spatial_embedding(
        self, speech_mix_mc: torch.Tensor, speech_lengths: torch.Tensor
    ) -> torch.Tensor:
        if (
            not hasattr(self.enh_model, "spatial_encoder")
            or self.enh_model.spatial_encoder is None
            or not hasattr(self.enh_model, "spatial_encoder_encoder")
            or self.enh_model.spatial_encoder_encoder is None
        ):
            raise ValueError(
                "enh_model.spatial_encoder or enh_model.spatial_encoder_encoder is not configured."
            )

        encoder_input = speech_mix_mc
        num_channels_arg = None
        if speech_mix_mc.dim() == 3 and speech_mix_mc.shape[2] == 1:
            encoder_input = speech_mix_mc.squeeze(-1)
            num_channels_arg = 1
        elif speech_mix_mc.dim() == 2:
            num_channels_arg = 1

        try:
            feature_mix_mc, flens_mc = self.enh_model.spatial_encoder_encoder(
                encoder_input, speech_lengths, fs=None
            )
        except TypeError:
            feature_mix_mc, flens_mc = self.enh_model.spatial_encoder_encoder(
                encoder_input, speech_lengths
            )

        pooling = getattr(self.enh_model, "spatial_encoder_pooling", True)
        try:
            return self.enh_model.spatial_encoder(
                feature_mix_mc,
                flens_mc,
                num_channels=num_channels_arg,
                pooling=pooling,
            )
        except TypeError:
            return self.enh_model.spatial_encoder(
                feature_mix_mc,
                flens_mc,
                num_channels=num_channels_arg,
            )


def compute_embedding(
    extractor: SpatialEmbeddingExtractor,
    speech_mix: np.ndarray,
    dtype: str,
    device: str,
) -> torch.Tensor:
    mix = torch.as_tensor(speech_mix, dtype=getattr(torch, dtype)).unsqueeze(0)
    lengths = torch.as_tensor([mix.size(1)], dtype=torch.long)
    mix = to_device(mix, device=device)
    lengths = to_device(lengths, device=device)
    embedding = extractor.compute_spatial_embedding(mix, lengths).detach().cpu()
    if embedding.dim() == 3:
        embedding = embedding.mean(dim=1)
    if embedding.dim() != 2 or embedding.size(0) != 1:
        raise RuntimeError(f"Unexpected embedding shape: {tuple(embedding.shape)}")
    return embedding[0]


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())


def l2_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.norm(a - b, p=2).item())


def summarize(values: Iterable[float]) -> Dict[str, float]:
    values = list(values)
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
        }
    values_np = np.asarray(values, dtype=np.float64)
    return {
        "count": int(values_np.size),
        "mean": float(mean(values)),
        "median": float(median(values)),
        "std": float(values_np.std(ddof=0)),
        "min": float(values_np.min()),
        "max": float(values_np.max()),
    }


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze Spatial Encoder embedding distances for "
            "oracle / same-audio+different-RIR / different-audio+same-RIR."
        )
    )
    parser.add_argument("--train_config", type=str, default=None)
    parser.add_argument("--model_file", type=str, required=True)
    parser.add_argument("--inference_config", type=str, default=None)
    parser.add_argument("--npz_data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--pool_npz_scp", type=str, default=None)
    parser.add_argument("--key_file", type=str, default=None)
    parser.add_argument("--max_utts", type=int, default=0)
    parser.add_argument("--skip_utts", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--fs", type=int, default=8000)
    parser.add_argument("--ngpu", type=int, default=0)
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float16", "float32", "float64"],
    )
    return parser


def main() -> None:
    args = get_parser().parse_args()

    npz_data_dir = Path(args.npz_data_dir).expanduser().resolve()
    npz_scp = npz_data_dir / "npz.scp"
    if not npz_scp.exists():
        raise FileNotFoundError(f"Missing npz.scp: {npz_scp}")

    key_file = Path(args.key_file).expanduser().resolve() if args.key_file else npz_scp
    keys = read_keys(key_file)
    if args.skip_utts:
        keys = keys[args.skip_utts :]
    if args.max_utts and args.max_utts > 0:
        keys = keys[: args.max_utts]
    if not keys:
        raise RuntimeError("No utterances selected")

    device = "cuda" if args.ngpu >= 1 else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("ngpu >= 1 but CUDA is not available")

    extractor = SpatialEmbeddingExtractor(
        train_config=args.train_config,
        model_file=args.model_file,
        inference_config=args.inference_config,
        device=device,
        dtype=args.dtype,
    )

    ablator = NpzSpatialAblationPreprocessor(
        npz_scp=str(npz_scp),
        pool_npz_scp=args.pool_npz_scp,
        sample_rate=args.fs,
        seed=args.seed,
        epoch=args.epoch,
        mix_type=infer_mix_type(npz_data_dir),
    )

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "distances.csv"
    summary_path = output_dir / "summary.json"

    rows = []
    for index, uid in enumerate(keys, start=1):
        partner_uid = ablator._select_pool_uid(uid)
        anchor = ablator._load_bundle_from_map(uid, ablator.npz_map)
        partner = ablator._load_bundle_from_map(partner_uid, ablator.pool_npz_map)

        oracle_mix = ablator._synthesize_from_bundle(
            uid, anchor, anchor["rir_path"], anchor["room_param_path"]
        )
        swap_rir_mix = ablator._synthesize_from_bundle(
            uid, anchor, partner["rir_path"], partner["room_param_path"]
        )
        swap_audio_mix = ablator._synthesize_from_bundle(
            partner_uid, partner, anchor["rir_path"], anchor["room_param_path"]
        )

        emb_oracle = compute_embedding(extractor, oracle_mix, args.dtype, device)
        emb_swap_rir = compute_embedding(extractor, swap_rir_mix, args.dtype, device)
        emb_swap_audio = compute_embedding(
            extractor, swap_audio_mix, args.dtype, device
        )

        row = {
            "index": index,
            "uid": uid,
            "partner_uid": partner_uid,
            "anchor_rir_path": anchor["rir_path"],
            "partner_rir_path": partner["rir_path"],
            "oracle_norm": float(torch.norm(emb_oracle, p=2).item()),
            "swap_rir_norm": float(torch.norm(emb_swap_rir, p=2).item()),
            "swap_audio_norm": float(torch.norm(emb_swap_audio, p=2).item()),
            "cosine_similarity_oracle_swap_rir": cosine_similarity(
                emb_oracle, emb_swap_rir
            ),
            "cosine_similarity_oracle_swap_audio": cosine_similarity(
                emb_oracle, emb_swap_audio
            ),
            "cosine_similarity_swap_rir_swap_audio": cosine_similarity(
                emb_swap_rir, emb_swap_audio
            ),
            "cosine_distance_oracle_swap_rir": 1.0
            - cosine_similarity(emb_oracle, emb_swap_rir),
            "cosine_distance_oracle_swap_audio": 1.0
            - cosine_similarity(emb_oracle, emb_swap_audio),
            "cosine_distance_swap_rir_swap_audio": 1.0
            - cosine_similarity(emb_swap_rir, emb_swap_audio),
            "l2_distance_oracle_swap_rir": l2_distance(emb_oracle, emb_swap_rir),
            "l2_distance_oracle_swap_audio": l2_distance(emb_oracle, emb_swap_audio),
            "l2_distance_swap_rir_swap_audio": l2_distance(
                emb_swap_rir, emb_swap_audio
            ),
        }
        rows.append(row)
        print(
            "[{}/{}] {} partner={} cos(o,sr)={:.6f} cos(o,sa)={:.6f}".format(
                index,
                len(keys),
                uid,
                partner_uid,
                row["cosine_distance_oracle_swap_rir"],
                row["cosine_distance_oracle_swap_audio"],
            ),
            flush=True,
        )

    fieldnames = list(rows[0].keys())
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "num_utts": len(rows),
        "train_config": str(Path(args.train_config).expanduser()),
        "model_file": str(Path(args.model_file).expanduser()),
        "npz_data_dir": str(npz_data_dir),
        "metrics": {
            "cosine_distance_oracle_swap_rir": summarize(
                row["cosine_distance_oracle_swap_rir"] for row in rows
            ),
            "cosine_distance_oracle_swap_audio": summarize(
                row["cosine_distance_oracle_swap_audio"] for row in rows
            ),
            "cosine_distance_swap_rir_swap_audio": summarize(
                row["cosine_distance_swap_rir_swap_audio"] for row in rows
            ),
            "l2_distance_oracle_swap_rir": summarize(
                row["l2_distance_oracle_swap_rir"] for row in rows
            ),
            "l2_distance_oracle_swap_audio": summarize(
                row["l2_distance_oracle_swap_audio"] for row in rows
            ),
            "l2_distance_swap_rir_swap_audio": summarize(
                row["l2_distance_swap_rir_swap_audio"] for row in rows
            ),
        },
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    print("", flush=True)
    print(f"Wrote CSV: {csv_path}", flush=True)
    print(f"Wrote summary: {summary_path}", flush=True)
    for metric_name, metric_summary in summary["metrics"].items():
        print(
            "{} mean={:.6f} median={:.6f} std={:.6f}".format(
                metric_name,
                metric_summary["mean"],
                metric_summary["median"],
                metric_summary["std"],
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()

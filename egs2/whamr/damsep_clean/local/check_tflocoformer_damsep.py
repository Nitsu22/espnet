"""Validate noisy WHAMR tuples, reused shapes, and real FP32 batch-four CUDA."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml
from check_linked_dump import read_scp

from espnet2.tasks.damsep import DAMSEPTask

FIELDS = (
    "speech_mix",
    "speech_ref1",
    "speech_ref2",
    "speech_reverb1",
    "speech_reverb2",
)


def load_audio(path):
    audio, rate = sf.read(path, dtype="float64", always_2d=True)
    if rate != 8000 or audio.shape[1] != 2 or not np.isfinite(audio).all():
        raise ValueError(f"Expected finite stereo 8 kHz waveform: {path}")
    return audio


def check_data(statsdir, expected_counts=None):
    report = dict(condition="whamr", sample_rate=8000, channel=0, splits={}, hashes={})
    maps = {}
    expected_counts = expected_counts or dict(tr=20000, cv=5000, tt=3000)
    for split, count in expected_counts.items():
        noisy = Path("dump_whamr/raw") / f"{split}_mix_both_reverb_min_8k"
        nf = Path("dump_clean/raw") / f"{split}_mix_clean_reverb_min_8k"
        rev = Path("dump_reverb/raw") / f"{split}_mix_both_reverb_min_8k"
        paths = [
            noisy / "wav.scp",
            noisy / "spk1.scp",
            noisy / "spk2.scp",
            rev / "spk1_reverb.scp",
            rev / "spk2_reverb.scp",
        ]
        current = [read_scp(path) for path in paths]
        nf_paths = [nf / name for name in ("wav.scp", "spk1.scp", "spk2.scp")]
        nf_maps = [read_scp(path) for path in nf_paths]
        ids = sorted(current[0])
        noise_path = noisy / "noise1.scp"
        noise = read_scp(noise_path) if noise_path.exists() else None
        if len(ids) != count or any(set(m) != set(ids) for m in current + nf_maps):
            raise ValueError(f"Unaligned/incomplete WHAMR sources: {split}")
        if split != "tt" and (noise is None or set(noise) != set(ids)):
            raise ValueError(f"Missing or incomplete WHAMR noise: {split}")
        for path in paths + nf_paths + ([noise_path] if noise is not None else []):
            report["hashes"][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        shapes = []
        if split != "tt":
            phase = "train" if split == "tr" else "valid"
            for field in FIELDS:
                path = statsdir / phase / f"{field}_shape"
                shape = dict(
                    line.split(maxsplit=1) for line in path.read_text().splitlines()
                )
                if set(shape) != set(ids):
                    raise ValueError(
                        f"Stats IDs differ from actual noisy tuple: {path}"
                    )
                shapes.append(shape)
            if any(shape != shapes[0] for shape in shapes[1:]):
                raise ValueError(f"Unaligned existing five-series statistics: {split}")
        sampled = ids[:: max(1, len(ids) // 8)][:8]
        if split == "cv":
            longest = max(ids, key=lambda uid: int(shapes[0][uid].split(",")[0]))
            sampled = list(dict.fromkeys(sampled + [longest]))
            valid_shapes = shapes[0]
        errors = dict(clean_teacher=0.0, nf_reverb_sum=0.0, noisy_sum=0.0)
        for uid in sampled:
            audio = [load_audio(m[uid]) for m in current]
            nf_audio = [load_audio(m[uid]) for m in nf_maps]
            noise_audio = load_audio(noise[uid]) if noise is not None else None
            all_audio = (
                audio + nf_audio + ([noise_audio] if noise_audio is not None else [])
            )
            if any(x.shape != audio[0].shape for x in all_audio):
                raise ValueError(f"Cross-source lengths/channels differ: {uid}")
            if shapes and len(audio[0]) != int(shapes[0][uid].split(",")[0]):
                raise ValueError(f"Noisy waveform disagrees with reused shape: {uid}")
            errors["clean_teacher"] = max(
                errors["clean_teacher"],
                *(float(np.max(np.abs(audio[i] - nf_audio[i]))) for i in (1, 2)),
            )
            errors["nf_reverb_sum"] = max(
                errors["nf_reverb_sum"],
                float(np.max(np.abs(nf_audio[0] - audio[3] - audio[4]))),
            )
            if noise_audio is not None:
                errors["noisy_sum"] = max(
                    errors["noisy_sum"],
                    float(np.max(np.abs(audio[0] - audio[3] - audio[4] - noise_audio))),
                )
            if (
                errors["clean_teacher"] > 4e-5
                or errors["nf_reverb_sum"] > 4e-5
                or errors["noisy_sum"] > 1e-4
            ):
                raise ValueError(f"WHAMR source semantics failed: {uid}: {errors}")
        report["splits"][split] = dict(
            count=count,
            sampled_ids=sampled,
            noise_sum_checked=noise is not None,
            max_errors=errors,
        )
        maps[split] = current
    return report, maps, valid_shapes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant", choices=("clean_reverb", "three_loss"), required=True
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--data-only", action="store_true")
    parser.add_argument(
        "--statsdir", type=Path, default=Path("exp/tflocoformer_damsep_stats_8k")
    )
    args = parser.parse_args()
    report, maps, shapes = check_data(args.statsdir)
    config = Path(f"conf/tuning/train_tflocoformer_damsep_whamr_8k_{args.variant}.yaml")
    if not args.data_only:
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Exactly one allocated CUDA GPU is required")
        options = DAMSEPTask.get_parser().parse_args(
            [
                "--config",
                str(config),
                "--output_dir",
                str(args.report.parent),
                "--ngpu",
                "1",
            ]
        )
        np.random.seed(options.seed)
        torch.manual_seed(options.seed)
        model = DAMSEPTask.build_model(options).cuda()
        # Includes the initialized but frozen CTF branch: A/B must match exactly.
        digest = hashlib.sha256()
        for name, tensor in model.state_dict().items():
            digest.update(name.encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        report["initial_state_sha256"] = digest.hexdigest()

        def batch(split, ids, training):
            processor = DAMSEPTask.build_preprocess_fn(options, training)
            items = [
                (
                    uid,
                    processor(
                        uid,
                        {
                            field: load_audio(m[uid])
                            for field, m in zip(FIELDS, maps[split])
                        },
                    ),
                )
                for uid in ids
            ]
            _, values = DAMSEPTask.build_collate_fn(options, training)(items)
            return {
                k: v.cuda().float() if v.is_floating_point() else v.cuda()
                for k, v in values.items()
            }

        ids = []
        for uid, path in maps["tr"][0].items():
            if sf.info(path).frames >= 32000:
                ids.append(uid)
            if len(ids) == 4:
                break
        if len(ids) != 4:
            raise ValueError("Four full training segments are unavailable")
        values = batch("tr", ids, True)
        if values["speech_mix"].shape != (4, 32000):
            raise ValueError("Preprocessor did not produce physical batch4 x 4 seconds")
        config_dict = yaml.safe_load(config.read_text())
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            **config_dict["optim_conf"],
        )
        torch.cuda.reset_peak_memory_stats()
        model.train()
        loss, stats, _ = model(**values)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite batch-four loss")
        loss.backward()
        bad = [
            name
            for name, p in model.named_parameters()
            if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all())
        ]
        if bad or any(
            p.grad is not None for p in model.parameters() if not p.requires_grad
        ):
            raise RuntimeError(f"Invalid active/frozen gradients: {bad}")
        torch.nn.utils.clip_grad_norm_(model.parameters(), options.grad_clip)
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise RuntimeError("Nonfinite parameters after AdamW step")
        report["cuda_train"] = dict(
            batch_size=4,
            samples=32000,
            loss=float(loss.detach()),
            stats={k: float(v) for k, v in stats.items()},
            peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
            finite_gradients=True,
            optimizer_step=True,
        )
        del values, loss
        optimizer.zero_grad(set_to_none=True)
        model.eval()
        longest = max(shapes, key=lambda uid: int(shapes[uid].split(",")[0]))
        values = batch("cv", [longest], False)
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            loss, stats, _ = model(**values)
            output = model.separate(values["speech_mix"])
        if not torch.isfinite(loss) or any(
            not torch.isfinite(v).all() for v in output.values()
        ):
            raise RuntimeError("Nonfinite longest validation/inference output")
        report["cuda_valid"] = dict(
            uid=longest,
            samples=values["speech_mix"].shape[1],
            loss=float(loss),
            stats={k: float(v) for k, v in stats.items()},
            peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
        )
        report.update(
            device=torch.cuda.get_device_name(0),
            trainable_parameters=sum(
                p.numel() for p in model.parameters() if p.requires_grad
            ),
            frozen_parameters=sum(
                p.numel() for p in model.parameters() if not p.requires_grad
            ),
            ctf_enabled=model.ctf_enabled,
        )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

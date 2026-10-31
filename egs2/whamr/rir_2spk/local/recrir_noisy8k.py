"""Verify noisy Rec-RIR data/CUDA training, finalize, and score native 8-kHz LS."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import soundfile as sf
import yaml

FIELDS = ("speech_mix", "speech_direct1", "speech_direct2", "speech_reverb1", "speech_reverb2")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    Path(path).write_text(json.dumps(data, indent=2) + "\n")


def rows(path):
    pairs = [line.split(maxsplit=1) for line in Path(path).read_text().splitlines() if line.strip()]
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("Duplicate IDs: " + str(path))
    return result


def load_wave(path):
    wave, rate = sf.read(path, dtype="float32", always_2d=True)
    if rate != 8000 or not np.isfinite(wave).all():
        raise ValueError("Expected finite native 8-kHz audio: " + path)
    return wave


def prepare(args):
    manifest = json.loads((args.dump / "preparation.json").read_text())
    spec = manifest["spec"]
    if not (spec["noise"] is True and spec["sample_rate"] == "8k" and spec["num_spk"] == 2):
        raise ValueError("Require the existing noisy two-speaker 8-kHz dump")
    if manifest["counts"] != {"tr": 20000, "cv": 5000, "tt": 3000}:
        raise ValueError("Incomplete noisy corpus")
    for name, expected in manifest["output_hashes"].items():
        if digest(args.dump / name) != expected:
            raise ValueError("Prepared index changed: " + name)
    reference = args.nf_dump.parent / "dump_ls_direct_reverb/raw/tt_rir_2spk_nf_min_8k"
    measurement = json.loads((reference / "preparation.json").read_text())
    if not measurement["full_dataset"] or measurement["config"]["sample_rate"] != 8000 or measurement["config"]["rir_length"] != 16000:
        raise ValueError("Require the completed native-8-kHz LS test reference")
    test = args.dump / "raw/tt_rir_2spk_noisy_min_8k"
    if set(rows(test / "wav.scp")) != set(rows(reference / "reference.scp")):
        raise ValueError("Noisy test and LS reference IDs differ")
    for key in ("rir_direct1.scp", "rir_direct2.scp", "rir_ref1.scp", "rir_ref2.scp"):
        if digest(test / key) != measurement["source_index_sha256"][key]:
            raise ValueError("Noisy test and LS reference RIR source pairing differs")
    report = dict(manifest_sha256=digest(args.dump / "preparation.json"), counts={},
                  config_sha256=digest(args.config), teacher_indexes_match_nf=True,
                  sampled_noise_components_match=True, input_only_fold_length=200000)
    report["ls_reference_manifest_sha256"] = digest(reference / "preparation.json")
    for split, expected_count in manifest["counts"].items():
        folder = args.dump / "raw" / f"{split}_rir_2spk_noisy_min_8k"
        nf = args.nf_dump / "raw" / f"{split}_rir_2spk_nf_min_8k"
        maps = {name: rows(folder / f"{name}.scp") for name in FIELDS}
        noise = rows(folder / "noise_ref.scp")
        lengths = rows(folder / "utt2num_samples")
        ids = sorted(maps["speech_mix"])
        if len(ids) != expected_count or any(set(m) != set(ids) for m in [*maps.values(), noise, lengths]):
            raise ValueError("Mixture, teachers, noise and lengths have different IDs")
        if digest(folder / "wav.scp") != digest(folder / "speech_mix.scp"):
            raise ValueError("Training must read the actual noisy mixture")
        for name in (*FIELDS[1:], "rir_direct1", "rir_direct2", "rir_ref1", "rir_ref2"):
            if digest(folder / f"{name}.scp") != digest(nf / f"{name}.scp"):
                raise ValueError("Noise-free and noisy teacher/RIR indexes differ: " + name)

        def inspect(uid):
            count = int(lengths[uid])
            if not 128 < count < 200000:
                raise ValueError("Invalid native length or fold threshold: " + uid)
            for path in [*(maps[name][uid] for name in FIELDS), noise[uid]]:
                info = sf.info(path)
                if info.samplerate != 8000 or info.frames != count or info.channels not in (1, 2):
                    raise ValueError("Wrong rate, channels or unaligned teacher: " + uid)
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(inspect, ids))
        for uid in ids[::max(1, len(ids) // 32)]:
            mix = load_wave(maps["speech_mix"][uid])[:, 0]
            clean = sum(load_wave(maps[name][uid])[:, 0] for name in ("speech_reverb1", "speech_reverb2"))
            added = load_wave(noise[uid])[:, 0]
            if not np.any(added) or np.max(np.abs(mix - clean - added)) > 2e-6:
                raise ValueError("Noisy mixture does not equal both sources plus noise: " + uid)
        if split in ("tr", "cv"):
            dest = args.stats / ("train" if split == "tr" else "valid")
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "speech_mix_shape").write_text("".join(f"{uid} {lengths[uid]}\n" for uid in ids))
        report["counts"][split] = len(ids)
        print(f"Verified noisy {split}: {len(ids)} utterances", flush=True)
    save(args.run / "data_check.json", report)


def probe(args):
    import torch
    from espnet2.tasks.rir import RIRTask
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Use exactly one allocated CUDA GPU")
    cfg = RIRTask.get_parser().parse_args(["--config", str(args.config), "--output_dir", str(args.run / "probe")])
    if cfg.init_param or cfg.rir_model_type != "rec_rir_pit":
        raise ValueError("Use the original scratch two-speaker Rec-RIR")
    torch.set_num_threads(1)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    model = RIRTask.build_model(cfg).cuda().train()
    report = dict(config_sha256=digest(args.config), device=torch.cuda.get_device_name(0),
                  amp=cfg.use_amp, parameters=sum(p.numel() for p in model.parameters()))

    def batch(split, ids, train):
        folder = args.dump / "raw" / f"{split}_rir_2spk_noisy_min_8k"
        maps = {k: rows(folder / f"{k}.scp") for k in FIELDS}
        proc = RIRTask.build_preprocess_fn(cfg, train)
        examples = [(uid, proc(uid, {k: load_wave(maps[k][uid]) for k in FIELDS})) for uid in ids]
        _, data = RIRTask.build_collate_fn(cfg, train)(examples)
        return {k: (v.float() if v.is_floating_point() else v).cuda() for k, v in data.items()}

    folder = args.dump / "raw/tr_rir_2spk_noisy_min_8k"
    ids = [uid for uid, n in rows(folder / "utt2num_samples").items() if int(n) >= 32000][:4]
    train = batch("tr", ids, True)
    if train["speech_mix"].shape != (4, 32000):
        raise ValueError("Expected four real four-second crops")
    optimizer = torch.optim.AdamW(model.parameters(), **cfg.optim_conf)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp)
    before = next(model.parameters()).detach().clone()
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    successful = 0
    for attempt in range(10):
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=cfg.use_amp):
            loss, stats, _ = model(**train)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite original Rec-RIR loss")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        active = [p for p in model.parameters() if p.requires_grad]
        if any(p.grad is None for p in active):
            raise RuntimeError("Missing original Rec-RIR gradient")
        finite = all(torch.isfinite(p.grad).all() for p in active)
        if finite:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip, error_if_nonfinite=True)
        scaler.step(optimizer)
        scaler.update()
        successful += int(finite)
        if successful == 2:
            break
    if successful != 2 or torch.equal(before, next(model.parameters())):
        raise RuntimeError("No two finite AMP updates after scaler calibration")
    report.update(train_loss=float(loss), train_shape=list(train["speech_mix"].shape),
                  successful_updates=successful, scaler_attempts=attempt + 1, scaler_scale=scaler.get_scale(),
                  train_peak_gib=torch.cuda.max_memory_allocated() / 2**30,
                  verification_seconds=time.monotonic() - start)
    model.eval()
    with torch.no_grad():
        ctf = model.estimate_ctf(train["speech_mix"][:1])
        rir = model.estimate_rir(train["speech_mix"][0], rir_length=16000)
        if ctf.shape != (1, 2, 129, 60) or rir.shape != (2, 16000) or not torch.isfinite(rir).all():
            raise RuntimeError("Native CTF or ordinary-PIM two-RIR inference failed")
    model.zero_grad(set_to_none=True)
    del optimizer, loss, train
    torch.cuda.empty_cache()
    lengths = rows(args.dump / "raw/cv_rir_2spk_noisy_min_8k/utt2num_samples")
    uid = max(lengths, key=lambda key: int(lengths[key]))
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=cfg.use_amp):
        loss, _, _ = model(**batch("cv", [uid], False))
    if not torch.isfinite(loss):
        raise RuntimeError("Nonfinite longest full-utterance validation loss")
    report.update(validation_uid=uid, validation_samples=int(lengths[uid]), validation_loss=float(loss),
                  validation_peak_gib=torch.cuda.max_memory_allocated() / 2**30, status="passed")
    save(args.run / "gpu_check.json", report)
    print(json.dumps(report, indent=2), flush=True)


def finalize(args):
    import torch
    from espnet2.train.reporter import Reporter
    config = yaml.safe_load((args.model / "config.yaml").read_text())
    reporter = Reporter()
    reporter.load_state_dict(torch.load(args.model / "checkpoint.pth", map_location="cpu")["reporter"])
    epoch = reporter.get_epoch()
    finished = epoch >= config["max_epoch"] or reporter.check_early_stopping(
        config["patience"], *config["early_stopping_criterion"])
    best = args.model / "valid.loss.best.pth"
    if not finished or not best.is_file() or best.resolve().name != f'{reporter.get_best_epoch("valid", "loss", "min")}epoch.pth':
        raise RuntimeError("Training stopping criterion or best-model identity is invalid")
    if "The gradients at all steps are invalid" in (args.model / "train.log").read_text():
        raise RuntimeError("Trainer stopped with invalid gradients")
    save(args.run / "training_complete.json", dict(epoch=epoch, checkpoint=str(best.resolve()),
         checkpoint_sha256=digest(best), config_sha256=digest(args.model / "config.yaml")))


def evaluate(args):
    if not (args.run / "training_complete.json").is_file():
        raise RuntimeError("Training must finish normally before evaluation")
    output = args.model / "evaluation_noisy_test_best"
    output.mkdir(exist_ok=False)
    test = args.dump / "raw/tt_rir_2spk_noisy_min_8k"
    reference = args.nf_dump.parent / "dump_ls_direct_reverb/raw/tt_rir_2spk_nf_min_8k"
    if len(rows(test / "wav.scp")) != 3000 or set(rows(test / "wav.scp")) != set(rows(reference / "reference.scp")):
        raise ValueError("Native LS reference and noisy test IDs differ")
    save(output / "metadata.json", dict(inference="ordinary sweep PIM", sample_rate=8000,
         rir_length=16000, count=3000, input="full noisy WHAMR utterance, channel 0",
         output_subtype="FLOAT", checkpoint_sha256=digest(args.model / "valid.loss.best.pth")))
    subprocess.run([sys.executable, "-m", "espnet2.bin.rec_rir_pit_inference",
        "--train_config", str(args.model / "config.yaml"), "--model_file", str(args.model / "valid.loss.best.pth"),
        "--wav_scp", str(test / "wav.scp"), "--output_dir", str(output / "inference"),
        "--sample_rate", "8000", "--rir_length", "16000", "--output_subtype", "FLOAT", "--device", "cuda"], check=True)
    subprocess.run([sys.executable, "local/evaluate_ls_direct_reverb.py", "--reference_dir", str(reference),
        "--prediction", "RecRIR_noisy8k_best", str(output / "inference/rir.scp"), "--out_dir", str(output / "ls_score")], check=True)
    save(args.run / "evaluation_complete.json", dict(results=str(output / "ls_score/comparison.json")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "probe", "finalize", "evaluate"))
    for name in ("config", "dump", "nf-dump", "stats", "run", "model"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    {"prepare": prepare, "probe": probe, "finalize": finalize, "evaluate": evaluate}[args.phase](args)


if __name__ == "__main__":
    main()

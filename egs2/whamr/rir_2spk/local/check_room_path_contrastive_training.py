#!/usr/bin/env python3
"""One disposable CUDA update and ordinary RIR inference on two real pairs.

The input is a self-contained ``dump/raw/tr_*`` directory. Two distinct RIR
pair IDs with at least four seconds in both views are selected, each view is
independently cropped by the actual training preprocessor, and four waveforms
are passed through the real BiMamba. Only the JSON report is saved; the model
and optimizer update are discarded. No training checkpoint is produced.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True


def read_index(path):
    entries = {}
    with Path(path).open() as source:
        for line in source:
            key, value = line.rstrip().split(maxsplit=1)
            if key in entries:
                raise ValueError(f"Duplicate utterance ID in {path}: {key}")
            entries[key] = value
    return entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.config = args.config.resolve()
    args.data_dir = args.data_dir.resolve()
    args.output = args.output.resolve()
    if args.output.exists():
        raise ValueError("Use a fresh output report path")
    if args.data_dir.parent.name != "raw":
        raise ValueError("--data-dir must be a split directory inside dump/raw/")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    runtime = args.output.parent / "probe_runtime"
    # Keep imported scientific-library caches next to this task's report.
    os.environ["NUMBA_CACHE_DIR"] = str(runtime / "numba")
    os.environ["MPLCONFIGDIR"] = str(runtime / "matplotlib")
    os.environ["XDG_CACHE_HOME"] = str(runtime / "cache")
    os.environ["TORCH_EXTENSIONS_DIR"] = str(runtime / "torch_extensions")
    os.environ["CUDA_CACHE_PATH"] = str(runtime / "cuda")
    (runtime / "cuda").mkdir(parents=True, exist_ok=True)

    import numpy as np
    import soundfile as sf
    import torch
    from espnet2.tasks.rir import RIRTask

    if not torch.cuda.is_available():
        raise RuntimeError("This readiness check requires CUDA and real Mamba kernels")
    cfg = RIRTask.get_parser().parse_args([
        "--config", str(args.config), "--output_dir", str(args.output.parent)])
    if (cfg.rir_model_type != "pooled_bimamba_room_path_contrastive"
            or cfg.preprocessor != "rec_rir_room_path_contrastive"):
        raise ValueError("Use the paired room/path contrastive model and preprocessor")
    if cfg.init_param or cfg.init is not None or cfg.use_amp:
        raise ValueError("The scratch readiness check expects default initialization and no AMP")
    if cfg.model_conf["predictor_conf"].get("time_module", "bimamba") != "bimamba":
        raise ValueError("This check must exercise the actual BiMamba configuration")
    if cfg.sample_rate != 16000 or cfg.preprocessor_conf["speech_segment"] != 64000:
        raise ValueError("Expected 16 kHz training with four-second views")
    if cfg.preprocessor_conf["rir_length"] != 32000:
        raise ValueError("Expected two-second RIR targets")
    if cfg.batch_size != 2 or cfg.batch_type != "unsorted":
        raise ValueError("Expected two fixed paired items, hence four waveform inputs")
    if list(cfg.early_stopping_criterion) != ["valid", "loss_sweep", "min"]:
        raise ValueError("Early stopping must use valid.loss_sweep (min)")
    if list(cfg.val_scheduler_criterion) != ["valid", "loss_sweep"]:
        raise ValueError("Scheduler selection must use valid.loss_sweep")
    if [list(value) for value in cfg.best_model_criterion] != [["valid", "loss_sweep", "min"]]:
        raise ValueError("Best checkpoint selection must use valid.loss_sweep (min)")
    if (cfg.model_conf.get("lambda_room", 0.01) < 0
            or cfg.model_conf.get("lambda_path", 0.01) < 0
            or cfg.model_conf.get("validation_contrastive", False)):
        raise ValueError("Check nonnegative loss weights and primary-view-only validation")
    if cfg.optim != "adamw":
        raise ValueError("This readiness check expects the recipe's AdamW optimizer")

    names = ("speech_mix", "speech_mix_view2", "rir_direct1", "rir_direct2",
             "rir_ref1", "rir_ref2")
    indexes = {name: read_index(args.data_dir / (name + ".scp")) for name in names}
    lengths = {key: int(value) for key, value in
               read_index(args.data_dir / "utt2num_samples").items()}
    if any(set(mapping) != set(lengths) for mapping in indexes.values()):
        raise ValueError("Paired SCP streams and sample counts must have identical IDs")
    ids = [key for key in sorted(lengths) if lengths[key] >= 64000][:2]
    if len(ids) != 2:
        raise ValueError("Need two distinct pairs with at least 64000 samples per view")
    dump_root = args.data_dir.parent.parent
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    preprocess = RIRTask.build_preprocess_fn(cfg, train=True)
    examples = []
    for uid in ids:
        data = {}
        for name in names:
            path = Path(indexes[name][uid])
            if not path.is_absolute():
                raise ValueError("The committed dump indexes must use absolute WAV paths")
            resolved = path.resolve()
            try:
                resolved.relative_to(dump_root)
            except ValueError as error:
                raise ValueError(f"Non-self-contained data path: {path}") from error
            data[name], rate = sf.read(resolved)
            if rate != cfg.sample_rate or not np.isfinite(data[name]).all():
                raise ValueError(f"Invalid sample rate or nonfinite audio: {path}")
        if (len(data["speech_mix"]) != lengths[uid]
                or len(data["speech_mix_view2"]) != lengths[uid]):
            raise ValueError(f"The two full waveform lengths must match metadata: {uid}")
        prepared = preprocess(uid, data)
        # ESPnetDataset performs this conversion after preprocessing too.
        # CommonCollateFn itself preserves the supplied NumPy dtype.
        examples.append((uid, {name: value.astype(np.float32)
                               for name, value in prepared.items()}))
    _, batch = RIRTask.build_collate_fn(cfg, train=True)(examples)
    if (batch["speech_mix_lengths"].tolist() != [64000, 64000]
            or batch["speech_mix_view2_lengths"].tolist() != [64000, 64000]):
        raise AssertionError("The model must receive four full four-second views")
    batch = {name: value.cuda() for name, value in batch.items()}
    model = RIRTask.build_model(cfg).cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(), **cfg.optim_conf)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    loss, stats, weight = model(**batch)
    if int(weight) != 2 or not torch.isfinite(loss):
        raise AssertionError("Expected a finite training objective over two RIR pairs")
    for key in ("loss_sweep", "loss_room", "loss_path"):
        if key not in stats or not torch.isfinite(stats[key]):
            raise AssertionError(f"Missing or nonfinite loss: {key}")
    torch.testing.assert_close(loss, stats["loss_sweep"] + model.lambda_room * stats["loss_room"]
                               + model.lambda_path * stats["loss_path"])
    loss.backward()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and (parameter.grad is None
                                       or not torch.isfinite(parameter.grad).all()):
            raise AssertionError(f"Missing or nonfinite active parameter gradient: {name}")
    components = ("room_projection", "path_projection", "ctf_predictor.room_pooling",
                  "ctf_predictor.pooling", "ctf_predictor.room_path_fusion")
    gradients = {}
    for component in components:
        values = [parameter.grad.abs().sum() for name, parameter in model.named_parameters()
                  if parameter.requires_grad and name.startswith(component + ".")]
        total = sum(values)
        if not values or not torch.isfinite(total):
            raise AssertionError(f"Missing or nonfinite gradient in {component}")
        coefficient = {"room_projection": model.lambda_room,
                       "path_projection": model.lambda_path}.get(component, 1.0)
        if coefficient > 0 and total <= 0:
            raise AssertionError(f"No nonzero gradient in active {component}")
        if coefficient == 0 and total != 0:
            raise AssertionError(f"Disabled projection receives gradient: {component}")
        gradients[component] = float(total)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip,
                                         error_if_nonfinite=True)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    update_seconds = time.perf_counter() - start
    training_peak = torch.cuda.max_memory_allocated()
    training_reserved = torch.cuda.max_memory_reserved()
    model.eval()
    primary = {name: value for name, value in batch.items()
               if not name.startswith("speech_mix_view2")}
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        valid_loss, valid_stats, _ = model(**primary)
        torch.testing.assert_close(valid_loss, valid_stats["loss_sweep"], rtol=0, atol=0)
        if not torch.isfinite(valid_loss):
            raise AssertionError("Primary-view-only validation loss must be finite")
        ctf = model.estimate_ctf(primary["speech_mix"][0])
        rir = model.estimate_rir(primary["speech_mix"][0], rir_length=32000)
    if (ctf.shape != (2, 257, 60) or rir.shape != (2, 32000)
            or not torch.isfinite(ctf).all() or not torch.isfinite(rir).all()):
        raise AssertionError("Expected finite ordinary CTF and two-RIR PIM inference")
    torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - start
    report = dict(
        status="passed", lambda_room=model.lambda_room, lambda_path=model.lambda_path,
        config=str(args.config), data_dir=str(args.data_dir),
        config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(), seed=cfg.seed,
        gpu=torch.cuda.get_device_name(), cuda_version=torch.version.cuda,
        torch_version=torch.__version__, device_index=torch.cuda.current_device(),
        paired_batch_size=2, input_waveforms=4, view_samples=64000, utterances=ids,
        parameters=sum(parameter.numel() for parameter in model.parameters()),
        trainable_parameters=sum(parameter.numel() for parameter in model.parameters()
                                 if parameter.requires_grad),
        predictor_parameters=sum(parameter.numel() for parameter in model.ctf_predictor.parameters()),
        first_update_seconds=update_seconds, validation_and_inference_seconds=inference_seconds,
        peak_training_allocated_bytes=training_peak,
        peak_training_reserved_bytes=training_reserved,
        peak_inference_allocated_bytes=torch.cuda.max_memory_allocated(),
        gradient_norm=float(norm), component_gradient_l1=gradients,
        losses={key: float(value) for key, value in stats.items()},
        valid_loss_sweep=float(valid_stats["loss_sweep"]),
        active_parameters_have_finite_gradients=True,
        primary_view_only_validation=True, inference_requires_view2=False,
        early_stopping_criterion=list(cfg.early_stopping_criterion),
        ctf_shape=list(ctf.shape), rir_shape=list(rir.shape),
        checkpoint_saved=False,
    )
    with args.output.open("x") as destination:
        json.dump(report, destination, indent=2, sort_keys=True)
        destination.write("\n")
    print(f"CUDA paired room/path readiness passed; report: {args.output}")


if __name__ == "__main__":
    main()

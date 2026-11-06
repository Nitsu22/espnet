"""Check NB/Full against the original baseline's data and padded-batch path.

This GPU gate performs short optimizer checks, not baseline retraining. It
neither creates nor loads shared initial weights. Training jobs must start
afresh after this disposable check has published ready.json.
"""

import argparse
import gc
import hashlib
import json
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import soundfile as sf
import torch
import yaml

from espnet2.rir.rec_rir.espnet_model_pit import ESPnetRecRIRPITModel
from espnet2.samplers.folded_batch_sampler import FoldedBatchSampler
from espnet2.tasks.rir import RIRTask

from audit_original_baseline_comparison import audit_configs


VARIANTS = ("nb", "full")
NAMES = (
    "speech_mix", "speech_direct1", "speech_direct2", "speech_reverb1", "speech_reverb2"
)
SPEECH_MODULES = (
    "encoder", "noise_layers", "spch_layers", "decoder_spch", "decoder_rev"
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def config_path(variant):
    return Path(f"conf/tuning/train_rec_rir_original_baseline_{variant}.yaml")


def task_args(variant, output):
    return RIRTask.get_parser().parse_args(
        ["--config", str(config_path(variant)), "--output_dir", str(output)]
    )


def read_scp(path):
    rows = [line.split(maxsplit=1) for line in Path(path).read_text().splitlines()]
    mapping = dict(rows)
    assert len(mapping) == len(rows) and mapping, f"Invalid SCP: {path}"
    return mapping


def input_maps(config, split):
    specs = config[f"{split}_data_path_and_name_and_type"]
    assert [name for _, name, _ in specs] == list(NAMES)
    assert all(kind == "sound" for _, _, kind in specs)
    return {name: read_scp(path) for path, name, _ in specs}


def batch_from_ids(args, maps, ids, train=True):
    processor = RIRTask.build_preprocess_fn(args, train=train)
    rows = []
    np.random.seed(args.seed)
    for uid in ids:
        data = {}
        for name in NAMES:
            # Retain original PCM decoding and channel count. The unchanged
            # baseline preprocessor selects the first channel and crops jointly.
            wave, rate = sf.read(maps[name][uid], dtype="float32")
            assert rate == 8000 and wave.ndim in (1, 2)
            data[name] = wave
        processed = processor(uid, data)
        # ESPnetDataset casts floating arrays after preprocessing. In
        # particular, RecRIRPITPreprocessor returns float64 even for float32
        # input; reproduce the dataset cast before the standalone collate.
        for name in NAMES:
            processed[name] = processed[name].astype(args.train_dtype, copy=False)
        rows.append((uid, processed))
    _, batch = RIRTask.build_collate_fn(args, train=train)(rows)
    return {name: value.cuda() for name, value in batch.items()}


def worst_validation_ids(args, maps, lengths):
    batch_type = args.valid_batch_type or args.batch_type
    assert batch_type == "folded"
    category = Path(args.valid_data_path_and_name_and_type[0][0]).parent / "utt2category"
    sampler = FoldedBatchSampler(
        batch_size=args.valid_batch_size or args.batch_size,
        shape_files=args.valid_shape_file,
        fold_lengths=args.fold_length,
        sort_in_batch=args.sort_in_batch,
        sort_batch=args.sort_batch,
        drop_last=args.drop_last_iter,
        min_batch_size=1,
        utt2category_file=str(category) if category.exists() else None,
    )
    assert all(set(maps[name]) == set(lengths) for name in NAMES)
    return max(sampler, key=lambda ids: len(ids) * max(lengths[uid] for uid in ids))


def legacy_and_pit_checks(model, batch):
    """Exercise the original padded-batch dispatch, including inference."""
    assert model.length_aware is False
    lengths = torch.tensor([16000, 12000, 16000], device="cuda")
    values = {}
    for name in NAMES:
        wave = batch[name][:3, :16000].clone()
        assert wave.shape == (3, 16000)
        wave[1, 12000:] = 0.375
        values[name] = wave
        values[f"{name}_lengths"] = lengths
    calls = []
    handle = model.rec_rir.register_forward_pre_hook(
        lambda module, inputs: calls.append(tuple(inputs[0].shape))
    )
    try:
        with torch.no_grad():
            actual, actual_stats, actual_weight = model(**values)
            # A single full-width network call proves this did not silently
            # group by actual length or remove the deliberately nonzero padding.
            assert calls == [(3, 2, 129, 126)], calls
            expected, expected_stats, expected_weight = ESPnetRecRIRPITModel.forward(
                model, **values
            )
            torch.testing.assert_close(actual.reshape(-1), expected.reshape(-1))
            torch.testing.assert_close(actual_weight, expected_weight)
            assert actual_stats.keys() == expected_stats.keys()
            for name in actual_stats:
                torch.testing.assert_close(actual_stats[name], expected_stats[name])
            swapped = dict(values)
            for kind in ("direct", "reverb"):
                swapped[f"speech_{kind}1"] = values[f"speech_{kind}2"]
                swapped[f"speech_{kind}2"] = values[f"speech_{kind}1"]
            torch.testing.assert_close(
                actual.reshape(-1), model(**swapped)[0].reshape(-1)
            )
            ctf = model.estimate_ctf(values["speech_mix"], lengths)
            reference = ESPnetRecRIRPITModel.estimate_ctf(
                model, values["speech_mix"], lengths
            )
            assert ctf.shape == (3, 2, 129, 60) and torch.isfinite(ctf).all()
            torch.testing.assert_close(ctf, reference)
            single = model.estimate_ctf(values["speech_mix"][0])
            assert single.shape == (2, 129, 60)
            rir = model.estimate_rir(batch["speech_mix"][:1], rir_length=16000)
            assert rir.shape == (1, 2, 16000) and torch.isfinite(rir).all()
    finally:
        handle.remove()


def checkpoint_check(network):
    """Check relation-pooling values, input gradients and parameter gradients."""
    was_training = network.training
    was_checkpoint = network.relation_checkpoint
    network.train()
    clean = torch.randn(1, 2, 65, 96, device="cuda", requires_grad=True)
    reverb = torch.randn_like(clean, requires_grad=True)
    parameters = list(network.clean_proj.parameters())
    parameters += list(network.reverb_proj.parameters())
    parameters += list(network.relation_mlp.parameters())
    values, gradients = [], []
    try:
        for enabled in (False, True):
            network.relation_checkpoint = enabled
            result = network._pool_lag_sequence(clean, reverb)
            values.append(result.detach())
            gradients.append(torch.autograd.grad(
                result.square().mean(), [clean, reverb] + parameters
            ))
        torch.testing.assert_close(values[0], values[1], atol=1e-6, rtol=1e-5)
        for first, second in zip(*gradients):
            assert torch.isfinite(first).all() and torch.isfinite(second).all()
            torch.testing.assert_close(first, second, atol=1e-6, rtol=1e-5)
        network.relation_checkpoint = False
        with torch.no_grad():
            q, k = network.clean_proj(clean), network.reverb_proj(reverb)
            pooled = network._pool_lag_sequence(clean, reverb)
            for lag in (0, 1, 59):
                left, right = q[:, :, :65 - lag], k[:, :, lag:]
                relation = network.relation_mlp(
                    torch.cat([left, right, left * right], dim=-1)
                )
                expected = relation.mean(2, dtype=torch.float32).to(relation.dtype)
                torch.testing.assert_close(pooled[:, :, lag], expected)
    finally:
        network.relation_checkpoint = was_checkpoint
        network.train(was_training)


def check_package(inputs):
    report = json.loads((inputs / "preparation.json").read_text())
    assert report["schema"] == "rec_rir_original_baseline_comparison_v1"
    assert report["no_audio_conversion"] is True
    assert digest(inputs / "baseline_config.yaml") == report["baseline_config_sha256"]
    assert report["config_sha256"] == report["baseline_config_sha256"]
    assert digest(inputs / report["audio_manifest"]) == report["audio_manifest_sha256"]
    for key in ("index_sha256", "shape_sha256"):
        for relative, expected in report[key].items():
            assert digest(inputs / relative) == expected, relative
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=Path("exp/original_baseline_inputs"))
    parser.add_argument("--output", type=Path,
                        default=Path("exp/original_baseline_comparison/profile"))
    parser.add_argument("--audit", type=Path,
                        default=Path("exp/original_baseline_comparison/input_audit.json"))
    args = parser.parse_args()
    assert torch.cuda.is_available(), "This verification gate requires a GPU"
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "ready.json").exists():
        raise FileExistsError("Verification already completed; use a fresh output directory")
    current_audit = audit_configs(args.inputs)
    completed_audit = json.loads(args.audit.read_text())
    assert completed_audit["exact_audio_bytes_verified"] is True
    assert completed_audit["original_baseline_retrained"] is False
    assert completed_audit["shared_speech_initializer"] is False
    for name, expected in current_audit.items():
        assert completed_audit[name] == expected, f"Input audit is stale: {name}"
    preparation = check_package(args.inputs)
    configs = {name: yaml.safe_load(config_path(name).read_text()) for name in VARIANTS}
    for variant, config in configs.items():
        assert config["rir_model_type"] == "rec_rir_pit_lag_relation"
        assert config["model_conf"]["length_aware"] is False
        assert config["model_conf"]["ctf_block_type"] == variant
        assert config["model_conf"]["ctf_length"] == 60
        assert config["model_conf"]["relation_checkpoint"] is True
        assert config["init_param"] == [] and config["pretrain_path"] is None
        assert config["seed"] == 0 and config["batch_size"] == 4
    torch.set_num_threads(1)
    maps = input_maps(configs["nb"], "train")
    ids = []
    for uid in sorted(maps["speech_mix"]):
        if all(sf.info(maps[name][uid]).frames >= 32000 for name in NAMES):
            ids.append(uid)
        if len(ids) == 4:
            break
    assert len(ids) == 4, "Need four original utterances at least four seconds long"
    valid_maps = input_maps(configs["nb"], "valid")
    # The CPU audit verified every WAV byte. Reuse its source header inventory
    # instead of reading 25,000 headers while holding a GPU allocation.
    valid_lengths_by_name = {name: {} for name in NAMES}
    with (args.inputs / preparation["audio_manifest"]).open() as stream:
        for line in stream:
            record = json.loads(line)
            if record["split"] == "valid":
                valid_lengths_by_name[record["name"]][record["uid"]] = record["frames"]
    assert all(set(mapping) == set(valid_maps["speech_mix"])
               for mapping in valid_lengths_by_name.values())
    valid_common_lengths = {
        uid: min(valid_lengths_by_name[name][uid] for name in NAMES)
        for uid in valid_maps["speech_mix"]
    }
    valid_uid, valid_path = max(
        valid_maps["speech_mix"].items(),
        key=lambda row: valid_lengths_by_name["speech_mix"][row[0]],
    )
    report = dict(
        schema="rec_rir_original_baseline_gpu_gate_v1",
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        torch=torch.__version__, gpu=torch.cuda.get_device_name(),
        preparation_sha256=digest(args.inputs / "preparation.json"),
        input_audit_sha256=digest(args.audit),
        baseline_config_sha256=preparation["baseline_config_sha256"],
        original_baseline_checkpoints=preparation["checkpoints"],
        no_audio_conversion=True, shared_speech_initialization=False,
        original_baseline_retrained=False, shared_speech_initializer=False,
        trained_baseline_reused=True, train_ids=ids, variants={},
    )
    for variant in VARIANTS:
        cfg = task_args(variant, args.output)
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.backends.cudnn.enabled = cfg.cudnn_enabled
        torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
        torch.backends.cudnn.deterministic = cfg.cudnn_deterministic
        model = RIRTask.build_model(cfg).cuda().train()
        batch = batch_from_ids(cfg, input_maps(configs[variant], "train"), ids)
        assert cfg.optim == "adamw"
        optimizer = torch.optim.AdamW(model.parameters(), **cfg.optim_conf)
        scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp)
        shapes = []
        handle = model.rec_rir.register_forward_hook(
            lambda module, inputs, outputs: shapes.append([list(x.shape) for x in outputs])
        )
        torch.cuda.reset_peak_memory_stats()
        first = next(model.parameters())
        before = first.detach().clone()
        durations, overflows = [], []
        for attempt in range(32):
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.cuda.amp.autocast(enabled=cfg.use_amp):
                loss, _, weight = model(**batch)
                assert loss.numel() == 1
                loss = loss.reshape(-1).mean()
            assert torch.isfinite(loss) and weight.item() == 4
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            missing = [name for name, param in model.named_parameters()
                       if param.requires_grad and param.grad is None]
            assert not missing, missing
            nonfinite = [name for name, param in model.named_parameters()
                         if param.grad is not None and not torch.isfinite(param.grad).all()]
            if nonfinite:
                if not cfg.use_amp:
                    raise AssertionError(f"Nonfinite FP32 gradients: {nonfinite}")
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                assert scaler.get_scale() < previous_scale
                overflows.append(dict(attempt=attempt, scale=previous_scale,
                                      next_scale=scaler.get_scale(),
                                      nonfinite_parameters=nonfinite[:8]))
                print(variant, "AMP scale calibration", json.dumps(overflows[-1]), flush=True)
                continue
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.grad_clip, norm_type=cfg.grad_clip_type
            )
            assert torch.isfinite(norm)
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.synchronize()
            durations.append(time.perf_counter() - start)
            if len(durations) == 4:
                break
        assert len(durations) == 4, "AMP did not reach four finite optimizer updates"
        assert not torch.equal(before, first), "Optimizer did not change parameters"
        assert all(torch.isfinite(param).all() for param in model.parameters())
        handle.remove()
        assert shapes[0] == [[4, 2, 2, 129, 251], [4, 2, 2, 129, 60],
                             [4, 2, 2, 129, 251]], shapes[0]
        train_peak = torch.cuda.max_memory_allocated() / 2**30
        optimizer.zero_grad(set_to_none=True)
        model.eval()
        legacy_and_pit_checks(model, batch)
        checkpoint_check(model.rec_rir)
        speech_ids = {id(param) for name in SPEECH_MODULES
                      for param in getattr(model.rec_rir, name).parameters()}
        head_ids = {
            id(param)
            for name in ("clean_proj", "reverb_proj", "relation_mlp", "ctf_layers", "decoder_CTF")
            for param in getattr(model.rec_rir, name).parameters()
        }
        assert not speech_ids.intersection(head_ids)
        if variant == "full":
            assert all(layer.full is model.rec_rir.ctf_layers[0].full
                       for layer in model.rec_rir.ctf_layers)
        valid_ids = worst_validation_ids(cfg, valid_maps, valid_common_lengths)
        valid_batch = batch_from_ids(cfg, valid_maps, valid_ids, train=False)
        valid_shapes = []
        handle = model.rec_rir.register_forward_hook(
            lambda module, inputs, outputs: valid_shapes.append([list(x.shape) for x in outputs])
        )
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        valid_start = time.perf_counter()
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=cfg.use_amp):
            valid_loss, valid_stats, valid_weight = model(**valid_batch)
        torch.cuda.synchronize()
        valid_seconds = time.perf_counter() - valid_start
        valid_peak = torch.cuda.max_memory_allocated() / 2**30
        handle.remove()
        assert valid_loss.numel() == 1 and torch.isfinite(valid_loss).all()
        assert all(torch.isfinite(value).all() for value in valid_stats.values())
        valid_size, valid_samples = valid_batch["speech_mix"].shape
        assert valid_size == len(valid_ids) and valid_weight.item() == valid_size
        assert valid_samples == max(valid_common_lengths[uid] for uid in valid_ids)
        valid_frames = valid_samples // cfg.model_conf["hop_len"] + 1
        assert valid_shapes == [
            [[valid_size, 2, 2, 129, valid_frames], [valid_size, 2, 2, 129, 60],
             [valid_size, 2, 2, 129, valid_frames]]
        ], valid_shapes
        del valid_loss, valid_stats, valid_weight
        wave, rate = sf.read(valid_path, dtype="float32")
        assert rate == 8000 and wave.ndim in (1, 2)
        # The baseline's force_single_channel preprocessor selects channel 0.
        if wave.ndim == 2:
            wave = wave[:, 0]
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=cfg.use_amp):
            ctf = model.estimate_ctf(torch.from_numpy(wave.copy()).cuda().unsqueeze(0))
        assert ctf.shape == (1, 2, 129, 60) and torch.isfinite(ctf).all()
        item = dict(
            config_sha256=digest(config_path(variant)),
            parameters=sum(param.numel() for param in model.parameters()),
            train_peak_gib=train_peak,
            worst_folded_valid_peak_gib=valid_peak,
            worst_folded_valid_seconds=valid_seconds,
            worst_folded_valid_ids=list(valid_ids),
            worst_folded_valid_batch_size=valid_size,
            worst_folded_valid_samples=valid_samples,
            worst_folded_valid_output_shapes=valid_shapes[0],
            longest_valid_single_inference_peak_gib=torch.cuda.max_memory_allocated() / 2**30,
            update_seconds=float(np.median(durations[1:])),
            last_check_loss=float(loss.detach()), batch_size=4,
            amp_calibration_overflows=overflows, final_grad_scale=scaler.get_scale(),
            successful_optimizer_updates=len(durations), output_shapes=shapes[0],
            longest_valid_id=valid_uid, longest_valid_samples=len(wave),
            legacy_forward_agreement=True, legacy_inference_agreement=True,
            common_pit=True, rir_finite=True, finite_gradients=True,
            checkpoint_agreement=True, causal_lag_endpoints=True,
            independent_ctf_parameters=True, shared_speech_initialization=False,
            validation_memory_scope="actual folded batch maximizing batch size * padded length",
        )
        report["variants"][variant] = item
        (args.output / f"{variant}.json").write_text(json.dumps(item, indent=2) + "\n")
        print(variant, json.dumps(item), flush=True)
        del model, optimizer, scaler, batch, valid_batch, loss, first, before, ctf
        gc.collect()
        torch.cuda.empty_cache()
    # Publishing last prevents partially successful checks from enabling jobs.
    ready = args.output / "ready.json"
    pending = args.output / "ready.json.tmp"
    pending.write_text(json.dumps(report, indent=2) + "\n")
    pending.replace(ready)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

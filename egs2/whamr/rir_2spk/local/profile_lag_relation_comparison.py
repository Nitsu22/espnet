"""Gate the three-head NF8k comparison on real-data CUDA checks and timings."""

import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import soundfile as sf
import torch
import yaml

from espnet2.tasks.rir import RIRTask
from espnet2.torch_utils.load_pretrained_model import load_pretrained_model


VARIANTS = ('baseline', 'nb', 'full')
NAMES = ('speech_mix', 'speech_direct1', 'speech_direct2',
         'speech_reverb1', 'speech_reverb2')
SHARED = ('encoder', 'noise_layers', 'spch_layers', 'decoder_spch', 'decoder_rev')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def config_path(variant):
    return Path(f'conf/tuning/train_rec_rir_nf8k_fair_{variant}.yaml')


def task_args(variant, output):
    return RIRTask.get_parser().parse_args(
        ['--config', str(config_path(variant)), '--output_dir', str(output)])


def read_scp(path):
    return dict(line.split(maxsplit=1) for line in path.read_text().splitlines())


def batch_from_ids(args, folder, ids):
    processor = RIRTask.build_preprocess_fn(args, train=True)
    maps = {name: read_scp(folder / f'{name}.scp') for name in NAMES}
    rows = []
    np.random.seed(args.seed)
    for uid in ids:
        data = {}
        for name in NAMES:
            wave, rate = sf.read(maps[name][uid], dtype='float32')
            assert rate == 8000 and wave.ndim == 1
            data[name] = wave
        rows.append((uid, processor(uid, data)))
    _, batch = RIRTask.build_collate_fn(args, train=True)(rows)
    return {name: value.cuda() for name, value in batch.items()}


def length_and_pit_checks(model, batch):
    # Deliberately unequal lengths and nonzero padding exercise pre-STFT grouping.
    values = {}
    lengths = torch.tensor([16000, 12000, 16000], device='cuda')
    for name in NAMES:
        wave = batch[name][:3, :16000].clone()
        assert wave.shape == (3, 16000)
        wave[1, 12000:] = 123.0
        values[name] = wave
        values[name + '_lengths'] = lengths
    with torch.no_grad():
        grouped, _, _ = model(**values)
        separate = []
        for idx, length in enumerate(lengths.tolist()):
            item = {name: values[name][idx:idx + 1, :length] for name in NAMES}
            item.update({name + '_lengths': lengths[idx:idx + 1] for name in NAMES})
            separate.append(model(**item)[0])
        torch.testing.assert_close(grouped, torch.stack(separate).mean())
        changed = {key: value.clone() for key, value in values.items()}
        for name in NAMES:
            changed[name][1, 12000:] = -456.0
        torch.testing.assert_close(grouped, model(**changed)[0])
        swapped = dict(values)
        for kind in ('direct', 'reverb'):
            swapped[f'speech_{kind}1'] = values[f'speech_{kind}2']
            swapped[f'speech_{kind}2'] = values[f'speech_{kind}1']
        torch.testing.assert_close(grouped, model(**swapped)[0])
        ctf = model.estimate_ctf(values['speech_mix'], lengths)
        assert ctf.shape == (3, 2, 129, 60)
        # Check inference restores original order after sorting length groups.
        for idx, length in enumerate(lengths.tolist()):
            expected = model.estimate_ctf(values['speech_mix'][idx:idx + 1, :length])
            torch.testing.assert_close(ctf[idx:idx + 1], expected)
        rir = model.estimate_rir(batch['speech_mix'][:1], rir_length=16000)
        assert rir.shape == (1, 2, 16000) and torch.isfinite(rir).all()
    try:
        model.estimate_ctf(torch.zeros(1, 7551, device='cuda'))
    except ValueError as error:
        assert '7552' in str(error)
    else:
        raise AssertionError('CTF60 accepted fewer than 60 valid frames')


def checkpoint_check(network):
    network.train()
    clean = torch.randn(1, 2, 65, 96, device='cuda', requires_grad=True)
    reverb = torch.randn_like(clean, requires_grad=True)
    parameters = list(network.clean_proj.parameters())
    parameters += list(network.reverb_proj.parameters())
    parameters += list(network.relation_mlp.parameters())
    values, gradients = [], []
    for enabled in (False, True):
        network.relation_checkpoint = enabled
        result = network._pool_lag_sequence(clean, reverb)
        values.append(result.detach())
        gradients.append(torch.autograd.grad(result.square().mean(),
                                            [clean, reverb] + parameters))
    torch.testing.assert_close(values[0], values[1], atol=1e-6, rtol=1e-5)
    for first, second in zip(*gradients):
        torch.testing.assert_close(first, second, atol=1e-6, rtol=1e-5)
    # Independently inspect the lag endpoints, before any temporal blocks.
    network.relation_checkpoint = False
    with torch.no_grad():
        q, k = network.clean_proj(clean), network.reverb_proj(reverb)
        pooled = network._pool_lag_sequence(clean, reverb)
        for lag in (0, 1, 59):
            expected = network.relation_mlp(torch.cat(
                [q[:, :, :65-lag], k[:, :, lag:],
                 q[:, :, :65-lag] * k[:, :, lag:]], dim=-1)).mean(2)
            torch.testing.assert_close(pooled[:, :, lag], expected)
    network.relation_checkpoint = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dump', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'ready.json').exists():
        raise FileExistsError('Comparison is already profiled; do not overwrite it')
    torch.set_num_threads(1)
    configs = {variant: yaml.safe_load(config_path(variant).read_text())
               for variant in VARIANTS}
    common = []
    for config in configs.values():
        value = copy.deepcopy(config)
        value.pop('rir_model_type')
        for key in ('length_aware', 'dim_output_CTF', 'ctf_length',
                    'ctf_block_type', 'relation_checkpoint'):
            value['model_conf'].pop(key, None)
        common.append(value)
    assert common[0] == common[1] == common[2], 'Non-head training settings differ'
    assert configs['baseline']['model_conf']['length_aware'] is True
    source_args = task_args('baseline', args.output)
    torch.manual_seed(source_args.seed)
    baseline = RIRTask.build_model(source_args)
    # Save only random, untrained speech weights; every head still starts afresh.
    shared = {name: value for name, value in baseline.state_dict().items()
              if any(name.startswith(f'rec_rir.{module}.') for module in SHARED)}
    initial = args.output / 'shared_speech_init.pth'
    torch.save(shared, initial)
    del baseline
    folder = args.dump / 'raw/tr_rir_2spk_nf_min_8k'
    lengths = dict((uid, int(length)) for uid, length in
                   (line.split() for line in (folder / 'utt2num_samples').read_text().splitlines()))
    ids = [uid for uid in sorted(lengths) if lengths[uid] >= 32000][:4]
    assert len(ids) == 4
    report = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  torch=torch.__version__, gpu=torch.cuda.get_device_name(),
                  dataset_sha256=digest(args.dump / 'preparation.json'),
                  shared_init_sha256=digest(initial), variants={})
    for variant in VARIANTS:
        cfg = task_args(variant, args.output)
        torch.manual_seed(cfg.seed)
        model = RIRTask.build_model(cfg).cuda().train()
        load_pretrained_model(str(initial), model, False, map_location='cuda')
        for name, value in shared.items():
            assert torch.equal(model.state_dict()[name].cpu(), value)
        if variant == 'baseline':
            # The optional wrapper keeps original checkpoint keys and architecture.
            from espnet2.rir.rec_rir.espnet_model_pit import ESPnetRecRIRPITModel
            old_config = dict(configs[variant]['model_conf'])
            old_config.pop('length_aware')
            old = ESPnetRecRIRPITModel(**old_config)
            old.load_state_dict(model.state_dict(), strict=True)
            del old
        batch = batch_from_ids(cfg, folder, ids)
        optimizer = torch.optim.AdamW(model.parameters(), **cfg.optim_conf)
        scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp)
        shapes = []
        handle = model.rec_rir.register_forward_hook(
            lambda module, inputs, outputs: shapes.append([list(x.shape) for x in outputs]))
        durations = []
        torch.cuda.reset_peak_memory_stats()
        first = next(model.parameters())
        before = first.detach().clone()
        for step in range(4):
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.cuda.amp.autocast(enabled=cfg.use_amp):
                loss, _, weight = model(**batch)
            assert torch.isfinite(loss) and weight.item() == 4
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            missing = [name for name, parameter in model.named_parameters()
                       if parameter.requires_grad and parameter.grad is None]
            assert not missing, missing
            assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters()
                       if parameter.grad is not None)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.synchronize()
            durations.append(time.perf_counter() - start)
        assert not torch.equal(before, first)
        handle.remove()
        assert shapes[0] == [[4, 2, 2, 129, 251], [4, 2, 2, 129, 60],
                             [4, 2, 2, 129, 251]], shapes[0]
        peak = torch.cuda.max_memory_allocated() / 2**30
        optimizer.zero_grad(set_to_none=True)
        model.eval()
        length_and_pit_checks(model, batch)
        if variant != 'baseline':
            checkpoint_check(model.rec_rir)
            speech_ids = {id(p) for name in SHARED for p in getattr(model.rec_rir, name).parameters()}
            assert not speech_ids.intersection(id(p) for p in model.rec_rir.ctf_layers.parameters())
            if variant == 'full':
                assert all(layer.full is model.rec_rir.ctf_layers[0].full
                           for layer in model.rec_rir.ctf_layers)
        # Include the longest validation input in the memory/shape gate.
        model.eval()
        valid = args.dump / 'raw/cv_rir_2spk_nf_min_8k'
        uid, length = max((line.split() for line in (valid / 'utt2num_samples').read_text().splitlines()),
                          key=lambda row: int(row[1]))
        wave, rate = sf.read(read_scp(valid / 'speech_mix.scp')[uid], dtype='float32')
        assert rate == 8000 and len(wave) == int(length)
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=cfg.use_amp):
            ctf = model.estimate_ctf(torch.from_numpy(wave).cuda().unsqueeze(0))
        assert ctf.shape == (1, 2, 129, 60) and torch.isfinite(ctf).all()
        item = dict(config_sha256=digest(config_path(variant)),
                    parameters=sum(p.numel() for p in model.parameters()),
                    train_peak_gib=peak, valid_peak_gib=torch.cuda.max_memory_allocated() / 2**30,
                    update_seconds=float(np.median(durations[1:])),
                    initial_loss=float(loss.detach()), batch_size=4,
                    output_shapes=shapes[0], longest_valid_samples=int(length),
                    shared_speech_initialization=True, finite_gradients=True,
                    padding_invariant=True, utterance_mean=True, common_pit=True,
                    checkpoint_agreement=variant != 'baseline')
        report['variants'][variant] = item
        (args.output / f'{variant}.json').write_text(json.dumps(item, indent=2) + '\n')
        print(variant, json.dumps(item), flush=True)
        del model, optimizer, scaler, batch, loss, first, before, ctf
        gc.collect()
        torch.cuda.empty_cache()
    # Publish the gate last, so downstream jobs never consume partial results.
    (args.output / 'ready.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()

"""One scratch optimizer update and RIR inference on a real training example."""
import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from espnet2.tasks.rir import RIRTask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    task_args = RIRTask.get_parser().parse_args(['--config', args.config, '--output_dir', str(args.output_dir)])
    assert not task_args.init_param
    torch.manual_seed(task_args.seed)
    np.random.seed(task_args.seed)
    model = RIRTask.build_model(task_args).cuda().train()
    processor = RIRTask.build_preprocess_fn(task_args, train=True)
    data = {}
    ids = []
    sweep_only = task_args.rir_model_type == 'rec_rir_sweep'
    names = ('speech_mix', 'rir_ref') if sweep_only else ('speech_mix', 'speech_direct', 'speech_reverb')
    if sweep_only:
        assert not hasattr(model.rec_rir, 'decoder_spch')
        assert not hasattr(model.rec_rir, 'decoder_rev')
    for name in names:
        uid, path = (args.data_dir / f'{name}.scp').read_text().splitlines()[0].split(maxsplit=1)
        ids.append(uid)
        wav, sr = sf.read(path)
        assert sr == 16000 and wav.ndim == 2 and wav.shape[1] == 2
        data[name] = wav
    assert len(set(ids)) == 1
    data = processor(ids[0], data)
    batch = {}
    for name, wav in data.items():
        assert wav.ndim == 1
        batch[name] = torch.tensor(wav, dtype=torch.float32, device='cuda')[None]
        batch[name + '_lengths'] = torch.tensor([len(wav)], device='cuda')
    optimizer = torch.optim.AdamW(model.parameters(), **task_args.optim_conf)
    first = next(model.parameters())
    before = first.detach().clone()
    scaler = torch.cuda.amp.GradScaler(enabled=task_args.use_amp)
    with torch.cuda.amp.autocast(enabled=task_args.use_amp):
        loss, stats, weight = model(**batch)
    assert torch.isfinite(loss)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    torch.nn.utils.clip_grad_norm_(model.parameters(), task_args.grad_clip)
    scaler.step(optimizer)
    scaler.update()
    assert not torch.equal(before, first)
    model.eval()
    with torch.no_grad():
        rir = model.estimate_rir(batch['speech_mix'][0], rir_length=32000)
    assert rir.shape == (32000,) and torch.isfinite(rir).all()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = dict(uid=ids[0], loss=float(loss.detach()), samples=int(batch['speech_mix'].shape[1]),
                  scratch=True, sweep_only=sweep_only, amp=task_args.use_amp, finite_gradients=True,
                  parameters_updated=True, rir_samples=len(rir), torch=torch.__version__)
    (args.output_dir / 'check.json').write_text(json.dumps(report, indent=2))
    sf.write(args.output_dir / 'rir.wav', rir.cpu().numpy(), 16000, subtype='FLOAT')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

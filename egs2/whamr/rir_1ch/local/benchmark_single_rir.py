"""Comparable scratch FP32 training/inference timings; no dataset I/O timed."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from espnet2.tasks.rir import RIRTask


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True, type=Path)
    p.add_argument('--data-dir', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--warmup', type=int, default=3)
    p.add_argument('--iterations', type=int, default=10)
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    if min(args.batch_size, args.warmup, args.iterations) < 1:
        p.error('Batch size, warmup and iterations must be positive')
    if args.output.exists():
        p.error('Use a new output file')
    device = torch.device(args.device)
    if device.type != 'cuda':
        p.error('Timing requires CUDA for consistent synchronized measurements')
    config = RIRTask.get_parser().parse_args([
        '--config', str(args.config), '--output_dir', '/tmp/rir-benchmark-unused'])
    if config.use_amp or config.init_param or config.optim != 'adamw':
        p.error('This benchmark expects scratch FP32 AdamW configurations')
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    model = RIRTask.build_model(config).to(device).train()
    processor = RIRTask.build_preprocess_fn(config, train=True)
    names = (['speech_mix', 'rir_ref'] if config.rir_model_type == 'rec_rir_sweep'
             else ['speech_mix', 'speech_direct', 'speech_reverb'])
    if config.rir_model_type == 'rec_rir_sweep_v2':
        names = ['speech_mix', 'rir_ref', 'rir_direct']
    maps = {n: dict(line.split(maxsplit=1) for line in
                   (args.data_dir / (n + '.scp')).read_text().splitlines()) for n in names}
    ids = list(maps['speech_mix'])[:args.batch_size]
    if len(ids) != args.batch_size:
        raise ValueError('Insufficient utterances')
    examples = []
    for uid in ids:
        data = {}
        for name in names:
            wav, sr = sf.read(maps[name][uid])
            if sr != 16000:
                raise ValueError('Expected 16 kHz')
            data[name] = wav
        examples.append(processor(uid, data))
    batch = {}
    for name in names:
        sizes = [len(x[name]) for x in examples]
        if len(set(sizes)) != 1:
            raise ValueError('Use equal-duration examples to avoid padding in timings')
        batch[name] = torch.tensor(np.stack([x[name] for x in examples]),
                                   dtype=torch.float32, device=device)
        batch[name + '_lengths'] = torch.tensor(sizes, device=device)
    if batch['speech_mix'].shape[-1] != 64000:
        raise ValueError('Expected four-second input crops')
    optimizer = torch.optim.AdamW(model.parameters(), **config.optim_conf)

    def train_step():
        optimizer.zero_grad(set_to_none=True)
        loss, _, _ = model(**batch)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite loss')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip,
                                              error_if_nonfinite=True)
        optimizer.step()

    def measure(fn):
        for _ in range(args.warmup):
            fn()
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        times = []
        for _ in range(args.iterations):
            start = time.perf_counter()
            fn()
            torch.cuda.synchronize(device)
            times.append(time.perf_counter() - start)
        return dict(mean_seconds=float(np.mean(times)), median_seconds=float(np.median(times)),
                    std_seconds=float(np.std(times)),
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(device))

    training = measure(train_step)
    optimizer.zero_grad(set_to_none=True)
    del optimizer
    model.eval()
    speech = batch['speech_mix'][0].clone()
    del batch
    torch.cuda.empty_cache()

    @torch.no_grad()
    def infer():
        rir = model.estimate_rir(speech, rir_length=32000)
        if rir.shape != (32000,) or not torch.isfinite(rir).all():
            raise RuntimeError('Invalid RIR inference')

    inference = measure(infer)
    report = dict(config=str(args.config), config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
                  gpu=torch.cuda.get_device_name(device), torch=torch.__version__,
                  parameters=sum(x.numel() for x in model.parameters()),
                  trainable_parameters=sum(x.numel() for x in model.parameters() if x.requires_grad),
                  batch_size=args.batch_size, samples=64000, inference_batch_size=1,
                  warmup=args.warmup, iterations=args.iterations, utterances=ids,
                  input_sha256=hashlib.sha256(speech.cpu().numpy().tobytes()).hexdigest(),
                  training=training, inference=inference,
                  note='Scratch FP32; training includes loss/backward/clip/AdamW; inference includes CTF-to-RIR; excludes data I/O')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        json.dump(report, f, indent=2, allow_nan=False)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

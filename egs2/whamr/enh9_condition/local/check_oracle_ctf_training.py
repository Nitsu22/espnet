"""Check real Oracle inputs, an optimizer step and checkpoint-based inference.

Probe weights are discarded. GPU measurements use a paid compute allocation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import subprocess

import numpy as np
import soundfile as sf
import torch

from espnet2.bin.enh_rir_ctf_inference import SeparateSpeech
from espnet2.tasks.enh_rir import EnhancementRIRCTFTask
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
from espnet2.torch_utils.device_funcs import to_device
from prepare_oracle_ctf_dump import validate_prepared_dump


def read_scp(path):
    return dict(line.split(maxsplit=1) for line in path.read_text().splitlines())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--dump', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-smoke', action='store_true')
    args = parser.parse_args()
    manifest = validate_prepared_dump(args.dump, require_full=not args.allow_smoke)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('A CUDA allocation is required')
    set_all_random_seed(0)
    config = EnhancementRIRCTFTask.get_parser().parse_args(
        ['--config', str(args.config), '--output_dir', str(args.output.parent)]
    )
    model = EnhancementRIRCTFTask.build_model(config).to(args.device)
    is_film = config.separator == 'tflocoformer_nocashe_ctf_film_all'
    if not is_film and config.separator != 'tflocoformer_nocashe_ctf_concat':
        raise ValueError('Expected an Oracle CTF separator')
    if config.separator_conf['ctf_taps'] != 60:
        raise ValueError('Oracle dump requires the native 60-tap grid')
    if config.encoder_conf != {'n_fft': 256, 'hop_length': 64}:
        raise ValueError('Keep the DAI-8 speech STFT 256/hop64')
    preprocess = EnhancementRIRCTFTask.build_preprocess_fn(config, train=True)
    collate = EnhancementRIRCTFTask.build_collate_fn(config, train=True)

    def maps(split):
        folder = args.dump / 'raw' / f'{split}_mix_clean_reverb_min_8k'
        return {name: read_scp(folder / (name + '.scp'))
                for name in ('wav', 'spk1', 'spk2', 'rir_ctf')}

    def example(values, uid):
        result = {}
        for source, target in [('wav', 'speech_mix'), ('spk1', 'speech_ref1'),
                               ('spk2', 'speech_ref2')]:
            wave, rate = sf.read(
                values[source][uid], dtype=config.train_dtype, always_2d=True
            )
            if rate != 8000:
                raise ValueError('Unexpected sample rate')
            result[target] = wave
        result['rir_ctf'] = np.load(values['rir_ctf'][uid], allow_pickle=False)
        return result

    training = maps('tr')
    eligible = [uid for uid in training['wav']
                if manifest['baseline_audio']['tr'][uid]['wav.scp']['frames'] >= 32000]
    if len(eligible) < 4:
        raise ValueError('Need four full-crop examples for batch4 validation')
    examples = [(uid, preprocess(uid, example(training, uid))) for uid in eligible[:4]]
    _, batch = collate(examples)
    batch = to_device(batch, args.device)
    if tuple(batch['speech_mix'].shape[:2]) != (4, 32000):
        raise ValueError('Expected a physical four-example, four-second crop')
    if batch['speech_mix'].dtype != torch.float32:
        raise ValueError('Keep the DAI-8 FP32 audio reader')
    if tuple(batch['rir_ctf'].shape) != (4, 60, 2, 129, 2):
        raise ValueError('CTF axes were changed by cropping/collation')
    if args.device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), **config.optim_conf)
    # Identity FiLM initially blocks gradients into its context encoder.
    # A second update must propagate gradients through the learned FiLM path.
    for update in range(2 if is_film else 1):
        loss, _, _ = model(**batch)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite training loss')
        loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise ValueError('Nonfinite gradient: ' + name)
        projection = model.separator.ctf_projection
        prompt_grad = (projection[0] if is_film else projection).weight.grad
        if update == (1 if is_film else 0):
            if prompt_grad is None or not prompt_grad.abs().sum().item():
                raise ValueError('No gradient reaches the CTF projection')
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        training_loss = float(loss.detach())
    del batch, examples, loss, optimizer
    validation = maps('cv')
    longest = max(
        validation['wav'],
        key=lambda uid: manifest['baseline_audio']['cv'][uid]['wav.scp']['frames'],
    )
    valid_preprocess = EnhancementRIRCTFTask.build_preprocess_fn(config, train=False)
    _, validation_batch = collate([(longest, valid_preprocess(longest, example(validation, longest)))])
    validation_batch = to_device(validation_batch, args.device)
    model.eval()
    with torch.no_grad():
        validation_loss, _, _ = model(**validation_batch)
    if not torch.isfinite(validation_loss):
        raise ValueError('Nonfinite full-length validation loss')
    max_allocated = torch.cuda.max_memory_allocated() if args.device == 'cuda' else None
    max_reserved = torch.cuda.max_memory_reserved() if args.device == 'cuda' else None
    del validation_batch
    # The existing CTF inference runner must read the same model/config contract.
    uid = eligible[0]
    sample = valid_preprocess(uid, example(training, uid))
    sample['speech_mix'] = sample['speech_mix'][:8000]
    with tempfile.TemporaryDirectory(prefix='oracle_ctf_probe_') as scratch:
        checkpoint = Path(scratch) / 'probe.pth'
        torch.save(model.state_dict(), checkpoint)
        inference = SeparateSpeech(
            train_config=str(args.config), model_file=str(checkpoint),
            device=args.device, dtype='float32', normalize_output_wav=True,
        )
        # Avoid retaining two copies of model parameters while measuring memory.
        del model
        waves = inference(sample['speech_mix'][None], sample['rir_ctf'][None])
        if len(waves) != 2 or any(w.shape != (1, 8000) or not np.isfinite(w).all() for w in waves):
            raise ValueError('Checkpoint inference output is invalid')
    report = dict(status='passed', device=args.device, counts=manifest['counts'],
                  separator=config.separator,
                  config=str(args.config.resolve()), parameters=sum(p.numel() for p in inference.enh_model.parameters()),
                  config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
                  manifest_sha256=hashlib.sha256((args.dump / 'preparation.json').read_bytes()).hexdigest(),
                  commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  train_ids=eligible[:4], training_loss=training_loss,
                  longest_valid_id=longest, longest_valid_samples=sf.info(validation['wav'][longest]).frames,
                  validation_loss=float(validation_loss),
                  peak_allocated_bytes=max_allocated, peak_reserved_bytes=max_reserved,
                  checkpoint_inference=True, probe_weights_saved=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()

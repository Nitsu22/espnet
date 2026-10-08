"""GPU scratch update, teacher-permutation invariance, and two-RIR inference."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import soundfile as sf
import torch
from espnet2.tasks.rir import RIRTask


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--data-dir',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--batch-size',type=int,default=4)
    parser.add_argument('--warmup-steps', type=int, default=0)
    parser.add_argument('--timed-steps', type=int, default=0)
    parser.add_argument('--valid-data-dir', type=Path)
    args=parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a new output')
    cfg=RIRTask.get_parser().parse_args(['--config',str(args.config),'--output_dir','/tmp/two-spk-check'])
    assert not cfg.init_param and not cfg.use_amp
    torch.manual_seed(cfg.seed);np.random.seed(cfg.seed)
    model=RIRTask.build_model(cfg).cuda().train()
    proc=RIRTask.build_preprocess_fn(cfg,train=True)
    sweep=cfg.rir_model_type in ('tflocoformer_sweep_v2_pit','pooled_bimamba_sweep_v2_pit')
    prefixes=('rir_ref','rir_direct') if sweep else ('speech_direct','speech_reverb')
    names=['speech_mix']+[f'{p}{i}' for p in prefixes for i in (1,2)]
    maps={k:dict(line.split(maxsplit=1) for line in (args.data_dir/f'{k}.scp').read_text().splitlines()) for k in names}
    reverb_maps=[dict(line.split(maxsplit=1) for line in (args.data_dir/f'speech_reverb{i}.scp').read_text().splitlines()) for i in (1,2)]
    ids=list(maps['speech_mix'])[:args.batch_size];examples=[]
    for uid in ids:
        data={}
        for k in names:
            data[k],sr=sf.read(maps[k][uid]);assert sr==16000
        clean_sum=sum(sf.read(m[uid])[0] for m in reverb_maps)
        assert np.max(np.abs(clean_sum-data['speech_mix']))<2e-6, 'Input must be noise-free mixture'
        examples.append(proc(uid,data))
    batch={}
    for k in names:
        lengths=[len(e[k]) for e in examples]
        assert len(set(lengths))==1
        batch[k]=torch.tensor(np.stack([e[k] for e in examples]),dtype=torch.float32,device='cuda')
        batch[k+'_lengths']=torch.tensor(lengths,device='cuda')
    assert batch['speech_mix'].shape[-1]==64000
    opt=torch.optim.AdamW(model.parameters(),**cfg.optim_conf)
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
    loss,stats,_=model(**batch)
    assert torch.isfinite(loss).all()
    loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.grad_clip,error_if_nonfinite=True)
    opt.step();opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize();seconds=time.perf_counter()-start
    peak=torch.cuda.max_memory_allocated()
    # Optional charged GPU profiling. These updates are discarded; training
    # starts from the unchanged config seed, not this measurement model.
    if args.warmup_steps < 0 or args.timed_steps < 0:
        raise ValueError('Use nonnegative step counts')
    step_seconds = []
    for step in range(args.warmup_steps + args.timed_steps):
        torch.cuda.synchronize()
        start = time.perf_counter()
        measured_loss = model(**batch)[0]
        assert torch.isfinite(measured_loss)
        measured_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip, error_if_nonfinite=True)
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        if step >= args.warmup_steps:
            step_seconds.append(time.perf_counter() - start)
    peak = max(peak, torch.cuda.max_memory_allocated())
    model.eval()
    with torch.no_grad():
        current=model(**batch)[0]
        swapped=dict(batch)
        for prefix in prefixes:
            swapped[prefix+'1'],swapped[prefix+'2']=batch[prefix+'2'],batch[prefix+'1']
        permuted=model(**swapped)[0]
        torch.testing.assert_close(current,permuted,rtol=2e-5,atol=2e-6)
        rir=model.estimate_rir(batch['speech_mix'][0],rir_length=32000)
    assert rir.shape==(2,32000) and torch.isfinite(rir).all()
    valid_profile = []
    if args.valid_data_dir is not None:
        valid_proc = RIRTask.build_preprocess_fn(cfg, train=False)
        valid_maps = {k: dict(line.split(maxsplit=1) for line in
                             (args.valid_data_dir / f'{k}.scp').read_text().splitlines()) for k in names}
        lengths = dict(line.split() for line in
                       (args.valid_data_dir / 'utt2num_samples').read_text().splitlines())
        ordered = sorted(lengths, key=lambda uid: int(lengths[uid]))
        # Five equal-probability bin midpoints approximate validation time.
        # The maximum-length sample separately checks worst-case memory.
        with torch.no_grad():
            for quantile in (.1, .3, .5, .7, .9, 1.0):
                uid = ordered[min(len(ordered)-1, int(quantile * (len(ordered)-1)))]
                example = {}
                for k in names:
                    example[k], rate = sf.read(valid_maps[k][uid])
                    assert rate == 16000
                example = valid_proc(uid, example)
                observation = {k: torch.tensor(v[None], dtype=torch.float32, device='cuda')
                               for k, v in example.items()}
                for k, v in example.items():
                    observation[k + '_lengths'] = torch.tensor([len(v)], device='cuda')
                assert observation['speech_mix'].shape[1] == int(lengths[uid])
                torch.cuda.reset_peak_memory_stats()
                model(**observation)  # warm up kernels for this sequence length
                torch.cuda.synchronize()
                start = time.perf_counter()
                valid_loss = model(**observation)[0]
                torch.cuda.synchronize()
                assert torch.isfinite(valid_loss)
                valid_profile.append(dict(quantile=quantile, uid=uid,
                                          samples=int(lengths[uid]), seconds=time.perf_counter()-start,
                                          peak_allocated_bytes=torch.cuda.max_memory_allocated()))
    report=dict(config=str(args.config),config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
                gpu=torch.cuda.get_device_name(),batch_size=args.batch_size,utterances=ids,
                parameters=sum(p.numel() for p in model.parameters()),
                first_update_seconds=seconds,peak_allocated_bytes=peak,gradient_norm=float(norm),
                losses={k:float(v) for k,v in stats.items()},teacher_permutation_invariant=True,
                noise_free_mixture_verified=True,rir_shape=list(rir.shape))
    if step_seconds:
        report['steady_update_seconds'] = float(np.median(step_seconds))
        report['timed_step_seconds'] = step_seconds
    if valid_profile:
        report['validation_profile'] = valid_profile
        report['estimated_valid_step_seconds'] = float(np.mean([x['seconds'] for x in valid_profile[:-1]]))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()

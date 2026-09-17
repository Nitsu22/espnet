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
    args=parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a new output')
    cfg=RIRTask.get_parser().parse_args(['--config',str(args.config),'--output_dir','/tmp/two-spk-check'])
    assert not cfg.init_param and not cfg.use_amp
    torch.manual_seed(cfg.seed);np.random.seed(cfg.seed)
    model=RIRTask.build_model(cfg).cuda().train()
    proc=RIRTask.build_preprocess_fn(cfg,train=True)
    sweep=cfg.rir_model_type=='tflocoformer_sweep_v2_pit'
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
    report=dict(config=str(args.config),config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
                gpu=torch.cuda.get_device_name(),batch_size=args.batch_size,utterances=ids,
                parameters=sum(p.numel() for p in model.parameters()),
                first_update_seconds=seconds,peak_allocated_bytes=peak,gradient_norm=float(norm),
                losses={k:float(v) for k,v in stats.items()},teacher_permutation_invariant=True,
                noise_free_mixture_verified=True,rir_shape=list(rir.shape))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()

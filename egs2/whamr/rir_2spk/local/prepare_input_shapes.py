"""Build input-only batching shapes from header-verified two-speaker indexes."""
import argparse
import hashlib
import json
import yaml
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--dump',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    config=yaml.safe_load(args.config.read_text())
    assert config['rir_model_type'] in ('rec_rir_pit','tflocoformer_ctf_pit','tflocoformer_sweep_v2_pit')
    assert config['model_conf']['extract_feats_in_collect_stats'] is False
    assert config['force_single_channel'] is True
    assert config['preprocessor_conf']['force_single_channel'] is True
    saved=json.loads((args.dump/'preparation.json').read_text())
    assert saved['spec']['num_spk']==2 and saved['spec']['sample_rate']=='16k'
    assert saved['spec']['length']=='min' and saved['spec']['noise'] is False
    report={}
    for split,mode in [('tr','train'),('cv','valid')]:
        folder=Path('raw')/f'{split}_rir_2spk_nf_min_16k'
        for name in ('utt2num_samples','wav.scp'):
            path=folder/name
            actual=hashlib.sha256((args.dump/path).read_bytes()).hexdigest()
            assert actual==saved['output_hashes'][str(path)],path
        shape=(args.dump/folder/'utt2num_samples').read_text()
        rows=dict(line.split() for line in shape.splitlines())
        ids={line.split(maxsplit=1)[0] for line in (args.dump/folder/'wav.scp').read_text().splitlines()}
        assert set(rows)==ids and len(rows)==saved['counts'][split]
        assert all(int(n)>0 for n in rows.values())
        dest=args.output/mode;dest.mkdir(parents=True,exist_ok=True)
        target=dest/'speech_mix_shape'
        if target.exists():assert target.read_text()==shape
        else:target.write_text(shape)
        report[mode]=dict(count=len(rows),sha256=hashlib.sha256(shape.encode()).hexdigest())
    (args.output/'input_shapes.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()

import unittest
import numpy as np
import torch

from espnet2.rir.rec_rir.tflocoformer_sweep_v2_pit import ESPnetTFLocoformerSweepV2PITModel
from espnet2.train.preprocessor_rec_rir_sweep_v2_pit import RecRIRSweepV2PITPreprocessor


def model():
    return ESPnetTFLocoformerSweepV2PITModel(
        sr=16000,n_fft=64,win_len=64,hop_len=32,num_freqs=33,ctf_taps=5,
        n_layers=1,emb_dim=8,num_groups=2,n_heads=2,attention_dim=8,
        pos_enc='nope',ffn_hidden_dim=[8,8],conv1d_kernel=3,pim_sweep_duration=.1)


def test_paired_pit_permutation_and_tail():
    m=model()
    direct=torch.zeros(2,2,256)
    direct[:,0,20]=1; direct[:,1,30]=-2
    target=torch.zeros_like(direct)
    target[:,0,20]=3; target[:,1,158]=-1
    ctf=torch.zeros(2,2,33,5,dtype=torch.complex64)
    ctf[:,0,:,0]=3;ctf[:,1,:,4]=.5
    loss,_=m.paired_pit_loss(ctf,direct,target)
    assert loss<1e-5,loss
    swapped=ctf.flip(1).detach().requires_grad_()
    loss2,_=m.paired_pit_loss(swapped,direct,target)
    torch.testing.assert_close(loss,loss2)
    # Swap just one reference pair in one example: PIT must be per utterance.
    d2=direct.clone();r2=target.clone()
    d2[0]=direct[0].flip(0);r2[0]=target[0].flip(0)
    loss3,perm=m.paired_pit_loss(ctf,d2,r2)
    torch.testing.assert_close(loss,loss3)
    assert perm[0]!=perm[1]
    wrong=torch.zeros_like(ctf,requires_grad=True)
    bad,_=m.paired_pit_loss(wrong,direct,target)
    bad.backward()
    assert bad>0 and torch.isfinite(wrong.grad).all()
    assert wrong.grad[...,4].abs().sum()>0


def test_forward_and_preprocessor():
    proc=RecRIRSweepV2PITPreprocessor(True,speech_segment=256,rir_length=256)
    data={'speech_mix':np.random.randn(512,2)}
    for i in (1,2):
        h=np.zeros((256,2));h[50,0]=i
        data[f'rir_direct{i}']=h.copy();h[100,0]=.2*i
        data[f'rir_ref{i}']=h
    prepared=proc('test',data)
    assert len(prepared['speech_mix'])==256
    assert all(prepared[f'rir_direct{i}'][50]==1 for i in (1,2))
    batch={k:torch.tensor(v,dtype=torch.float32)[None] for k,v in prepared.items()}
    batch['speech_mix_lengths']=torch.tensor([256])
    m=model();loss,stats,_=m(**batch);loss.backward()
    grads=[p.grad for p in m.parameters() if p.grad is not None]
    assert torch.isfinite(loss).all() and grads and all(torch.isfinite(g).all() for g in grads)
    assert sum(g.abs().sum() for g in grads)>0


if __name__=='__main__':
    suite=unittest.TestSuite(unittest.FunctionTestCase(v) for k,v in list(globals().items()) if k.startswith('test_'))
    raise SystemExit(not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful())

"""Compare optimized local windows to independently clipped official Mamba."""
import argparse
import copy
import json
from pathlib import Path

import torch

from espnet2.rir.rec_rir.pooled_bimamba import TimeLocalBiMamba


def oracle(block, x):
    batch, frames, freqs, dim = x.shape
    seq = block.norm(x).permute(0, 2, 1, 3).reshape(batch * freqs, frames, dim)
    outputs = []
    for t in range(frames):
        left, right = max(0, t-block.radius), min(frames, t+block.radius+1)
        clip = seq[:, left:right]
        forward = block.forward_mamba(clip.contiguous())[:, t-left]
        backward = block.backward_mamba(clip.flip(1).contiguous()).flip(1)[:, t-left]
        outputs.append(block.projection(torch.cat((forward, backward), -1)))
    return x + torch.stack(outputs, 1).reshape(batch, freqs, frames, dim).permute(0, 2, 1, 3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a fresh validation output')
    torch.manual_seed(0)
    results = []
    for frames, window, chunk, freqs in ((11, 5, 7, 2), (129, 125, 257, 2),
                                       (7, 5, 257, 129)):
        block = TimeLocalBiMamba(8, 4, 4, window, 1e-5, chunk).cuda().train()
        reference = copy.deepcopy(block)
        x = torch.randn(1, frames, freqs, 8, device='cuda', requires_grad=True)
        ref_x = x.detach().clone().requires_grad_()
        actual, expected = block(x), oracle(reference, ref_x)
        torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)
        probe = torch.randn_like(actual)
        (actual * probe).sum().backward()
        (expected * probe).sum().backward()
        torch.testing.assert_close(x.grad, ref_x.grad, rtol=1e-4, atol=1e-5)
        refs = dict(reference.named_parameters())
        error = 0.0
        for name, p in block.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
            torch.testing.assert_close(p.grad, refs[name].grad, rtol=2e-4, atol=8e-5)
            error = max(error, float((p.grad-refs[name].grad).abs().max()))
        results.append(dict(frames=frames, window=window,
                            output_max_error=float((actual-expected).abs().max()),
                            input_gradient_max_error=float((x.grad-ref_x.grad).abs().max()),
                            parameter_gradient_max_error=error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(gpu=torch.cuda.get_device_name(),
                                           official_mamba_output_and_gradient_match=True,
                                           cases=results), indent=2)+'\n')


if __name__ == '__main__':
    main()

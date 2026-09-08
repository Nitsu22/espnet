"""Check real-data training, distributed gradients, and checkpoint round-trip."""

import argparse
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from espnet2.tasks.enh import EnhancementTask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--dumpdir", default=os.environ.get(
        "TFGRIDNET_DUMP_DIR", "dump_roland_sms_wsj_1ch"))
    args = parser.parse_args()
    rank = int(os.environ.get("LOCAL_RANK", 0))
    if args.cpu:
        device = torch.device("cpu")
        torch.set_num_threads(2)
    else:
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
        dist.init_process_group("nccl")
    config = "conf/tuning/train_tfgridnet_sms_wsj_1ch_paper.yaml"
    opts = EnhancementTask.get_parser().parse_args([
        "--config", config, "--output_dir", "exp_tfgridnet_sms_wsj_1ch_paper/smoke"
    ])
    torch.manual_seed(opts.seed)
    model = EnhancementTask.build_model(opts).to(device)
    preprocess = EnhancementTask.build_preprocess_fn(opts, train=True)
    root = Path(args.dumpdir) / "raw/train_si284_directpath"
    data = {}
    for filename, key in [("wav.scp", "speech_mix"),
                          ("spk1.scp", "speech_ref1"),
                          ("spk2.scp", "speech_ref2")]:
        line = (root / filename).read_text().splitlines()[rank]
        uid, path = line.split(maxsplit=1)
        audio, rate = sf.read(path, dtype="float32", always_2d=True)
        assert rate == 8000 and audio.shape[1] == 6
        data[key] = audio
    raw = {k: v.copy() for k, v in data.items()}
    np.random.seed(0)
    start = np.random.randint(0, len(data["speech_mix"]) - 32000)
    np.random.seed(0)
    data = preprocess(uid, data)
    for key in data:
        assert data[key].ndim == 1
        # For these samples the preprocessor should just select microphone 0.
        if np.max(np.abs(raw["speech_mix"])) <= 1:
            np.testing.assert_allclose(data[key], raw[key][start:start + 32000, 0])
    size = 1024 if args.cpu else 32000
    batch = {k: torch.from_numpy(v[:size]).unsqueeze(0).to(device)
             for k, v in data.items()}
    batch["speech_mix_lengths"] = torch.tensor([size], device=device)
    assert sum(p.numel() for p in model.parameters()) == 5494120
    if args.cpu:
        model.eval()
        with torch.no_grad():
            spectrum, _ = model.separator.enc(batch["speech_mix"], batch["speech_mix_lengths"])
            direct = torch.stft(
                batch["speech_mix"], n_fft=256, hop_length=64,
                window=torch.hann_window(256).sqrt(), return_complex=True,
            ).transpose(1, 2)
            torch.testing.assert_close(spectrum.real, direct.real)
            torch.testing.assert_close(spectrum.imag, direct.imag)
            restored, _ = model.separator.dec(spectrum, batch["speech_mix_lengths"])
            torch.testing.assert_close(restored, batch["speech_mix"], atol=1e-6, rtol=1e-5)
            loss1 = model(**batch)[0]
            swapped = dict(batch, speech_ref1=batch["speech_ref2"],
                           speech_ref2=batch["speech_ref1"])
            loss2 = model(**swapped)[0]
            torch.testing.assert_close(loss1, loss2)
            scaled = {k: v * 3 if k != "speech_mix_lengths" else v
                      for k, v in batch.items()}
            torch.testing.assert_close(loss1, model(**scaled)[0], rtol=1e-4, atol=1e-5)
        # Compare the implementation with the explicit joint PIT formula.
        pred, _, other = model.separator(batch["speech_mix"], batch["speech_mix_lengths"])
        wrapper = model.loss_wrappers[0]
        refs = [batch["speech_ref1"], batch["speech_ref2"]]
        scale = other["tfgridnet_mix_std"]
        costs = []
        for order in [(0, 1), (1, 0)]:
            value = 0
            for i, j in enumerate(order):
                r = refs[i] / scale
                rt = model.separator.enc(r, batch["speech_mix_lengths"])[0]
                st = model.separator.enc(pred[j] / scale, batch["speech_mix_lengths"])[0]
                value += ((r - pred[j] / scale).abs().mean()
                          + (rt.abs() - st.abs()).abs().mean())
            costs.append(value)
        rs, ps = sum(refs) / scale, sum(pred) / scale
        rm = model.separator.enc(rs, batch["speech_mix_lengths"])[0].abs()
        pm = model.separator.enc(ps, batch["speech_mix_lengths"])[0].abs()
        expected = torch.stack(costs).min() + (rs - ps).abs().mean() + (rm - pm).abs().mean()
        torch.testing.assert_close(loss1.squeeze(), expected)
        del pred, other, costs, expected
    model.train()
    training_model = model if args.cpu else DistributedDataParallel(model, device_ids=[rank])
    optimizer = torch.optim.Adam(model.parameters(), **opts.optim_conf)
    loss, _, _ = training_model(**batch)
    assert torch.isfinite(loss).all()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.parameters() if p.requires_grad)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), opts.grad_clip)
    optimizer.step()
    if rank == 0:
        output = Path("exp_tfgridnet_sms_wsj_1ch_paper/smoke")
        output.mkdir(parents=True, exist_ok=True)
        checkpoint = output / ("cpu.pth" if args.cpu else "gpu.pth")
        torch.save(model.state_dict(), checkpoint)
        saved = torch.load(checkpoint, map_location="cpu")
        assert saved.keys() == model.state_dict().keys()
    print(f"PASS rank={rank} samples={size} loss={loss.item():.6f} grad_norm={norm.item():.6f}", flush=True)
    if not args.cpu:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

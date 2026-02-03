#!/usr/bin/env python3
"""Convert a multi-channel checkpoint to a single-channel checkpoint for the first conv.

This script replaces `<module_key>.weight` (shape: [out, 2*M, kH, kW]) with a
single-channel version (shape: [out, 2, kH, kW]) by averaging across microphone
channels, separately for real and imaginary parts.

For tflocoformer_nocashe_mc, the input channel order is:
  [real_m0 .. real_m{M-1}, imag_m0 .. imag_m{M-1}]
so the reduction is:
  dst[:, 0] = mean(src[:, 0:M])      (real)
  dst[:, 1] = mean(src[:, M:2*M])    (imag)
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Average multi-channel conv weights to single-channel for a state_dict-only .pth"
        )
    )
    p.add_argument("--src", type=Path, required=True, help="Source .pth (state_dict)")
    p.add_argument("--dst", type=Path, required=True, help="Destination .pth (state_dict)")
    p.add_argument(
        "--module_key",
        type=str,
        default="separator.conv.0",
        help="Module key prefix in the state_dict (default: separator.conv.0)",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite --dst if it already exists",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.dst.exists() and not args.overwrite:
        raise SystemExit(f"[ERROR] Destination already exists: {args.dst} (use --overwrite)")

    try:
        import torch
    except Exception as e:
        raise SystemExit(f"[ERROR] Failed to import torch: {type(e).__name__}: {e}") from e

    state = torch.load(str(args.src), map_location="cpu")
    if not isinstance(state, dict):
        raise SystemExit(f"[ERROR] Expected a state_dict (dict), but got {type(state)} from {args.src}")

    weight_key = f"{args.module_key}.weight"
    bias_key = f"{args.module_key}.bias"

    if weight_key not in state:
        raise KeyError(f"'{weight_key}' not found in state_dict: {args.src}")
    if bias_key not in state:
        raise KeyError(f"'{bias_key}' not found in state_dict: {args.src}")

    w = state[weight_key]
    b = state[bias_key]

    if getattr(w, "ndim", None) != 4:
        raise ValueError(f"Expected 4D conv weight, but got shape={tuple(getattr(w, 'shape', ())) }")

    out_ch, in_ch, k_h, k_w = w.shape
    if in_ch % 2 != 0:
        raise ValueError(f"Expected even in_channels (real+imag), but got shape={tuple(w.shape)}")
    m = in_ch // 2
    if m < 1:
        raise ValueError(f"Invalid in_channels: {in_ch}")

    # Average across microphone channels, separately for real and imag.
    w_real = w[:, :m, :, :].mean(dim=1)
    w_imag = w[:, m:, :, :].mean(dim=1)
    w_new = torch.stack([w_real, w_imag], dim=1)  # (out_ch, 2, k_h, k_w)

    state[weight_key] = w_new
    state[bias_key] = b

    args.dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, str(args.dst))

    print(f"[INFO] Saved conv-averaged checkpoint: {args.dst}")
    print(f"[INFO] {weight_key}: {tuple(w.shape)} -> {tuple(w_new.shape)} (M={m})")


if __name__ == "__main__":
    main()


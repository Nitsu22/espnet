import logging
import os
import time
from pathlib import Path
from typing import Any

import torch

_CTX = {"utt_id": None}


def _rank() -> str:
    for key in ("ESPNET_DEBUG_RANK", "RANK", "LOCAL_RANK"):
        value = os.getenv(key)
        if value:
            return value
    return "0"


def _format_utt_id(utt_id: Any) -> str:
    if utt_id is None:
        return ""
    if isinstance(utt_id, (list, tuple)):
        preview = ",".join(str(x) for x in utt_id[:2])
        if len(utt_id) > 2:
            preview += ",..."
        return preview
    return str(utt_id)


def _log_line(message: str) -> None:
    debug_dir = os.getenv("ESPNET_DEBUG_DIR")
    if not debug_dir:
        logging.info(message)
        return
    path = Path(debug_dir) / f"forward_trace.rank{_rank()}.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", buffering=1) as fp:
        fp.write(message + "\n")


def _shape_summary(obj: Any) -> str:
    if isinstance(obj, torch.Tensor):
        return f"shape={tuple(obj.shape)} dtype={obj.dtype} device={obj.device}"
    if isinstance(obj, (list, tuple)):
        items = []
        for item in obj[:2]:
            items.append(_shape_summary(item))
        suffix = ",..." if len(obj) > 2 else ""
        return "[" + ", ".join(items) + suffix + "]"
    if isinstance(obj, dict):
        keys = list(obj.keys())[:2]
        items = []
        for key in keys:
            items.append(f"{key}={_shape_summary(obj[key])}")
        suffix = ",..." if len(obj) > 2 else ""
        return "{" + ", ".join(items) + suffix + "}"
    return type(obj).__name__


def _make_pre_hook(name: str, use_kwargs: bool = False):
    def _hook(module, args, kwargs=None):
        if use_kwargs and kwargs is not None and "utt_id" in kwargs:
            _CTX["utt_id"] = _format_utt_id(kwargs.get("utt_id"))
        utt_id = _CTX.get("utt_id", "")
        ts = time.strftime("%Y-%m-%dT%H:%M:%S")
        message = f"{ts} rank={_rank()} stage=enter name={name}"
        if utt_id:
            message += f" utt_id={utt_id}"
        if os.getenv("ESPNET_FWD_LOG_SHAPES") == "1":
            message += f" input={_shape_summary(args)}"
        _log_line(message)
    return _hook


def _make_post_hook(name: str):
    def _hook(module, args, output):
        utt_id = _CTX.get("utt_id", "")
        ts = time.strftime("%Y-%m-%dT%H:%M:%S")
        message = f"{ts} rank={_rank()} stage=exit name={name}"
        if utt_id:
            message += f" utt_id={utt_id}"
        if os.getenv("ESPNET_FWD_LOG_SHAPES") == "1":
            message += f" output={_shape_summary(output)}"
        _log_line(message)
    return _hook


def _register_hooks(module: torch.nn.Module, name: str, use_kwargs: bool = False) -> None:
    module.register_forward_pre_hook(_make_pre_hook(name, use_kwargs), with_kwargs=use_kwargs)
    module.register_forward_hook(_make_post_hook(name))


def attach_forward_logging(model: torch.nn.Module) -> None:
    if getattr(model, "_forward_log_attached", False):
        return
    setattr(model, "_forward_log_attached", True)

    _register_hooks(model, "model", use_kwargs=True)

    for attr in ("encoder", "separator", "decoder"):
        module = getattr(model, attr, None)
        if module is not None:
            _register_hooks(module, attr)

    detail = int(os.getenv("ESPNET_FWD_LOG_DETAIL", "0"))
    separator = getattr(model, "separator", None)
    if separator is None:
        return

    if detail >= 1 and hasattr(separator, "blocks"):
        for idx, block in enumerate(separator.blocks):
            _register_hooks(block, f"separator.blocks.{idx}")

    if detail >= 2:
        for idx, block in enumerate(getattr(separator, "blocks", [])):
            for sub_name in ("freq_path", "frame_path"):
                sub = getattr(block, sub_name, None)
                if sub is not None:
                    _register_hooks(sub, f"separator.blocks.{idx}.{sub_name}")

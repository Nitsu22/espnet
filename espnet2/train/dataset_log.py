import logging
import os
import time
from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
from torch.utils.data import get_worker_info

from espnet2.train.dataset import ESPnetDataset

_WARN_SEC = None


def _rank_for_log() -> str:
    for key in ("ESPNET_DEBUG_RANK", "RANK", "LOCAL_RANK"):
        value = os.getenv(key)
        if value:
            return value
    return "0"


def _warn_threshold() -> float:
    global _WARN_SEC
    if _WARN_SEC is not None:
        return _WARN_SEC
    value = os.getenv("ESPNET_DATA_LOAD_WARN_SEC")
    if not value:
        _WARN_SEC = 0.0
        return _WARN_SEC
    try:
        _WARN_SEC = float(value)
        return _WARN_SEC
    except ValueError:
        logging.warning("Invalid ESPNET_DATA_LOAD_WARN_SEC: %s", value)
        _WARN_SEC = 0.0
        return _WARN_SEC


def _log_line(message: str, worker_id: str) -> None:
    debug_dir = os.getenv("ESPNET_DEBUG_DIR")
    if not debug_dir:
        logging.info(message)
        return
    rank = _rank_for_log()
    path = Path(debug_dir) / f"data_load.rank{rank}.worker{worker_id}.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fp:
        fp.write(message + "\n")


class ESPnetDatasetLog(ESPnetDataset):
    def __getitem__(self, uid: Union[str, int]) -> Tuple[str, Dict[str, np.ndarray]]:
        start = time.perf_counter()
        result = super().__getitem__(uid)
        elapsed = time.perf_counter() - start

        warn_sec = _warn_threshold()
        if warn_sec > 0 and elapsed >= warn_sec:
            info = get_worker_info()
            worker_id = str(info.id) if info is not None else "main"
            utt_id, data = result
            data_keys = ",".join(sorted(data.keys()))
            ts = time.strftime("%Y-%m-%dT%H:%M:%S")
            _log_line(
                f"{ts} uid={utt_id} sec={elapsed:.3f} keys={data_keys}", worker_id
            )

        return result

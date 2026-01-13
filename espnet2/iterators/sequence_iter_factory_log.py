import itertools
import logging
import os
from functools import partial
from pathlib import Path
from typing import Optional

import numpy as np
from torch.utils.data import DataLoader
from typeguard import typechecked

from espnet2.iterators.sequence_iter_factory import SequenceIterFactory, worker_init_fn


def _rank_for_log() -> str:
    for key in ("ESPNET_DEBUG_RANK", "RANK", "LOCAL_RANK"):
        value = os.getenv(key)
        if value:
            return value
    return "0"


def _log_line(message: str) -> None:
    debug_dir = os.getenv("ESPNET_DEBUG_DIR")
    if not debug_dir:
        logging.info(message)
        return
    path = Path(debug_dir) / f"iter_factory.rank{_rank_for_log()}.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fp:
        fp.write(message + "\n")


def _parse_timeout() -> float:
    value = os.getenv("ESPNET_DATALOADER_TIMEOUT")
    if not value:
        return 0.0
    try:
        timeout = float(value)
    except ValueError:
        _log_line(f"invalid_dataloader_timeout value={value}")
        return 0.0
    if timeout <= 0:
        return 0.0
    return timeout


class SequenceIterFactoryLog(SequenceIterFactory):
    @typechecked
    def build_iter(self, epoch: int, shuffle: Optional[bool] = None) -> DataLoader:
        if shuffle is None:
            shuffle = self.shuffle

        if self.num_iters_per_epoch is not None:
            n_batches = len(self.sampler)
            if self.num_iters_per_epoch < n_batches:
                n_batches = len(self.sampler)
                real_epoch, offset = divmod(self.num_iters_per_epoch * epoch, n_batches)

                if offset >= self.num_iters_per_epoch:
                    current_batches = self.sampler.generate(real_epoch + self.seed)
                    if shuffle:
                        np.random.RandomState(real_epoch + self.seed).shuffle(
                            current_batches
                        )
                    batches = current_batches[
                        offset - self.num_iters_per_epoch : offset
                    ]
                else:
                    prev_batches = self.sampler.generate(real_epoch - 1 + self.seed)
                    current_batches = self.sampler.generate(real_epoch + self.seed)
                    if shuffle:
                        np.random.RandomState(real_epoch - 1 + self.seed).shuffle(
                            prev_batches
                        )
                        np.random.RandomState(real_epoch + self.seed).shuffle(
                            current_batches
                        )
                    batches = (
                        prev_batches[offset - self.num_iters_per_epoch :]
                        + current_batches[:offset]
                    )

            else:
                _epoch, _cursor = divmod(self.num_iters_per_epoch * (epoch - 1), n_batches)
                _remain = self.num_iters_per_epoch
                batches = []
                current_batches = self.sampler.generate(_epoch + self.seed)
                if shuffle:
                    np.random.RandomState(_epoch + self.seed).shuffle(current_batches)
                while _remain > 0:
                    _batches = current_batches[_cursor : _cursor + _remain]
                    batches += _batches
                    if _cursor + _remain >= n_batches:
                        _epoch += 1
                        _cursor = 0
                        current_batches = self.sampler.generate(_epoch + self.seed)
                        if shuffle:
                            np.random.RandomState(_epoch + self.seed).shuffle(
                                current_batches
                            )
                    else:
                        _cursor = _cursor + _remain
                    _remain -= len(_batches)

                assert len(batches) == self.num_iters_per_epoch

        else:
            batches = self.sampler.generate(epoch + self.seed)
            if shuffle:
                np.random.RandomState(epoch + self.seed).shuffle(batches)

        if self.collate_fn is not None:
            kwargs = dict(collate_fn=self.collate_fn)
        else:
            kwargs = {}

        if self.shuffle_within_batch:
            batch_size = len(batches[0])
            batches = list(itertools.chain(*batches))
            np.random.RandomState(epoch + self.seed).shuffle(batches)
            reshuffled = []
            for ii in range(0, len(batches), batch_size):
                reshuffled.append(batches[ii : ii + batch_size])
            batches = reshuffled
            del reshuffled

        timeout = _parse_timeout()
        if timeout > 0 and self.num_workers > 0:
            kwargs["timeout"] = timeout
        elif timeout > 0 and self.num_workers == 0:
            _log_line(
                f"dataloader_timeout_ignored epoch={epoch} num_workers=0 timeout={timeout}"
            )

        if os.getenv("ESPNET_DEBUG_STALL") == "1":
            _log_line(
                "build_iter "
                f"epoch={epoch} shuffle={shuffle} num_workers={self.num_workers} "
                f"timeout={kwargs.get('timeout', 0)} num_batches={len(batches)}"
            )

        return DataLoader(
            dataset=self.dataset,
            batch_sampler=batches,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            worker_init_fn=partial(worker_init_fn, base_seed=epoch + self.seed),
            **kwargs,
        )

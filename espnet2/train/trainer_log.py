import logging
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from typeguard import typechecked

from espnet2.schedulers.abs_scheduler import AbsBatchStepScheduler, AbsScheduler
from espnet2.torch_utils.add_gradient_noise import add_gradient_noise
from espnet2.torch_utils.device_funcs import to_device
from espnet2.torch_utils.recursive_op import recursive_average
from espnet2.utils.kwargs2args import kwargs2args
from espnet2.train.distributed_utils import DistributedOption
from espnet2.train.reporter import SubReporter
from espnet2.train.trainer import GradScaler, Trainer, TrainerOptions, autocast, autocast_args

if torch.distributed.is_available():
    from torch.distributed import ReduceOp


class TrainerLog(Trainer):
    @classmethod
    @typechecked
    def train_one_epoch(
        cls,
        model: torch.nn.Module,
        iterator: Iterable[Tuple[List[str], Dict[str, torch.Tensor]]],
        optimizers: Sequence[torch.optim.Optimizer],
        schedulers: Sequence[Optional[AbsScheduler]],
        scaler: Optional[GradScaler],
        reporter: SubReporter,
        summary_writer,
        options: TrainerOptions,
        distributed_option: DistributedOption,
    ) -> bool:
        grad_noise = options.grad_noise
        accum_grad = options.accum_grad
        grad_clip = options.grad_clip
        grad_clip_type = options.grad_clip_type
        log_interval = options.log_interval
        no_forward_run = options.no_forward_run
        ngpu = options.ngpu
        use_wandb = options.use_wandb
        create_graph_in_tensorboard = options.create_graph_in_tensorboard
        distributed = distributed_option.distributed

        debug = os.getenv("ESPNET_DEBUG_STALL") == "1"
        debug_every = int(os.getenv("ESPNET_DEBUG_EVERY_N", "1"))
        debug_dir = os.getenv("ESPNET_DEBUG_DIR")
        iter_gap_warn = float(os.getenv("ESPNET_DEBUG_ITER_GAP_SEC", "0"))
        log_all_ranks = os.getenv("ESPNET_LOG_ALL_RANKS") == "1"

        if log_interval is None:
            try:
                log_interval = max(len(iterator) // 20, 10)
            except TypeError:
                log_interval = 100

        model.train()
        all_steps_are_invalid = True
        iterator_stop = torch.tensor(0).to("cuda" if ngpu > 0 else "cpu")

        rank = torch.distributed.get_rank() if distributed else 0
        if debug:
            os.environ["ESPNET_DEBUG_RANK"] = str(rank)
        if log_all_ranks:
            logging.getLogger().setLevel(logging.INFO)

        debug_fp = None
        if debug and debug_dir:
            Path(debug_dir).mkdir(parents=True, exist_ok=True)
            debug_path = Path(debug_dir) / f"train_trace.rank{rank}.log"
            debug_fp = debug_path.open("a", encoding="utf-8", buffering=1)

        def _dbg(stage: str, iiter: int, extra: str = ""):
            if not debug:
                return
            ts = time.strftime("%Y-%m-%dT%H:%M:%S")
            msg = (
                f"{ts} rank={rank} epoch={reporter.get_epoch()} "
                f"iter={iiter} stage={stage}"
            )
            if extra:
                msg += f" {extra}"
            if debug_fp is not None:
                debug_fp.write(msg + "\n")
            if debug_every > 0 and iiter % debug_every == 0:
                logging.info(msg)

        start_time = time.perf_counter()
        last_iter_end = start_time
        for iiter, (utt_id, batch) in enumerate(
            reporter.measure_iter_time(iterator, "iter_time"), 1
        ):
            assert isinstance(batch, dict), type(batch)

            if debug and iter_gap_warn > 0:
                gap = time.perf_counter() - last_iter_end
                if gap >= iter_gap_warn:
                    _dbg("iter_gap", iiter, f"gap_sec={gap:.3f}")

            if distributed:
                if debug:
                    _dbg("pre_all_reduce_iter_stop", iiter)
                torch.distributed.all_reduce(iterator_stop, ReduceOp.SUM)
                if debug:
                    _dbg("post_all_reduce_iter_stop", iiter)
                if iterator_stop > 0:
                    break

            if isinstance(utt_id, (list, tuple)):
                utt_preview = ",".join(utt_id[:2])
                if len(utt_id) > 2:
                    utt_preview += ",..."
            else:
                utt_preview = str(utt_id)

            if debug:
                _dbg("iter_start", iiter, f"utt_id={utt_preview}")

            batch["utt_id"] = utt_id
            batch = to_device(batch, "cuda" if ngpu > 0 else "cpu")

            if no_forward_run:
                all_steps_are_invalid = False
                continue

            if (
                create_graph_in_tensorboard
                and iiter == 1
                and summary_writer is not None
            ):
                if distributed:
                    _model = getattr(model, "module")
                else:
                    _model = model
                    if _model is not None:
                        try:
                            _args = kwargs2args(_model.forward, batch)
                        except (ValueError, TypeError):
                            logging.warning(
                                "inpect.signature() is failed for the model. "
                                "The graph can't be added for tensorboard."
                            )
                        else:
                            try:
                                summary_writer.add_graph(
                                    _model, _args, use_strict_trace=False
                                )
                            except Exception:
                                logging.warning(
                                    "summary_writer.add_graph() "
                                    "is failed for the model. "
                                    "The graph can't be added for tensorboard."
                                )
                            del _args
                    else:
                        logging.warning(
                            "model.module is not found (This should be a bug.)"
                        )
                del _model

            with autocast(
                scaler is not None,
                **autocast_args,
            ):
                with reporter.measure_time("forward_time"):
                    if debug:
                        _dbg("before_forward", iiter)
                    retval = model(**batch)
                    if debug:
                        _dbg("after_forward", iiter)

                    # Supporting two patterns for the returned value from the model
                    if isinstance(retval, dict):
                        loss = retval["loss"]
                        stats = retval["stats"]
                        weight = retval["weight"]
                        optim_idx = retval.get("optim_idx")
                        if optim_idx is not None and not isinstance(optim_idx, int):
                            if not isinstance(optim_idx, torch.Tensor):
                                raise RuntimeError(
                                    "optim_idx must be int or 1dim torch.Tensor, "
                                    f"but got {type(optim_idx)}"
                                )
                            if optim_idx.dim() >= 2:
                                raise RuntimeError(
                                    "optim_idx must be int or 1dim torch.Tensor, "
                                    f"but got {optim_idx.dim()}dim tensor"
                                )
                            if optim_idx.dim() == 1:
                                for v in optim_idx:
                                    if v != optim_idx[0]:
                                        raise RuntimeError(
                                            "optim_idx must be 1dim tensor "
                                            "having same values for all entries"
                                        )
                                optim_idx = optim_idx[0].item()
                            else:
                                optim_idx = optim_idx.item()

                    else:
                        loss, stats, weight = retval
                        optim_idx = None

                stats = {k: v for k, v in stats.items() if v is not None}
                if ngpu > 1 or distributed:
                    loss = (loss * weight.type(loss.dtype)).sum()
                    if debug:
                        _dbg("before_recursive_average", iiter)
                    stats, weight = recursive_average(stats, weight, distributed)
                    if debug:
                        _dbg("after_recursive_average", iiter)
                    loss /= weight
                if distributed:
                    loss *= torch.distributed.get_world_size()

                loss /= accum_grad

            reporter.register(stats, weight)

            with reporter.measure_time("backward_time"):
                if debug:
                    _dbg("before_backward", iiter)
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                if debug:
                    _dbg("after_backward", iiter)

            if iiter % accum_grad == 0:
                if scaler is not None:
                    for iopt, optimizer in enumerate(optimizers):
                        if optim_idx is not None and iopt != optim_idx:
                            continue
                        scaler.unscale_(optimizer)

                if grad_noise:
                    add_gradient_noise(
                        model,
                        reporter.get_total_count(),
                        duration=100,
                        eta=1.0,
                        scale_factor=0.55,
                    )

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=grad_clip,
                    norm_type=grad_clip_type,
                )
                if not isinstance(grad_norm, torch.Tensor):
                    grad_norm = torch.tensor(grad_norm)

                if not torch.isfinite(grad_norm):
                    logging.warning(
                        f"The grad norm is {grad_norm}. Skipping updating the model."
                    )

                    if scaler is not None:
                        for iopt, optimizer in enumerate(optimizers):
                            if optim_idx is not None and iopt != optim_idx:
                                continue
                            scaler.step(optimizer)
                            scaler.update()

                else:
                    reporter.register(
                        {
                            "grad_norm": grad_norm,
                            "clip": torch.where(
                                grad_norm > grad_clip,
                                grad_norm.new_tensor(100),
                                grad_norm.new_tensor(0),
                            ),
                            "loss_scale": scaler.get_scale() if scaler else 1.0,
                        }
                    )
                    all_steps_are_invalid = False
                    with reporter.measure_time("optim_step_time"):
                        for iopt, (optimizer, scheduler) in enumerate(
                            zip(optimizers, schedulers)
                        ):
                            if optim_idx is not None and iopt != optim_idx:
                                continue
                            if scaler is not None:
                                scaler.step(optimizer)
                                scaler.update()
                            else:
                                optimizer.step()
                            if isinstance(scheduler, AbsBatchStepScheduler):
                                scheduler.step()
                for iopt, optimizer in enumerate(optimizers):
                    if optim_idx is not None and iopt != optim_idx:
                        continue
                    optimizer.zero_grad()

                reporter.register(
                    dict(
                        {
                            f"optim{i}_lr{j}": pg["lr"]
                            for i, optimizer in enumerate(optimizers)
                            for j, pg in enumerate(optimizer.param_groups)
                            if "lr" in pg
                        },
                        train_time=time.perf_counter() - start_time,
                    ),
                )
                start_time = time.perf_counter()

            reporter.next()
            if iiter % log_interval == 0:
                logging.info(reporter.log_message(-log_interval))
                if summary_writer is not None:
                    reporter.tensorboard_add_scalar(summary_writer, -log_interval)
                if use_wandb:
                    reporter.wandb_log()
            last_iter_end = time.perf_counter()

        else:
            if distributed:
                iterator_stop.fill_(1)
                torch.distributed.all_reduce(iterator_stop, ReduceOp.SUM)
        if debug_fp is not None:
            debug_fp.close()
        return all_steps_are_invalid

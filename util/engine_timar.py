import copy
import json
import math
import os
import shutil
import sys
import time
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.amp.autocast_mode
import torch.nn.functional as F
from tqdm import tqdm

from . import lr_sched, misc
from .metrics import average_metrics, compute_sample_metrics


def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


@torch.no_grad()
def normalize_tensor(
    x: torch.Tensor, mean: np.ndarray, std: np.ndarray, eps: float = 1e-8
) -> torch.Tensor:
    device = x.device
    mean_t = torch.from_numpy(mean).to(device=device, dtype=x.dtype)
    std_t = torch.from_numpy(std).to(device=device, dtype=x.dtype)
    return (x - mean_t) / (std_t + eps)


@torch.no_grad()
def denormalize_tensor(
    z: torch.Tensor, mean: np.ndarray, std: np.ndarray
) -> torch.Tensor:
    device = z.device
    mean_t = torch.from_numpy(mean).to(device=device, dtype=z.dtype)
    std_t = torch.from_numpy(std).to(device=device, dtype=z.dtype)
    return z * std_t + mean_t


def train_one_epoch(
    model,
    model_params,
    ema_params,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    log_writer=None,
    args=None,
):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 20

    optimizer.zero_grad()

    if log_writer is not None:
        print("log_dir: {}".format(log_writer.log_dir))

    for data_iter_step, (
        file_name,
        speech_1,
        motion_1,
        speech_2,
        motion_2,
    ) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        lr_sched.adjust_learning_rate(
            optimizer, data_iter_step / len(data_loader) + epoch, args  # type: ignore
        )

        speech_1 = speech_1.to(device, non_blocking=True)
        motion_1 = motion_1.to(device, non_blocking=True)
        speech_2 = speech_2.to(device, non_blocking=True)
        motion_2 = motion_2.to(device, non_blocking=True)

        if args.do_norm:  # type: ignore
            motion_1 = normalize_tensor(motion_1, args.mean, args.std)  # type: ignore
            motion_2 = normalize_tensor(motion_2, args.mean, args.std)  # type: ignore

        # forward
        with torch.amp.autocast_mode.autocast("cuda"):
            loss = model(speech_1, motion_1, speech_2, motion_2)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss_scaler(
            loss,
            optimizer,
            clip_grad=args.grad_clip,  # type: ignore
            parameters=model.parameters(),
            update_grad=True,
        )
        optimizer.zero_grad()

        torch.cuda.synchronize()

        update_ema(ema_params, model_params, rate=args.ema_rate)  # type: ignore

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None:
            """We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)  # type: ignore
            log_writer.add_scalar("train_loss", loss_value_reduce, epoch_1000x)
            log_writer.add_scalar("lr", lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def test(
    model_without_ddp,
    ema_params,
    data_loader: Iterable,
    args,
    use_ema=True,
):
    save_folder = os.path.join(
        args.output_dir,
        "{}ariter{}-diffsteps{}-temp{}-{}cfg{}-mask_agent{}-do_norm{}-context{}-causal{}-l2r_order{}-{}".format(
            "" if "noised" not in args.test_data_path else "noised_",
            args.num_iter,
            args.num_sampling_steps,
            args.temperature,
            args.cfg_schedule,
            args.cfg,
            args.mask_history_agent,
            args.do_norm,
            args.sampling_context_time,
            args.use_causal_attn,
            args.l2r_order,
            args.test_data_path.replace("/", "_"),
        ),
    )
    if use_ema:
        save_folder = save_folder + "_ema"
    if args.test:
        save_folder = save_folder + "_test"
    print("Save to:", save_folder)
    if misc.get_rank() == 0:
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
    Path(save_folder).mkdir(parents=True, exist_ok=True)
    # switch to ema params
    if use_ema:
        model_state_dict = copy.deepcopy(model_without_ddp.state_dict())
        ema_state_dict = copy.deepcopy(model_without_ddp.state_dict())
        for i, (name, _value) in enumerate(model_without_ddp.named_parameters()):
            assert name in ema_state_dict
            ema_state_dict[name] = ema_params[i]
        print("Switch to ema")
        model_without_ddp.load_state_dict(ema_state_dict)

    model_without_ddp.eval()
    torch.set_grad_enabled(False)

    device = next(model_without_ddp.parameters()).device

    for i, (file_name, speech_1, motion_1, speech_2) in tqdm(
        enumerate(data_loader), total=len(data_loader), ncols=100
    ):
        torch.cuda.synchronize()
        start_time = time.time()

        file_name = file_name[0]
        if file_name.endswith("speaker1"):
            file_name = file_name.replace("speaker1", "speaker2")
        elif file_name.endswith("speaker2"):
            file_name = file_name.replace("speaker2", "speaker1")
        save_path = os.path.join(save_folder, "{}.npy".format(file_name))

        if os.path.exists(save_path):
            print("Skip existing", save_path)
            continue

        # generation
        s1_duration = speech_1.size(1) // model_without_ddp.speech_frequency
        s2_duration = speech_2.size(1) // model_without_ddp.speech_frequency
        m1_duration = motion_1.size(1) // model_without_ddp.motion_fps
        duration = min(s1_duration, s2_duration, m1_duration)

        num_chunks = duration // model_without_ddp.chunk_second
        assert (
            duration == num_chunks * model_without_ddp.chunk_second
        ), "Duration should be multiple of chunk_second"
        num_frames = int(model_without_ddp.chunk_second * model_without_ddp.motion_fps)

        speech_1 = speech_1.to(device, non_blocking=True)[
            :, : duration * model_without_ddp.speech_frequency
        ]
        motion_1 = motion_1.to(device, non_blocking=True)[
            :, : duration * model_without_ddp.motion_fps, :
        ]
        speech_2 = speech_2.to(device, non_blocking=True)[
            :, : duration * model_without_ddp.speech_frequency
        ]

        if args.do_norm:  # type: ignore
            motion_1 = normalize_tensor(motion_1, args.mean, args.std)  # type: ignore

        speech_1_chunks = speech_1.contiguous().view(
            -1,
            num_chunks,
            model_without_ddp.chunk_second * model_without_ddp.speech_frequency,
        )
        motion_1_chunks = motion_1.contiguous().view(
            -1, num_chunks, num_frames, motion_1.size(2)
        )
        speech_2_chunks = speech_2.contiguous().view(
            -1,
            num_chunks,
            model_without_ddp.chunk_second * model_without_ddp.speech_frequency,
        )

        bsz_idx = 0
        motion_2_pred_chunks = []
        history = None
        for chunk_idx in range(speech_1_chunks.size(1)):
            curr_motion, new_history = model_without_ddp.sample(
                speech_1_chunks[bsz_idx, chunk_idx],
                motion_1_chunks[bsz_idx, chunk_idx],
                speech_2_chunks[bsz_idx, chunk_idx],
                history,
                num_iter=args.num_iter,
                cfg=args.cfg,
                cfg_schedule=args.cfg_schedule,
                temperature=args.temperature,
                max_context_time=args.sampling_context_time,
                mask_history_agent=args.mask_history_agent,
                use_random_orders=not args.l2r_order,
            )
            if args.do_norm:  # type: ignore
                curr_motion = denormalize_tensor(curr_motion, args.mean, args.std)  # type: ignore
            motion_2_pred_chunks.append(curr_motion)
            history = new_history
        motion_2_pred = (
            torch.cat(motion_2_pred_chunks, dim=1).squeeze(0).detach().cpu().numpy()
        )

        if np.isnan(motion_2_pred).any():
            warnings.warn("NaN in pred, skip", file_name)
            continue

        np.save(save_path, motion_2_pred)

    return save_folder

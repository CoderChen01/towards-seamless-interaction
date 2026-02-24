import argparse
import copy
import datetime
import json
import os
import time
import warnings
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

import util.misc as misc
from datasets.dataset_fitting import get_fitting_datasets
from datasets.dataset_robustness_test import (
    get_test_dataset as get_test_dataset_noised,  # 保留 noised 版本
)
from datasets.dataset_test import get_test_dataset
from model import timar
from util.engine_timar import test, train_one_epoch
from util.misc import NativeScalerWithGradNormCount as NativeScaler


# -----------------------------
# statistics
# -----------------------------
def load_stat_info(stat_path: str) -> Tuple[np.ndarray, np.ndarray]:
    with open(stat_path, "r") as f:
        stat_dict = json.load(f)

    mean_exp = np.asarray(stat_dict["exp"]["mean"], dtype=np.float32)
    std_exp = np.asarray(stat_dict["exp"]["std"], dtype=np.float32)
    mean_jaw = np.asarray(stat_dict["jawpose"]["mean"], dtype=np.float32)
    std_jaw = np.asarray(stat_dict["jawpose"]["std"], dtype=np.float32)
    mean_neck = np.asarray(stat_dict["neck"]["mean"], dtype=np.float32)
    std_neck = np.asarray(stat_dict["neck"]["std"], dtype=np.float32)

    mean_all = np.concatenate([mean_exp, mean_jaw, mean_neck], axis=0)
    std_all = np.concatenate([std_exp, std_jaw, std_neck], axis=0)

    eps = np.finfo(np.float32).eps
    std_all = np.maximum(std_all, eps)

    return mean_all, std_all


def derive_noise_std_from_stats(args: argparse.Namespace) -> None:
    std_all: np.ndarray = args.std

    if getattr(args, "user_group_noise", False):
        std_exp = std_all[:50]
        std_jaw = std_all[50:53]
        std_pose = std_all[53:56]

        args.user_noise_std_exp = float(args.user_noise_alpha_exp) * float(
            np.mean(std_exp)
        )
        args.user_noise_std_jaw = float(args.user_noise_alpha_jaw) * float(
            np.mean(std_jaw)
        )
        args.user_noise_std_pose = float(args.user_noise_alpha_pose) * float(
            np.mean(std_pose)
        )
    else:
        global_std = float(np.mean(std_all))
        if float(getattr(args, "user_noise_alpha", 0.0)) > 0:
            args.user_noise_std = float(args.user_noise_alpha) * global_std


# -----------------------------
# model / optimizer / ckpt
# -----------------------------
def build_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    model = timar.__dict__[args.model](
        motion_fps=args.motion_fps,
        speech_frequency=args.speech_frequency,
        chunk_second=args.chunk_second,
        max_context_time=args.max_context_time,
        speech_snippet_embed_dim=args.speech_snippet_embed_dim,
        motion_embed_dim=args.motion_embed_dim,
        token_embed_dim=args.token_embed_dim,
        use_causal_attn=args.use_causal_attn,
        mask_ratio_min=args.mask_ratio_min,
        cond_drop_prob=args.cond_drop_prob,
        attn_dropout=args.attn_dropout,
        proj_dropout=args.proj_dropout,
        diffloss_d=args.diffloss_d,
        diffloss_w=args.diffloss_w,
        noise_schedule=args.noise_schedule,
        predict_xstart=not args.epsilon_pred,
        clip_denoised=args.clip_denoised,
        use_kl=args.use_kl,
        rescale_learned_sigmas=args.rescale_learned_sigmas,
        num_sampling_steps=args.num_sampling_steps,
        diffusion_batch_mul=args.diffusion_batch_mul,
        grad_checkpointing=args.grad_checkpointing,
        use_diffloss=not args.use_mseloss,
        use_encoder_only=args.use_encoder_only,
    )
    model.to(device)
    return model


def print_model_info(model: torch.nn.Module) -> None:
    print("Model = %s" % str(model))
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all_params = sum(p.numel() for p in model.parameters())
    n_wav2vec_params = sum(p.numel() for p in model.speech_encoder.wav2vec.parameters())
    print("Number of trainable parameters: {}M".format(n_params / 1e6))
    print("Number of wav2vec parameters: {}M".format(n_wav2vec_params / 1e6))
    print(
        "Number of all parameters: {}M".format((n_all_params + n_wav2vec_params) / 1e6)
    )


def build_optimizer_and_scaler(
    args: argparse.Namespace, model_without_ddp: torch.nn.Module
) -> Tuple[torch.optim.Optimizer, NativeScaler]:
    param_groups = misc.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()
    return optimizer, loss_scaler


def _resolve_ckpt_path(resume: str) -> str:
    if os.path.isfile(resume):
        return resume
    candidate = os.path.join(resume, "checkpoint-best.pth")
    if os.path.exists(candidate):
        return candidate
    raise ValueError(f"Checkpoint `{resume}` don't exist!")


def load_checkpoint_if_needed(
    args: argparse.Namespace,
    model_without_ddp: torch.nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer],
    loss_scaler: Optional[NativeScaler],
) -> Tuple[list, list]:
    model_params = list(model_without_ddp.parameters())

    if not args.resume:
        ema_params = copy.deepcopy(model_params)
        print("Training from scratch")
        return model_params, ema_params

    ckpt_path = _resolve_ckpt_path(args.resume)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    model_without_ddp.load_state_dict(checkpoint["model"])

    ema_state_dict = checkpoint["model_ema"]
    ema_params = [
        ema_state_dict[name].to(device)
        for name, _ in model_without_ddp.named_parameters()
    ]

    print("Resume checkpoint %s" % ckpt_path)
    if "epoch" in checkpoint:
        print(f"Resume CKT Epoch: {checkpoint['epoch']}")

    if optimizer is not None and loss_scaler is not None:
        if "optimizer" in checkpoint and "epoch" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            args.start_epoch = int(checkpoint["epoch"]) + 1
            if "scaler" in checkpoint:
                loss_scaler.load_state_dict(checkpoint["scaler"])
            print("With optim & sched!")

    del checkpoint
    return model_params, ema_params


# -----------------------------
# dataloaders
# -----------------------------
def build_train_val_loaders(args: argparse.Namespace, num_tasks: int, global_rank: int):
    dataset_train, dataset_val = get_fitting_datasets(args)

    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    sampler_val = torch.utils.data.DistributedSampler(
        dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    print("Sampler_train = %s" % str(sampler_train))
    print("Sampler_val = %s" % str(sampler_val))
    return data_loader_train, data_loader_val


def build_test_loader(args: argparse.Namespace, num_tasks: int, global_rank: int):
    if "noised" in str(args.test_data_path):
        dataset_test = get_test_dataset_noised(args)
    else:
        dataset_test = get_test_dataset(args)

    sampler_test = torch.utils.data.DistributedSampler(
        dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=False
    )
    print("Sampler_test = %s" % str(sampler_test))

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        sampler=sampler_test,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    return data_loader_test


# -----------------------------
# metrics helpers (sanity)
# -----------------------------
def _infer_gt_motion_path(test_data_path: str) -> str:
    if "noised" not in test_data_path:
        return test_data_path
    return str(Path(test_data_path).parent / test_data_path.split("_")[-1])


def _count_motion_files(root: str) -> int:
    p = Path(root)
    if not p.exists():
        return 0
    exts = {".npy", ".npz", ".pt", ".pth", ".json"}
    cnt = 0
    for fp in p.rglob("*"):
        if fp.is_file() and fp.suffix.lower() in exts:
            cnt += 1
    return cnt


def _quick_check_non_trivial_dir(save_dir: str) -> None:
    n_files = _count_motion_files(save_dir)
    if n_files == 0:
        raise RuntimeError(
            f"[test] No motion files found under pred_motion_path: {save_dir}"
        )


# -----------------------------
# args
# -----------------------------
def get_args_parser():
    parser = argparse.ArgumentParser("TIMAR main script", add_help=False)

    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--epochs", default=400, type=int)

    # Model parameters
    parser.add_argument("--model", default="timar_large", type=str, metavar="MODEL")
    parser.add_argument("--motion_fps", default=25, type=int)
    parser.add_argument("--speech_frequency", default=16_000, type=int)
    parser.add_argument("--chunk_second", default=1, type=int)
    parser.add_argument("--max_context_time", default=8, type=int)
    parser.add_argument("--speech_snippet_embed_dim", default=1024, type=int)
    parser.add_argument("--motion_embed_dim", default=56, type=int)
    parser.add_argument("--token_embed_dim", default=1024, type=int)
    parser.add_argument("--use_causal_attn", action="store_true")
    parser.add_argument("--sampling_context_time", default=None, type=int)
    parser.add_argument("--l2r_order", action="store_true")
    parser.add_argument("--use_encoder_only", action="store_true")
    parser.add_argument("--mask_history_agent", action="store_true")

    # Generation parameters
    parser.add_argument("--num_iter", default=1, type=int)
    parser.add_argument("--cfg", default=1.0, type=float)
    parser.add_argument("--cfg_schedule", default="linear", type=str)
    parser.add_argument("--cond_drop_prob", default=0.1, type=float)
    parser.add_argument("--eval_freq", type=int, default=40)
    parser.add_argument("--save_last_freq", type=int, default=5)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--eval_bsz", type=int, default=64)

    # Optimizer parameters
    parser.add_argument("--weight_decay", type=float, default=0.02)
    parser.add_argument("--grad_checkpointing", action="store_true")
    parser.add_argument("--lr", type=float, default=None, metavar="LR")
    parser.add_argument("--blr", type=float, default=1e-4, metavar="LR")
    parser.add_argument("--min_lr", type=float, default=0.0, metavar="LR")
    parser.add_argument("--lr_schedule", type=str, default="constant")
    parser.add_argument("--warmup_epochs", type=int, default=100, metavar="N")
    parser.add_argument("--ema_rate", default=0.9999, type=float)

    # MAR params
    parser.add_argument("--mask_ratio_min", type=float, default=0.7)
    parser.add_argument("--grad_clip", type=float, default=3.0)
    parser.add_argument("--attn_dropout", type=float, default=0.1)
    parser.add_argument("--proj_dropout", type=float, default=0.1)

    # Diffusion Loss params
    parser.add_argument("--use_mseloss", action="store_true")
    parser.add_argument("--diffloss_d", type=int, default=12)
    parser.add_argument("--diffloss_w", type=int, default=1536)
    parser.add_argument("--num_sampling_steps", type=str, default="100")
    parser.add_argument("--diffusion_batch_mul", type=int, default=1)
    parser.add_argument("--temperature", default=1.0, type=float)
    parser.add_argument("--noise_schedule", type=str, default="cosine")
    parser.add_argument("--clip_denoised", action="store_true")
    parser.add_argument("--epsilon_pred", action="store_true")
    parser.add_argument("--use_kl", action="store_true")
    parser.add_argument("--rescale_learned_sigmas", action="store_true")

    # Dataset parameters
    parser.add_argument("--train_data_path", type=str, default="./data/train/")
    parser.add_argument("--val_data_path", type=str, default="./data/ood/")
    parser.add_argument("--stat_path", type=str, default="./data/stat_flame.json")
    parser.add_argument("--do_norm", action="store_true")
    parser.add_argument("--test_data_path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=6666)
    parser.add_argument("--scale", type=str, default="large")
    parser.add_argument("--output_dir", default="./output_dir")
    parser.add_argument("--log_dir", default="./output_dir")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", default="")

    parser.add_argument("--start_epoch", default=0, type=int, metavar="N")
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument("--pin_mem", action="store_true")
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://")

    return parser


# -----------------------------
# main
# -----------------------------
def main(args: argparse.Namespace) -> None:
    misc.init_distributed_mode(args)

    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(", ", ",\n"))

    device = torch.device(args.device)

    seed = int(args.seed) + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    log_writer: Optional[SummaryWriter] = None
    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)

    model = build_model(args, device)
    model_without_ddp = model

    if misc.is_main_process():
        print_model_info(model_without_ddp)

    # load stats
    mean_all, std_all = load_stat_info(args.stat_path)
    args.mean = mean_all
    args.std = std_all
    derive_noise_std_from_stats(args)

    # -----------------------------
    # TEST
    # -----------------------------
    if args.test:
        if not args.test_data_path:
            raise ValueError("--test is set but --test_data_path is None")

        _, ema_params = load_checkpoint_if_needed(
            args=args,
            model_without_ddp=model_without_ddp,
            device=device,
            optimizer=None,
            loss_scaler=None,
        )

        torch.cuda.empty_cache()

        data_loader_test = build_test_loader(args, num_tasks, global_rank)
        save_dir = test(
            model_without_ddp,
            ema_params,
            data_loader_test,
            args,
            use_ema=True,
        )

        if args.distributed:
            torch.distributed.barrier()

        if misc.is_main_process():
            _quick_check_non_trivial_dir(save_dir)

            from calc_metric import calc_metrics

            gt_path = _infer_gt_motion_path(str(args.test_data_path))

            metric_args = argparse.Namespace(
                pred_motion_path=save_dir,
                gt_motion_path=gt_path,
                output_path=None,
                is_serial_computing=False,
            )

            calc_metrics(metric_args)

        return

    # -----------------------------
    # TRAIN
    # -----------------------------
    data_loader_train, _data_loader_val = build_train_val_loaders(
        args, num_tasks, global_rank
    )

    eff_batch_size = int(args.batch_size) * misc.get_world_size()
    if args.lr is None:
        args.lr = float(args.blr) * eff_batch_size / 256.0

    print("base lr: %.2e" % (args.lr * 256.0 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module

    optimizer, loss_scaler = build_optimizer_and_scaler(args, model_without_ddp)

    model_params, ema_params = load_checkpoint_if_needed(
        args=args,
        model_without_ddp=model_without_ddp,
        device=device,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
    )

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    best_loss = float("inf")

    for epoch in range(int(args.start_epoch), int(args.epochs)):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)  # type: ignore[attr-defined]

        rv = train_one_epoch(
            model=model,
            model_params=model_params,
            ema_params=ema_params,
            data_loader=data_loader_train,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            loss_scaler=loss_scaler,
            log_writer=log_writer,
            args=args,
        )

        curr_training_loss = float(rv["loss"])

        if epoch % int(args.save_last_freq) == 0 or (epoch + 1) == int(args.epochs):
            misc.save_model(
                args=args,
                model=model,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                loss_scaler=loss_scaler,
                epoch=epoch,
                ema_params=ema_params,
                epoch_name="last",
            )
            if curr_training_loss < best_loss:
                best_loss = curr_training_loss
                misc.save_model(
                    args=args,
                    model=model,
                    model_without_ddp=model_without_ddp,
                    optimizer=optimizer,
                    loss_scaler=loss_scaler,
                    epoch=epoch,
                    ema_params=ema_params,
                    epoch_name="best",
                )

        if misc.is_main_process() and log_writer is not None:
            log_writer.flush()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    args.log_dir = args.output_dir
    main(args)

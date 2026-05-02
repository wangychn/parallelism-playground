from dataclasses import dataclass, asdict
from typing import Literal
import argparse
import torch


def default_dtype():
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return "bfloat16"
    return "float16"


DType = Literal["float32", "bfloat16", "float16"]
Backend = Literal["nccl", "gloo"]

@dataclass(slots=True)
class TrainArgs:
    out_dir: str
    eval_interval: int
    eval_iters: int
    log_interval: int
    eval_only: bool
    always_save_checkpoint: bool

    wandb_log: bool
    wandb_project: str
    wandb_run_name: str

    dataset: str
    gradient_accumulation_steps: int
    batch_size: int
    block_size: int

    n_layer: int
    n_head: int
    n_embd: int
    dropout: float
    bias: bool

    learning_rate: float
    max_iters: int
    weight_decay: float
    beta1: float
    beta2: float
    grad_clip: float

    decay_lr: bool
    warmup_iters: int
    lr_decay_iters: int
    min_lr: float

    distributed: bool
    backend: Backend

    device: str
    dtype: DType
    compile: bool
    seed: int

    use_moe: bool
    num_experts: int
    num_experts_per_tok: int


def build_parser():
    parser = argparse.ArgumentParser(description="Training arguments for distributed MoE experiments.")

    io_group = parser.add_argument_group("I/O and Evaluation")
    io_group.add_argument("--out-dir", dest="out_dir", type=str, default="out")
    io_group.add_argument("--eval-interval", dest="eval_interval", type=int, default=2000)
    io_group.add_argument("--eval-iters", dest="eval_iters", type=int, default=200)
    io_group.add_argument("--log-interval", dest="log_interval", type=int, default=1)
    io_group.add_argument("--eval-only", action=argparse.BooleanOptionalAction, default=False)
    # Optional: keep checkpoint-save behavior configurable while the trainer is still being refactored.
    io_group.add_argument("--always-save-checkpoint", dest="always_save_checkpoint", action=argparse.BooleanOptionalAction, default=True)

    wandb_group = parser.add_argument_group("WandB")
    # Optional: useful later once experiment tracking is wired back in.
    wandb_group.add_argument("--wandb-log", dest="wandb_log", action=argparse.BooleanOptionalAction, default=False)
    wandb_group.add_argument("--wandb-project", dest="wandb_project", type=str, default="owt")
    wandb_group.add_argument("--wandb-run-name", dest="wandb_run_name", type=str, default="gpt2")

    data_group = parser.add_argument_group("Data")
    data_group.add_argument("--dataset", type=str, default="openwebtext")
    data_group.add_argument("--gradient-accumulation-steps", dest="gradient_accumulation_steps", type=int, default=40)
    data_group.add_argument("--batch-size", dest="batch_size", type=int, default=12)
    data_group.add_argument("--block-size", dest="block_size", type=int, default=1024)

    model_group = parser.add_argument_group("Model")
    model_group.add_argument("--n-layer", dest="n_layer", type=int, default=12)
    model_group.add_argument("--n-head", dest="n_head", type=int, default=12)
    model_group.add_argument("--n-embd", dest="n_embd", type=int, default=768)
    model_group.add_argument("--dropout", type=float, default=0.0)
    model_group.add_argument("--bias", action=argparse.BooleanOptionalAction, default=False)

    optim_group = parser.add_argument_group("Optimizer")
    optim_group.add_argument("--learning-rate", dest="learning_rate", type=float, default=6e-4)
    optim_group.add_argument("--max-iters", dest="max_iters", type=int, default=600000)
    optim_group.add_argument("--weight-decay", dest="weight_decay", type=float, default=1e-1)
    optim_group.add_argument("--beta1", type=float, default=0.9)
    optim_group.add_argument("--beta2", type=float, default=0.95)
    # Optional: clipping is easy to add later without blocking the main training refactor.
    optim_group.add_argument("--grad-clip", dest="grad_clip", type=float, default=1.0)

    lr_group = parser.add_argument_group("Learning Rate Schedule")
    lr_group.add_argument("--decay-lr", dest="decay_lr", action=argparse.BooleanOptionalAction, default=True)
    lr_group.add_argument("--warmup-iters", dest="warmup_iters", type=int, default=2000)
    lr_group.add_argument("--lr-decay-iters", dest="lr_decay_iters", type=int, default=600000)
    lr_group.add_argument("--min-lr", dest="min_lr", type=float, default=6e-5)

    dist_group = parser.add_argument_group("Distributed")
    dist_group.add_argument("--distributed", action=argparse.BooleanOptionalAction, default=False)
    dist_group.add_argument("--backend", type=str, default="nccl")

    system_group = parser.add_argument_group("System")
    system_group.add_argument("--device", type=str, default="cuda")
    system_group.add_argument("--dtype", type=str, default=default_dtype(), choices=["float32", "bfloat16", "float16"])
    system_group.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    system_group.add_argument("--seed", type=int, default=1337)

    moe_group = parser.add_argument_group("MoE")
    moe_group.add_argument("--use-moe", dest="use_moe", action=argparse.BooleanOptionalAction, default=True)
    moe_group.add_argument("--num-experts", dest="num_experts", type=int, default=10)
    moe_group.add_argument("--num-experts-per-tok", dest="num_experts_per_tok", type=int, default=2)

    return parser


def parse_args(argv=None) -> TrainArgs:
    ns = build_parser().parse_args(argv)
    return TrainArgs(**vars(ns))


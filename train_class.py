import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
import json
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from dataset.dataset import load_hf_dataset

from train_args import TrainArgs
from model.model import GPTConfig, GPT

from train_args import parse_args
from transformers import AutoTokenizer

from rich.console import Group
from rich.console import Console
from rich.text import Text
from rich.live import Live
from rich.progress import (
        Progress,
        TextColumn,
        BarColumn,
        TimeRemainingColumn,
        TaskProgressColumn,
)


class Trainer:

    def __init__(self, args : TrainArgs):
        self.args = args

        self.setup_distributed()
        self.setup_runtime()

        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.vocab_size = tokenizer.vocab_size

        self.iter_num = 0
        self.best_val_loss = 1e9
        self.best_iter = 0
        self.total_training_time_ms = 0.0
        self.time_remaining_ms = 0.0
        self.total_time_est_ms = 0.0

        # key lr variables
        self.max_lr = args.learning_rate
        self.warmup_iters = args.warmup_iters
        self.lr_decay_iters = args.lr_decay_iters
        self.min_lr = args.min_lr


        train_data, val_data = load_hf_dataset(args.dataset)
        self.train_loader, self.val_loader = self.build_train_val_loaders(train_data, val_data)

        # save full configuration used for training
        config_json = {self.args}
        with open(self.args.out_dir + "/full_config.json", "w") as configuration_file:
            json.dump(config_json, configuration_file, indent=4)
        with open(self.args.out_dir + "/best_val_loss_and_iter.txt", 'w') as file:
            print("resetting best val loss file")

        # get the relevant GPT configs and put into GPT config object
        valid_keys = GPTConfig.__init__.__code__.co_varnames
        filtered_args = {k: v for k, v in self.args.items() if k in valid_keys}
        gptconf = GPTConfig(**filtered_args)

        # initialize model
        self.model = GPT(gptconf)
        self.model.to(self.device)

        # make the optimizer
        self.optimizer = self.model.configure_optimizers(
            args.weight_decay, 
            args.learning_rate, 
            (args.beta1, args.beta2), 
            args.device_type)

        # get model size
        self.model.num_param = self.model.get_num_params(non_embedding=False)

        if self.args.compile:
            print("compiling the model... (takes a ~minute)")
            self.model = torch.compile(self.model)
        if self.ddp:
            # figure this part out
            self.model = DDP(self.model, device_ids=[self.ddp_local_rank])

    def train(self):
        evaluations_remaining = (self.args.max_iters - self.iter_num) // self.args.eval_interval + 1
        iter_latency_avg = 0.0

        self.console = Console()
        progress = Progress(
            TextColumn("[bold white]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(compact=False),
            TextColumn("-- [bold dark_cyan]BestIter:[/bold dark_cyan]{task.fields[best_iter]}"),
            TextColumn("[bold dark_cyan]BestValLoss:[/bold dark_cyan]{task.fields[best_val_loss]}"),
            TextColumn("-- [bold purple3]ETA:[/bold purple3]{task.fields[eta]}"),
            TextColumn("[bold purple3]Remaining:[/bold purple3]{task.fields[hour]}h{task.fields[min]}m"),
            TextColumn("[bold purple3]TotalEst:[/bold purple3]{task.fields[total_hour]}h{task.fields[total_min]}m"),
            TextColumn("-- [bold dark_magenta]iter_latency:[/bold dark_magenta]{task.fields[iter_latency]}ms"),
            TextColumn("[bold dark_magenta]peak_gpu_mb:[/bold dark_magenta]{task.fields[peak_gpu_mb]}MB"),
            TextColumn("-- [bold dark_cyan]loss:[/bold dark_cyan]{task.fields[loss]}"),
            TextColumn("[bold dark_cyan]lr:[/bold dark_cyan]{task.fields[lr]}"),
            console=self.console,
        )

        cli_text = Text(f"CLI: {' '.join(sys.argv)}", style="chartreuse1")

        t_start = time.time()
        t0 = t_start

        with Live(Group(progress.get_renderable(), cli_text), console=self.console, refresh_per_second=10) as live:
            task_id = progress.add_task(
                "[green]Training...",
                total=(self.args.max_iters - self.iter_num) + evaluations_remaining * self.args.eval_iters,
                eta="waiting...",
                total_hour="--",
                total_min="--",
                hour="--",
                min="--",
                best_val_loss=f"{self.best_val_loss:.3f}",
                best_iter=f"{self.best_iter}",
                iter_latency=f"{iter_latency_avg:.1f}",
                peak_gpu_mb="--",
                loss="--",
                lr="--",
            )
        
            # get batch
            x, y = next(iter(self.train_loader))
            local_iter_num = 0
            
            # ========== main training loop ===========

            while True:

                cur_lr = self.get_lr() if self.decay_lr else self.max_lr
                

                # increment crucial numbers
                self.iter_num += 1

                for param_group in self.optimizer:
                    param_group['lr'] = cur_lr

                pass 

    def get_lr(self):
        # linear warmup
        if self.iter_num < self.warmup_iters:
            return self.max_lr * self.iter_num / self.warmup_iters
            
        if self.iter_num > self.lr_decay_iters:
            return self.min_lr
        
        # cosine decay for lr 
        # https://keras.io/api/optimizers/learning_rate_schedules/cosine_decay/
        decay_ratio = (self.iter_num - self.warmup_iters) / (self.lr_decay_iters - self.warmup_iters)
        assert 0 <= decay_ratio <= 1

        lr_coef = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        assert 0 <= lr_coef <= 1

        return self.min_lr + lr_coef * (self.max_lr - self.min_lr)

    def setup_distributed(self):
        # torchrun --nproc_per_node=4 train.py
        # Process 0: RANK=0, LOCAL_RANK=0, WORLD_SIZE=4
        # Process 1: RANK=1, LOCAL_RANK=1, WORLD_SIZE=4
        # Process 2: RANK=2, LOCAL_RANK=2, WORLD_SIZE=4
        # Process 3: RANK=3, LOCAL_RANK=3, WORLD_SIZE=4

        # NCCL — NVIDIA's library, optimized for GPU-to-GPU transfers over NVLink or InfiniBand. Always use this for GPU training
        
        # distributed data parallelism
        self.ddp = int(os.environ.get("RANK", -1)) != -1
        if self.ddp:
            # setup handshake, so processes have communication channels
            init_process_group(backend=self.args.backend)
            # RANK: global process ID across all machines. If you have 2 machines × 4 GPUs, ranks are 0–7
            self.rank = int(os.environ["RANK"])
            # LOCAL_RANK: process ID within a single machine. Used to assign which GPU this process owns.
            self.local_rank = int(os.environ["LOCAL_RANK"])
            # WORLD_SIZE: total number of processes
            self.world_size = int(os.environ["WORLD_SIZE"])

            self.device = f"cuda:{self.local_rank}"

            torch.cuda.set_device(self.device)
            self.master_process = self.rank == 0
            # each process gets a different random seed so dropout masks and any stochasticity differs across GPUs
            self.seed_offset = self.rank

            # Gradient Accumulation: simulate higher batch size
            #   do multiple forward/backward passes and accumulate gradients before doing one optimizer step
            # Accumulation is distributed, so needs to divide evently amongst GPUs
            assert self.args.gradient_accumulation_steps % self.world_size == 0
            self.grad_accum_steps = self.args.gradient_accumulation_steps // self.world_size
        else:
            self.rank = 0
            self.local_rank = 0
            self.world_size = 1
            self.device = self.args.device
            self.master_process = True
            self.seed_offset = 0
            self.grad_accum_steps = self.args.gradient_accumulation_steps

        self.device_type = 'cuda' if 'cuda' in self.device else 'cpu'
        self.tokens_per_iter = (
            self.grad_accum_steps * self.world_size * self.args.batch_size * self.args.block_size
        )

    def setup_runtime(self):
        """
        sets up the backend runtime based on varaiables in setup distributed
        """
        
        if self.master_process:
            os.makedirs(self.out_dir, exist_ok=True)
        torch.manual_seed(1337 + self.seed_offset)
        torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

        ctx = nullcontext()
        if self.device_type == 'gpu':
            # note: float16 data type will automatically use a GradScaler
            # https://docs.pytorch.org/docs/2.11/amp.html -> AMP TORCH DOCS
            ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[self.dtype]
            ctx = torch.amp.autocast(device_type=self.device_type, dtype=ptdtype)


    def cleanup(self):
        if self.ddp:
            destroy_process_group()

            

def main():
    args = parse_args()
    print(args)
    # trainer = Trainer(args)
    # trainer.train()

if __name__ == '__main__':
    main()
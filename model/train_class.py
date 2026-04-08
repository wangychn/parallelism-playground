import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT

from train_args import parse_args

class Trainer:

    def __init__(self, args):
        self.args = args

        self.setup_distributed()
        self.setup_runtime()

        iter_num = 0
        best_val_loss = 1e9

        

    def train():
        pass

    def setup_distributed(self):
        """
        various inits
        """
         
        self.ddp = int(os.environ.get("RANK", -1)) != -1
        if self.ddp:
            init_process_group(backend=self.args.backend)
            self.rank = int(os.environ["RANK"])
            self.local_rank = int(os.environ["LOCAL_RANK"])
            self.world_size = int(os.environ["WORLD_SIZE"])
            self.device = f"cuda:{self.local_rank}"
            torch.cuda.set_device(self.device)
            self.master_process = self.rank == 0
            self.seed_offset = self.rank

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
        device_type = 'cuda' if 'cuda' in self.device else 'cpu' # for later use in torch.autocast
        # note: float16 data type will automatically use a GradScaler
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[self.dtype]
        ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    def cleanup(self):
        if self.ddp:
            destroy_process_group()

            


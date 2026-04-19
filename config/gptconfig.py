import torch

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True

    # moe config
    use_moe: bool = False
    num_experts: int = 8
    num_experts_per_tok: int = 2

from dataclasses import dataclass

import torch
import torch.nn as nn

from model import MLP


@dataclass
class Config:
    n_embd: int = 16
    bias: bool = True
    dropout: float = 0.0
    num_experts: int = 4
    num_experts_per_tok: int = 2


class MoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.experts = nn.ModuleList([MLP(config) for _ in range(config.num_experts)])
        self.gate = nn.Linear(config.n_embd, config.num_experts, bias=False)
        self.num_experts_per_tok = config.num_experts_per_tok

    def forward(self, x):
        def show(name, value):
            shape = tuple(value.shape) if hasattr(value, "shape") else tuple(value)
            print(f"{name:20} {shape}")

        def show_head(name, tensor, rows=2):
            if tensor.numel() == 0:
                print(f"{name:20} []")
                return
            head = tensor[:rows].detach().cpu()
            print(f"{name:20} {head}")

        show("x", x)
        orig_shape = x.shape
        show("orig_shape", orig_shape)

        gate = self.gate.weight.t()
        show("gate", gate)

        x = x.view(-1, x.shape[-1]); show("x", x)
        scores = self.gate(x); show("scores", scores)
        expert_weights, expert_indices = torch.topk(scores, self.num_experts_per_tok, dim=-1)
        expert_weights = expert_weights.softmax(dim=-1)
        show("expert_weights", expert_weights)
        show("expert_indices", expert_indices)
        flat_expert_indices = expert_indices.view(-1); show("flat_expert_indices", flat_expert_indices)
        x = x.repeat_interleave(self.num_experts_per_tok, dim=0); show("x", x)
        y = torch.empty_like(x)
        show("y", y)
        for i, expert in enumerate(self.experts):
            mask = flat_expert_indices == i
            routed_x = x[mask]
            show(f"expert_{i}_x", routed_x)
            show_head(f"expert_{i}_x_head", routed_x)
            out = expert(routed_x)
            show(f"expert_{i}_out", out)
            show_head(f"expert_{i}_out_head", out)
            y[mask] = out
        y = (y.view(*expert_weights.shape, -1) * expert_weights.unsqueeze(-1)).sum(dim=1)
        show("y", y)
        y = y.view(*orig_shape)
        show("final_y", y)
        return y


if __name__ == "__main__":
    cfg = Config()
    MoE(cfg)(torch.randn(2, 3, cfg.n_embd))

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
        show = lambda name, t: print(f"{name:20} {tuple(t.shape)}")
        show("x", x)
        x = x.view(-1, x.shape[-1]); show("flat_x", x)
        scores = self.gate(x); show("scores", scores)
        ew, ei = torch.topk(scores, self.num_experts_per_tok, dim=-1)
        ew = ew.softmax(dim=-1); show("expert_weights", ew); show("expert_indices", ei)
        flat_ei = ei.view(-1); show("flat_expert_idx", flat_ei)
        x = x.repeat_interleave(self.num_experts_per_tok, dim=0); show("routed_x", x)
        y = torch.empty_like(x)
        for i, expert in enumerate(self.experts):
            mask = flat_ei == i
            out = expert(x[mask])
            show(f"expert_{i}_out", out)
            y[mask] = out
        y = (y.view(*ew.shape, -1) * ew.unsqueeze(-1)).sum(dim=1)
        show("combined_y", y)
        return y


if __name__ == "__main__":
    cfg = Config()
    out = MoE(cfg)(torch.randn(2, 3, cfg.n_embd))
    print(f"{'final_y':20} {tuple(out.shape)}")

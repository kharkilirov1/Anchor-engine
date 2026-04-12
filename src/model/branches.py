import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.config import ModelConfig


class BranchHead(nn.Module):
    """Single branch: projects hidden state to logits with its own temperature."""

    def __init__(self, d_model: int, vocab_size: int, temperature: float = 1.0):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size, bias=False)
        self.temperature = temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x) / self.temperature


class BranchRouter(nn.Module):
    """Generates multiple hypotheses via parallel branch heads.
    Computes diversity loss to prevent branch collapse.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.diversity_target = cfg.branch_diversity_target
        temps = [
            0.8 + 0.4 * i / max(cfg.n_branches - 1, 1)
            for i in range(cfg.n_branches)
        ]
        self.branches = nn.ModuleList([
            BranchHead(cfg.d_model, cfg.vocab_size, temperature=t) for t in temps
        ])
        self.branch_offsets = nn.Parameter(torch.randn(cfg.n_branches, cfg.d_model) * 0.02)

    @staticmethod
    def _js_divergence(pi: torch.Tensor, pj: torch.Tensor) -> torch.Tensor:
        m = 0.5 * (pi + pj)
        log_m = torch.log(m.clamp_min(1e-8))
        js = 0.5 * (
            (pi * (torch.log(pi.clamp_min(1e-8)) - log_m)).sum(dim=-1)
            + (pj * (torch.log(pj.clamp_min(1e-8)) - log_m)).sum(dim=-1)
        )
        return js.mean()

    def forward(self, x: torch.Tensor) -> dict:
        branch_logits = []
        for idx, branch in enumerate(self.branches):
            branch_input = x + self.branch_offsets[idx].view(1, 1, -1)
            branch_logits.append(branch(branch_input))

        diversity_loss = torch.tensor(0.0, device=x.device)
        n_pairs = 0
        for i in range(len(branch_logits)):
            for j in range(i + 1, len(branch_logits)):
                pi = F.softmax(branch_logits[i], dim=-1)
                pj = F.softmax(branch_logits[j], dim=-1)
                js_div = self._js_divergence(pi, pj)
                diversity_loss = diversity_loss + F.relu(self.diversity_target - js_div)
                n_pairs += 1
        if n_pairs > 0:
            diversity_loss = diversity_loss / n_pairs

        avg_logits = torch.stack(branch_logits).mean(dim=0)

        return {
            "logits": avg_logits,
            "branch_logits": branch_logits,
            "diversity_loss": diversity_loss,
        }

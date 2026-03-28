import torch
import torch.nn as nn
from src.model.config import ModelConfig


class PlasticLayer(nn.Module):
    """Short-term plastic adapter on top of stable backbone.
    - Gradient-based update with small LR
    - L2 regularization toward initial state
    - Exponential decay of adapter weights
    Analogy: hippocampal fast learning (complementary learning systems).
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.adapter = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.plastic_hidden),
            nn.GELU(),
            nn.Linear(cfg.plastic_hidden, cfg.d_model),
        )
        self.initial_state: dict[str, torch.Tensor] = {}
        self._save_initial_state()

    def _save_initial_state(self):
        self.initial_state = {
            n: p.data.clone() for n, p in self.adapter.named_parameters()
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.adapter(x)

    def adapt_step(self, x: torch.Tensor, lr: float | None = None):
        """One gradient step of self-supervised adaptation."""
        if lr is None:
            lr = self.cfg.plastic_lr

        self.adapter.train()
        adapted = self.adapter(x.detach())

        loss = (adapted**2).mean()

        l2_loss = torch.tensor(0.0, device=x.device)
        for n, p in self.adapter.named_parameters():
            l2_loss = l2_loss + ((p - self.initial_state[n].to(p.device)) ** 2).mean()
        loss = loss + self.cfg.plastic_l2_weight * l2_loss

        loss.backward()
        with torch.no_grad():
            for p in self.adapter.parameters():
                if p.grad is not None:
                    p.add_(p.grad, alpha=-lr)
                    p.grad.zero_()

    def apply_decay(self, decay_rate: float | None = None):
        """Exponential decay toward initial state."""
        if decay_rate is None:
            decay_rate = self.cfg.plastic_decay
        with torch.no_grad():
            for n, p in self.adapter.named_parameters():
                init = self.initial_state[n].to(p.device)
                p.data.mul_(decay_rate).add_(init, alpha=1.0 - decay_rate)

    def reset(self):
        """Reset adapter to initial state."""
        with torch.no_grad():
            for n, p in self.adapter.named_parameters():
                p.data.copy_(self.initial_state[n])

"""Equilibrium Signal — unified routing/confidence/plasticity trigger.

Computes deviation of activations from running mean (accumulated during training).
Near-zero overhead: reuses LayerNorm statistics.

ED(x) = || (x - mu_running) / sigma_running ||

Small ED → forward pass (confident)
Medium ED → branching (uncertain, explore alternatives)
Large ED → backward pass (re-process through earlier layers)
Critical ED → plastic activation (adapt to new context)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class EquilibriumSignal(nn.Module):
    def __init__(self, d_model: int, momentum: float = 0.1, warmup_steps: int = 50):
        super().__init__()
        self.d_model = d_model
        self.momentum = momentum
        self.warmup_steps = warmup_steps
        # Running statistics (like BatchNorm)
        self.register_buffer("running_mean", torch.zeros(d_model))
        self.register_buffer("running_var", torch.ones(d_model))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))

    @property
    def is_warming_up(self) -> bool:
        return self.num_batches_tracked.item() < self.warmup_steps

    def forward(self, x: torch.Tensor) -> dict:
        """Compute equilibrium deviation for each token.

        Args:
            x: [B, T, D] — layer output activations

        Returns:
            dict with:
                ed: [B, T] — equilibrium deviation per token
                x: [B, T, D] — unchanged input (pass-through)
                warming_up: bool — True if still in warmup phase
        """
        # Update running stats during training
        if self.training:
            with torch.no_grad():
                batch_mean = x.detach().mean(dim=(0, 1))  # [D]
                batch_var = x.detach().var(dim=(0, 1))  # [D]
                self.running_mean.mul_(1 - self.momentum).add_(batch_mean, alpha=self.momentum)
                self.running_var.mul_(1 - self.momentum).add_(batch_var, alpha=self.momentum)
                self.num_batches_tracked += 1

        # Compute ED: normalized distance from running mean
        # [B, T, D]
        normalized = (x - self.running_mean) / (self.running_var.sqrt() + 1e-8)
        # [B, T] — L2 norm over feature dim, normalized by sqrt(d_model)
        ed = normalized.norm(dim=-1) / (self.d_model ** 0.5)

        return {"ed": ed, "x": x, "warming_up": self.is_warming_up}


class RoutingDecision(nn.Module):
    """Converts equilibrium deviation into routing decisions.

    Buckets are calibrated from running ED quantiles so Stage B does not collapse
    to a single route just because the absolute ED scale shifted.
    Small learnable offsets allow training to nudge boundaries around the
    quantile-derived defaults.
    """

    def __init__(
        self,
        init_thresholds: tuple[float, float, float] = (0.75, 1.0, 1.35),
        target_fractions: tuple[float, float, float, float] = (0.55, 0.25, 0.15, 0.05),
        threshold_momentum: float = 0.2,
        temperature: float = 8.0,
        offset_scale: float = 0.2,
    ):
        super().__init__()
        fractions = torch.tensor(target_fractions, dtype=torch.float32)
        fractions = fractions / fractions.sum().clamp_min(1e-8)
        self.register_buffer("target_cdf", fractions.cumsum(dim=0)[:-1])
        self.register_buffer("running_thresholds", torch.tensor(init_thresholds, dtype=torch.float32))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))
        self.threshold_momentum = threshold_momentum
        self.temperature = temperature
        self.offset_scale = offset_scale
        self.threshold_offsets = nn.Parameter(torch.zeros(3))

    def _batch_thresholds(self, ed: torch.Tensor) -> torch.Tensor:
        flat = ed.detach().reshape(-1)
        if flat.numel() == 0:
            return self.running_thresholds
        return torch.quantile(flat, self.target_cdf.to(device=ed.device, dtype=flat.dtype))

    def _ordered_thresholds(self, base: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        offsets = self.offset_scale * torch.tanh(self.threshold_offsets).to(base.device, base.dtype)
        raw = base + offsets
        min_gap = torch.tensor(1e-3, device=base.device, dtype=base.dtype)
        t1 = raw[0]
        t2 = torch.maximum(raw[1], t1 + min_gap)
        t3 = torch.maximum(raw[2], t2 + min_gap)
        return t1, t2, t3

    @property
    def theta1(self) -> torch.Tensor:
        thresholds = self._ordered_thresholds(self.running_thresholds)
        return thresholds[0]

    @property
    def theta2(self) -> torch.Tensor:
        thresholds = self._ordered_thresholds(self.running_thresholds)
        return thresholds[1]

    @property
    def theta3(self) -> torch.Tensor:
        thresholds = self._ordered_thresholds(self.running_thresholds)
        return thresholds[2]

    def forward(self, ed: torch.Tensor) -> dict:
        """Classify each token into routing buckets.

        Args:
            ed: [B, T] — equilibrium deviation

        Returns:
            dict with:
                route: [B, T] — 0=forward, 1=branch, 2=backward, 3=plastic
                route_probs: [B, T, 4] — soft routing probabilities
        """
        if self.training:
            batch_thresholds = self._batch_thresholds(ed)
            with torch.no_grad():
                self.running_thresholds.mul_(1 - self.threshold_momentum).add_(
                    batch_thresholds.to(self.running_thresholds.device, self.running_thresholds.dtype),
                    alpha=self.threshold_momentum,
                )
                self.num_batches_tracked += 1
            base_thresholds = self.running_thresholds.to(device=ed.device, dtype=ed.dtype)
        elif self.num_batches_tracked.item() > 0:
            base_thresholds = self.running_thresholds.to(device=ed.device, dtype=ed.dtype)
        else:
            base_thresholds = self._batch_thresholds(ed).to(device=ed.device, dtype=ed.dtype)

        t1, t2, t3 = self._ordered_thresholds(base_thresholds)
        left_width = (t2 - t1).clamp_min(1e-3)
        right_width = (t3 - t2).clamp_min(1e-3)
        centers = torch.stack(
            [
                t1 - left_width,
                (t1 + t2) * 0.5,
                (t2 + t3) * 0.5,
                t3 + right_width,
            ]
        )
        logits = -self.temperature * (ed.unsqueeze(-1) - centers).abs()
        probs = torch.softmax(logits, dim=-1)

        thresholds = torch.stack([t1, t2, t3])
        route = torch.bucketize(ed, thresholds)

        return {"route": route, "route_probs": probs, "thresholds": thresholds}


class TokenEnergyBudget(nn.Module):
    """Limits compute per token based on ED.

    Low ED → 1 pass (minimum compute)
    Medium ED → 2 passes (branching)
    High ED → 3+ passes (backward + re-process)

    Total budget across all tokens is capped.
    """

    def __init__(self, max_budget_per_token: int = 4, total_budget_ratio: float = 2.0):
        super().__init__()
        self.max_per_token = max_budget_per_token
        self.total_budget_ratio = total_budget_ratio

    def forward(self, ed: torch.Tensor, route_probs: torch.Tensor) -> torch.Tensor:
        """Compute energy budget per token.

        Args:
            ed: [B, T] — equilibrium deviation
            route_probs: [B, T, 4] — routing probabilities

        Returns:
            budget: [B, T] — integer compute budget per token (1 to max_per_token)
        """
        B, T = ed.shape
        total_budget = int(T * self.total_budget_ratio)

        # Base budget from route: forward=1, branch=2, backward=3, plastic=4
        base_costs = torch.tensor([1.0, 2.0, 3.0, 4.0], device=ed.device)
        expected_cost = (route_probs * base_costs).sum(dim=-1)  # [B, T]

        # Scale to fit total budget
        cost_sum = expected_cost.sum(dim=-1, keepdim=True)  # [B, 1]
        scale = total_budget / (cost_sum + 1e-8)
        scale = scale.clamp(max=1.0)  # don't inflate, only deflate

        budget = (expected_cost * scale).clamp(min=1, max=self.max_per_token)
        return budget.round().long()

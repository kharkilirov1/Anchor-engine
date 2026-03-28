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

    Thresholds are learnable parameters.
    """

    def __init__(self, init_thresholds: tuple[float, float, float] = (0.5, 0.5, 1.0)):
        super().__init__()
        # Constrained thresholds: theta1 < theta2 < theta3 via softplus deltas
        self.theta1_raw = nn.Parameter(torch.tensor(init_thresholds[0]))
        self.theta2_delta = nn.Parameter(torch.tensor(init_thresholds[1]))  # > 0 via softplus
        self.theta3_delta = nn.Parameter(torch.tensor(init_thresholds[2]))  # > 0 via softplus

    @property
    def theta1(self) -> torch.Tensor:
        return self.theta1_raw

    @property
    def theta2(self) -> torch.Tensor:
        return self.theta1 + F.softplus(self.theta2_delta)

    @property
    def theta3(self) -> torch.Tensor:
        return self.theta2 + F.softplus(self.theta3_delta)

    def forward(self, ed: torch.Tensor) -> dict:
        """Classify each token into routing buckets.

        Args:
            ed: [B, T] — equilibrium deviation

        Returns:
            dict with:
                route: [B, T] — 0=forward, 1=branch, 2=backward, 3=plastic
                route_probs: [B, T, 4] — soft routing probabilities
        """
        t1, t2, t3 = self.theta1, self.theta2, self.theta3

        p_fwd = torch.sigmoid(5.0 * (t1 - ed))
        p_branch = torch.sigmoid(5.0 * (ed - t1)) * torch.sigmoid(5.0 * (t2 - ed))
        p_back = torch.sigmoid(5.0 * (ed - t2)) * torch.sigmoid(5.0 * (t3 - ed))
        p_plastic = torch.sigmoid(5.0 * (ed - t3))

        # Stack and normalize
        probs = torch.stack([p_fwd, p_branch, p_back, p_plastic], dim=-1)  # [B, T, 4]
        probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-8)

        # Hard routing (argmax for inference, soft for training)
        route = probs.argmax(dim=-1)  # [B, T]

        return {"route": route, "route_probs": probs}


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

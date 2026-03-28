"""Adaptive Routing — data flows where it needs processing.

Based on Equilibrium Signal, tokens are routed:
- Forward: next layer (normal)
- Branch: split into 2-3 routes through different layer paths
- Backward: re-process through earlier layers (selective forgetting)
- Plastic: activate plastic layer for adaptation

Uses SoA scatter/gather for efficient batching:
tokens grouped by route into dense arrays for GPU efficiency.
"""
import torch
import torch.nn as nn
from src.model.equilibrium import EquilibriumSignal, RoutingDecision, TokenEnergyBudget


class ScatterGather(nn.Module):
    """Groups tokens by route into dense buckets, processes, then restores order.

    Scatter: sort tokens by route → dense per-route arrays
    Compute: process each route's tokens as a dense batch
    Gather: restore original token positions
    """

    @staticmethod
    def scatter(x: torch.Tensor, route: torch.Tensor, n_routes: int = 4) -> dict:
        """Group tokens by route.

        Args:
            x: [B, T, D] — token representations
            route: [B, T] — route assignment (0..n_routes-1)

        Returns:
            dict mapping route_id → {tokens: [N_i, D], indices: [(b,t) pairs]}
        """
        B, T, D = x.shape
        buckets = {}
        for r in range(n_routes):
            mask = (route == r)  # [B, T]
            if mask.any():
                # Gather tokens for this route
                indices = mask.nonzero(as_tuple=False)  # [N_i, 2] — (batch_idx, seq_idx)
                tokens = x[indices[:, 0], indices[:, 1]]  # [N_i, D]
                buckets[r] = {"tokens": tokens, "indices": indices}
        return buckets

    @staticmethod
    def gather(buckets: dict, shape: tuple, device: torch.device) -> torch.Tensor:
        """Restore tokens to original positions.

        Args:
            buckets: dict from scatter, with updated tokens
            shape: (B, T, D) — original shape
            device: target device

        Returns:
            x: [B, T, D] — reconstructed tensor
        """
        x = torch.zeros(shape, device=device)
        for r, bucket in buckets.items():
            indices = bucket["indices"]
            x[indices[:, 0], indices[:, 1]] = bucket["tokens"]
        return x


class AdaptiveRouter(nn.Module):
    """Full adaptive routing module.

    Integrates equilibrium signal, routing decision, energy budget,
    and scatter/gather for each transformer layer.
    """

    def __init__(self, d_model: int, n_layers: int):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers

        # Per-layer equilibrium signals
        self.eq_signals = nn.ModuleList([
            EquilibriumSignal(d_model) for _ in range(n_layers)
        ])
        self.router = RoutingDecision()
        self.energy = TokenEnergyBudget()

    def compute_route(self, x: torch.Tensor, layer_idx: int) -> dict:
        """Compute routing for tokens at a given layer.

        Args:
            x: [B, T, D] — activations after layer
            layer_idx: which layer just processed

        Returns:
            dict with ed, route, route_probs, budget
        """
        eq_out = self.eq_signals[layer_idx](x)
        route_out = self.router(eq_out["ed"])
        budget = self.energy(eq_out["ed"], route_out["route_probs"])

        return {
            "ed": eq_out["ed"],
            "route": route_out["route"],
            "route_probs": route_out["route_probs"],
            "budget": budget,
        }

    def get_route_stats(self, route: torch.Tensor) -> dict:
        """Get statistics about routing decisions.

        Args:
            route: [B, T] — route assignments

        Returns:
            dict with counts and ratios for each route
        """
        total = route.numel()
        stats = {}
        names = ["forward", "branch", "backward", "plastic"]
        for i, name in enumerate(names):
            count = (route == i).sum().item()
            stats[name] = count
            stats[f"{name}_ratio"] = count / total if total > 0 else 0
        return stats

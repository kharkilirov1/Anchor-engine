"""ABPT Stage B — Unified model with Equilibrium Signal + Adaptive Routing.

Stage A modules (AttnRes, Plastic, Branches, Verifier) are still present,
but now controlled by the unified equilibrium signal instead of being always-on.

Three fundamental mechanisms:
1. Equilibrium Signal — deviation from running mean drives everything
2. Adaptive Routing — tokens go forward/branch/backward/plastic based on ED
3. Token Energy Budget — limits compute per token
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.config import ModelConfig
from src.model.backbone import Backbone, TransformerBlock
from src.model.plastic import PlasticLayer
from src.model.branches import BranchRouter
from src.model.verifier import Verifier
from src.model.equilibrium import EquilibriumSignal, RoutingDecision, TokenEnergyBudget
from src.model.adaptive_routing import ScatterGather


class ABPTModelB(nn.Module):
    """Stage B: Unified ABPT with adaptive routing.

    Key difference from Stage A:
    - Equilibrium signal computed after each layer
    - Routing determines which tokens get branches, backward pass, or plasticity
    - Not all tokens go through all modules — only those that need it
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(cfg, i) for i in range(cfg.n_layers)
        ])
        self.ln_final = nn.LayerNorm(cfg.d_model)

        # Equilibrium signal per layer
        self.eq_signals = nn.ModuleList([
            EquilibriumSignal(cfg.d_model, momentum=cfg.eq_momentum, warmup_steps=cfg.eq_warmup_steps)
            for _ in range(cfg.n_layers)
        ])
        self.router = RoutingDecision(
            target_fractions=(
                cfg.route_forward_target,
                cfg.route_branch_target,
                cfg.route_backward_target,
                cfg.route_plastic_target,
            ),
            threshold_momentum=cfg.route_threshold_momentum,
            temperature=cfg.route_temperature,
            offset_scale=cfg.route_threshold_offset_scale,
        )
        self.energy = TokenEnergyBudget()

        # Stage A modules — activated selectively by routing
        if cfg.use_plastic:
            self.plastic = PlasticLayer(cfg)
        if cfg.use_branches:
            self.branch_router = BranchRouter(cfg)
        if cfg.use_verifier and cfg.use_branches:
            self.verifier = Verifier(cfg)

        # Single lm_head for non-branched tokens
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> dict:
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.drop(self.tok_emb(input_ids) + self.pos_emb(pos))

        result = {}
        layer_outputs = [x]
        all_route_stats = []

        for i, block in enumerate(self.blocks):
            # Normal forward pass through block
            x = block(x, layer_outputs)
            layer_outputs.append(x)

            # Compute equilibrium signal
            eq_out = self.eq_signals[i](x)
            ed = eq_out["ed"]  # [B, T]
            route_preview = self.router(ed)
            thresholds = route_preview["thresholds"]

            # During warmup: force all tokens to forward (accumulate stats only)
            if eq_out.get("warming_up", False):
                route = torch.zeros(B, T, dtype=torch.long, device=x.device)
                route_probs = route_preview["route_probs"]
            else:
                route = route_preview["route"]  # [B, T]: 0=fwd, 1=branch, 2=back, 3=plastic
                route_probs = route_preview["route_probs"]

            # Collect stats
            total = route.numel()
            stats = {
                "forward": (route == 0).sum().item() / total,
                "branch": (route == 1).sum().item() / total,
                "backward": (route == 2).sum().item() / total,
                "plastic": (route == 3).sum().item() / total,
                "mean_ed": ed.mean().item(),
                "theta1": thresholds[0].item(),
                "theta2": thresholds[1].item(),
                "theta3": thresholds[2].item(),
            }
            all_route_stats.append(stats)

            # === Apply routing effects ===

            # Backward tokens: re-process through previous layer (if exists)
            if i > 0:
                back_mask = (route == 2)  # [B, T]
                if back_mask.any():
                    back_indices = back_mask.nonzero(as_tuple=False)
                    back_tokens = x[back_indices[:, 0], back_indices[:, 1]]
                    # Re-process through previous block (selective forgetting + reinterpretation)
                    prev_layer_outs = layer_outputs[:i]  # exclude current
                    back_tokens_unsq = back_tokens.unsqueeze(1)  # [N, 1, D]
                    # Simple re-process: just run through previous block
                    prev_outs_for_back = [lo[back_indices[:, 0], back_indices[:, 1]].unsqueeze(1) for lo in prev_layer_outs]
                    reprocessed = self.blocks[i - 1](back_tokens_unsq, prev_outs_for_back)
                    x[back_indices[:, 0], back_indices[:, 1]] = reprocessed.squeeze(1)

            # Plastic tokens: apply plastic adaptation
            if self.cfg.use_plastic:
                plastic_mask = (route == 3)
                if plastic_mask.any():
                    p_indices = plastic_mask.nonzero(as_tuple=False)
                    p_tokens = x[p_indices[:, 0], p_indices[:, 1]].unsqueeze(1)
                    adapted = self.plastic(p_tokens)
                    x[p_indices[:, 0], p_indices[:, 1]] = adapted.squeeze(1)

        # Final norm
        hidden = self.ln_final(x)

        # === Output heads ===
        # Branch tokens get branch+verifier, others get lm_head
        if self.cfg.use_branches:
            # Use last layer's route for output decision
            last_route = route  # from final layer
            branch_mask = (last_route == 1)  # [B, T]

            if branch_mask.any() and branch_mask.sum() > 0:
                # Branch path
                b_indices = branch_mask.nonzero(as_tuple=False)
                b_tokens = hidden[b_indices[:, 0], b_indices[:, 1]].unsqueeze(1)
                branch_out = self.branch_router(b_tokens)
                result["diversity_loss"] = branch_out["diversity_loss"]
                result["branch_logits"] = branch_out["branch_logits"]

                if self.cfg.use_verifier:
                    v_out = self.verifier(branch_out["branch_logits"])
                    branch_logits = v_out["logits"]  # [N, 1, V]
                else:
                    branch_logits = branch_out["logits"]

                # Non-branch path
                all_logits = self.lm_head(hidden)  # [B, T, V]
                # Override branch positions
                all_logits[b_indices[:, 0], b_indices[:, 1]] = branch_logits.squeeze(1)
                logits = all_logits
            else:
                logits = self.lm_head(hidden)
                result["diversity_loss"] = torch.tensor(0.0, device=hidden.device)
        else:
            logits = self.lm_head(hidden)

        result["logits"] = logits
        result["route_stats"] = all_route_stats
        result["last_route_probs"] = route_probs
        result["last_route_thresholds"] = thresholds
        result["hidden"] = hidden

        # Loss
        if targets is not None:
            ce_loss = F.cross_entropy(
                logits.view(B * T, -1), targets.view(B * T)
            )
            total_loss = ce_loss
            if self.cfg.use_branches and "diversity_loss" in result:
                total_loss = total_loss + self.cfg.diversity_weight * result["diversity_loss"]
            result["loss"] = total_loss
            result["ce_loss"] = ce_loss

        return result

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def param_count_str(self) -> str:
        n = self.param_count()
        if n >= 1_000_000:
            return f"{n / 1_000_000:.1f}M"
        return f"{n / 1_000:.1f}K"

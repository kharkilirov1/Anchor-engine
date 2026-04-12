from __future__ import annotations

import torch
import torch.nn as nn

from src.model.config import ModelConfig


class ProposalRolloutBranch(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        hidden_dim = max(32, int(cfg.anchor_proposal_rollout_hidden))
        fusion_dim = cfg.d_model * 5
        self.seed_proj = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, cfg.d_model),
        )
        self.cond_proj = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, cfg.d_model),
        )
        self.input_proj = nn.Sequential(
            nn.Linear(cfg.d_model * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, cfg.d_model),
        )
        self.step_emb = nn.Embedding(max(1, int(cfg.anchor_proposal_rollout_steps)), cfg.d_model)
        self.cell = nn.GRUCell(cfg.d_model, cfg.d_model)
        self.state_norm = nn.LayerNorm(cfg.d_model)
        self.summary_gate = nn.Linear(cfg.d_model * 2, 1)

    def forward(
        self,
        anchor_repr: torch.Tensor,
        proposal_repr: torch.Tensor,
        context_repr: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        fusion = torch.cat(
            [
                anchor_repr,
                proposal_repr,
                context_repr,
                proposal_repr - anchor_repr,
                proposal_repr * anchor_repr,
            ],
            dim=-1,
        )
        condition = self.cond_proj(fusion)
        state = proposal_repr + float(self.cfg.anchor_proposal_rollout_residual_scale) * self.seed_proj(fusion)

        states: list[torch.Tensor] = []
        for step_idx in range(max(1, int(self.cfg.anchor_proposal_rollout_steps))):
            step_vec = self.step_emb.weight[step_idx]
            step_input = self.input_proj(torch.cat([condition, state, step_vec], dim=-1))
            state = self.cell(step_input.unsqueeze(0), state.unsqueeze(0)).squeeze(0)
            states.append(self.state_norm(state))

        rollout_states = torch.stack(states, dim=0)
        gate_in = torch.cat(
            [rollout_states, condition.unsqueeze(0).expand_as(rollout_states)],
            dim=-1,
        )
        summary_gate = torch.sigmoid(self.summary_gate(gate_in))
        summary = (summary_gate * rollout_states).sum(dim=0) / summary_gate.sum(dim=0).clamp_min(1e-6)
        return {
            "states": rollout_states,
            "summary": summary,
        }

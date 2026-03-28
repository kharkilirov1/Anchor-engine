import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.config import ModelConfig


class Verifier(nn.Module):
    """Compares branch hypotheses and selects the most robust one.
    Scoring: weighted combination of entropy and agreement.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.entropy_w = cfg.verifier_entropy_weight
        self.agreement_w = cfg.verifier_agreement_weight

    def _entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """Per-position entropy. Lower = more confident. [B, T]"""
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        return -(probs * log_probs).sum(dim=-1)

    def _agreement(self, branch_logits: list[torch.Tensor]) -> torch.Tensor:
        """How much each branch agrees with the mean. [B, T, N_branches]"""
        probs = [F.softmax(bl, dim=-1) for bl in branch_logits]
        mean_probs = torch.stack(probs).mean(dim=0)
        agreements = []
        for p in probs:
            cos = F.cosine_similarity(p, mean_probs, dim=-1)
            agreements.append(cos)
        return torch.stack(agreements, dim=-1)

    def forward(self, branch_logits: list[torch.Tensor]) -> dict:
        # Entropy score per branch: lower entropy = higher score
        entropies = torch.stack(
            [self._entropy(bl) for bl in branch_logits], dim=-1
        )  # [B, T, N]
        max_ent = entropies.max(dim=-1, keepdim=True).values.clamp(min=1e-8)
        entropy_scores = 1.0 - entropies / max_ent  # [B, T, N]

        agreement_scores = self._agreement(branch_logits)  # [B, T, N]

        scores = self.entropy_w * entropy_scores + self.agreement_w * agreement_scores

        branch_weights = F.softmax(scores * 5.0, dim=-1)  # [B, T, N]

        stacked = torch.stack(branch_logits, dim=-2)  # [B, T, N, V]
        weighted_logits = (stacked * branch_weights.unsqueeze(-1)).sum(dim=-2)

        confidence = branch_weights.max(dim=-1).values  # [B, T]

        return {
            "logits": weighted_logits,
            "confidence": confidence,
            "branch_weights": branch_weights,
            "entropy_scores": entropy_scores,
            "agreement_scores": agreement_scores,
        }

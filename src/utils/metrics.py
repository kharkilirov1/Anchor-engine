import torch
import torch.nn.functional as F
import math


def bits_per_byte(ce_loss: float) -> float:
    """Convert cross-entropy loss (nats) to bits per byte."""
    return ce_loss / math.log(2)


def branch_diversity(branch_logits: list[torch.Tensor]) -> float:
    """Average cosine distance between branch probability distributions."""
    if len(branch_logits) < 2:
        return 0.0
    total = 0.0
    n_pairs = 0
    for i in range(len(branch_logits)):
        for j in range(i + 1, len(branch_logits)):
            pi = F.softmax(branch_logits[i], dim=-1).reshape(-1, branch_logits[i].size(-1))
            pj = F.softmax(branch_logits[j], dim=-1).reshape(-1, branch_logits[j].size(-1))
            cos_sim = F.cosine_similarity(pi, pj, dim=-1).mean().item()
            total += 1.0 - cos_sim
            n_pairs += 1
    return total / n_pairs if n_pairs > 0 else 0.0

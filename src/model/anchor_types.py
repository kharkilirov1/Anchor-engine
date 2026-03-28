from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch


class AnchorState(str, Enum):
    CANDIDATE = "candidate"
    PROVISIONAL = "provisional"
    CONFIRMED = "confirmed"
    DECAYING = "decaying"
    DEAD_END = "dead_end"


@dataclass
class AnchorCandidate:
    start_idx: int
    end_idx: int
    repr: torch.Tensor
    score: torch.Tensor | float
    semantic_weight: torch.Tensor | float


@dataclass
class AnchorRecord:
    id: int
    start_idx: int
    end_idx: int
    repr: torch.Tensor
    score: torch.Tensor | float
    state: AnchorState
    support: torch.Tensor | float
    contradiction_pressure: torch.Tensor | float
    viability: torch.Tensor | float
    ttl: torch.Tensor | float
    parent_id: int | None = None
    branch_id: int | None = None
    descendant_mass: torch.Tensor | float | None = None
    descendant_coherence: torch.Tensor | float | None = None


@dataclass
class RevisionDecision:
    anchor_id: int
    action: str
    reason: str
    new_state: AnchorState
    alt_branch_used: bool
    action_probs: dict[str, float] | None = None

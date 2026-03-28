from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class AnchorProbeCase:
    name: str
    description: str
    input_ids: torch.Tensor
    target_ids: torch.Tensor
    expected_anchor_zone: tuple[int, int]
    expected_failure_mode: str


def _make_targets(input_ids: torch.Tensor) -> torch.Tensor:
    return torch.roll(input_ids, shifts=-1, dims=0)


def make_anchor_probe_cases(
    seq_len: int = 24,
    vocab_size: int = 512,
) -> list[AnchorProbeCase]:
    if seq_len < 16:
        raise ValueError('seq_len must be at least 16 for anchor probe cases')

    def clip(values: list[int]) -> torch.Tensor:
        return torch.tensor([v % vocab_size for v in values], dtype=torch.long)

    cases: list[AnchorProbeCase] = []

    stable = clip(
        [3] * 4 + [17] * 4 + [17] * 4 + [22] * 4 + [22] * 4 + [29] * 4
    )
    cases.append(
        AnchorProbeCase(
            name='stable_regime',
            description='Smooth regime shifts without a strong late contradiction pattern.',
            input_ids=stable,
            target_ids=_make_targets(stable),
            expected_anchor_zone=(4, 11),
            expected_failure_mode='none_or_low_pressure',
        )
    )

    quantifier_conflict = clip(
        [5] * 4 + [111] * 4 + [111] * 4 + [7] * 4 + [221] * 4 + [221] * 4
    )
    cases.append(
        AnchorProbeCase(
            name='quantifier_conflict',
            description='Early root stays coherent, then a later incompatible regime appears sharply.',
            input_ids=quantifier_conflict,
            target_ids=_make_targets(quantifier_conflict),
            expected_anchor_zone=(4, 11),
            expected_failure_mode='late_conflict',
        )
    )

    proof_mode_conflict = clip(
        [9] * 4 + [87] * 4 + [12] * 4 + [87] * 4 + [240] * 4 + [12] * 4
    )
    cases.append(
        AnchorProbeCase(
            name='proof_mode_conflict',
            description='Alternating motif with a late contradictory mode spike.',
            input_ids=proof_mode_conflict,
            target_ids=_make_targets(proof_mode_conflict),
            expected_anchor_zone=(4, 15),
            expected_failure_mode='mode_flip',
        )
    )

    complexity_conflict = clip(
        [14] * 4 + [14] * 4 + [41, 42, 43, 44] + [41, 42, 43, 44] + [250] * 4 + [41, 42, 43, 44]
    )
    cases.append(
        AnchorProbeCase(
            name='complexity_conflict',
            description='Structured mid-sequence pattern followed by a heavy late disruption.',
            input_ids=complexity_conflict,
            target_ids=_make_targets(complexity_conflict),
            expected_anchor_zone=(8, 15),
            expected_failure_mode='late_disruption',
        )
    )

    if any(case.input_ids.numel() != seq_len for case in cases):
        raise AssertionError('All anchor probe cases must match seq_len')

    return cases

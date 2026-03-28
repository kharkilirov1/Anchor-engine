from __future__ import annotations

from dataclasses import dataclass

import torch

from src.data.anchor_cases import AnchorProbeCase


@dataclass(frozen=True)
class SemanticTokenLegend:
    token_id: int
    label: str
    role: str


def semantic_token_legend() -> list[SemanticTokenLegend]:
    return [
        SemanticTokenLegend(11, 'FORALL', 'quantifier_root'),
        SemanticTokenLegend(12, 'EXISTS', 'conflicting_quantifier'),
        SemanticTokenLegend(13, 'VAR_N', 'bound_variable'),
        SemanticTokenLegend(14, 'CLAIM', 'assertion'),
        SemanticTokenLegend(15, 'STEP', 'proof_step'),
        SemanticTokenLegend(16, 'UNIFORM', 'uniform_descendant'),
        SemanticTokenLegend(17, 'WITNESS', 'existential_descendant'),
        SemanticTokenLegend(21, 'ASSUME_NOT', 'contradiction_root'),
        SemanticTokenLegend(22, 'DERIVE', 'contradiction_step'),
        SemanticTokenLegend(23, 'CONTRADICTION', 'contradiction_close'),
        SemanticTokenLegend(24, 'DIRECT', 'direct_mode_conflict'),
        SemanticTokenLegend(31, 'CONST', 'complexity_root'),
        SemanticTokenLegend(32, 'LOOKUP', 'constant_time_descendant'),
        SemanticTokenLegend(33, 'CACHE', 'constant_time_support'),
        SemanticTokenLegend(34, 'LOOP', 'linear_conflict'),
        SemanticTokenLegend(35, 'SCAN', 'linear_descendant'),
        SemanticTokenLegend(41, 'INDUCT', 'induction_root'),
        SemanticTokenLegend(42, 'BASE', 'induction_base_case'),
        SemanticTokenLegend(43, 'STEP_K', 'induction_step'),
        SemanticTokenLegend(44, 'STEP_K1', 'induction_successor'),
        SemanticTokenLegend(45, 'EXAMPLE', 'finite_example_conflict'),
        SemanticTokenLegend(51, 'EPS', 'epsilon_root'),
        SemanticTokenLegend(52, 'DELTA', 'delta_descendant'),
        SemanticTokenLegend(53, 'BOUND', 'formal_bound_descendant'),
        SemanticTokenLegend(54, 'CLOSE', 'intuitive_closeness_conflict'),
    ]


def _make_targets(input_ids: torch.Tensor) -> torch.Tensor:
    return torch.roll(input_ids, shifts=-1, dims=0)


def make_semantic_anchor_cases(seq_len: int = 24) -> list[AnchorProbeCase]:
    if seq_len != 24:
        raise ValueError('semantic probe cases currently require seq_len=24')

    def t(values: list[int]) -> torch.Tensor:
        return torch.tensor(values, dtype=torch.long)

    cases: list[AnchorProbeCase] = []

    forall_stable = t([
        11, 13, 11, 16,
        11, 13, 11, 16,
        11, 13, 11, 16,
        11, 13, 11, 16,
        11, 13, 11, 16,
        11, 13, 11, 16,
    ])
    cases.append(
        AnchorProbeCase(
            name='forall_stable',
            description='FORALL root followed by uniform descendants that stay semantically consistent.',
            input_ids=forall_stable,
            target_ids=_make_targets(forall_stable),
            expected_anchor_zone=(0, 11),
            expected_failure_mode='stable_quantifier_tree',
        )
    )

    forall_exists_conflict = t([
        11, 13, 11, 16,
        11, 13, 11, 16,
        11, 13, 11, 16,
        12, 17, 12, 17,
        12, 17, 12, 17,
        12, 17, 12, 17,
    ])
    cases.append(
        AnchorProbeCase(
            name='forall_exists_conflict',
            description='FORALL root later flips into EXISTS-style descendants.',
            input_ids=forall_exists_conflict,
            target_ids=_make_targets(forall_exists_conflict),
            expected_anchor_zone=(0, 11),
            expected_failure_mode='quantifier_flip',
        )
    )

    contradiction_stable = t([
        21, 14, 22, 15,
        21, 14, 22, 15,
        21, 14, 22, 15,
        21, 14, 23, 15,
        21, 14, 23, 15,
        21, 14, 23, 15,
    ])
    cases.append(
        AnchorProbeCase(
            name='contradiction_stable',
            description='ASSUME_NOT mode stays in contradiction style until closure.',
            input_ids=contradiction_stable,
            target_ids=_make_targets(contradiction_stable),
            expected_anchor_zone=(0, 15),
            expected_failure_mode='stable_contradiction_tree',
        )
    )

    contradiction_direct_conflict = t([
        21, 14, 22, 15,
        21, 14, 22, 15,
        21, 14, 22, 15,
        24, 14, 15, 15,
        24, 14, 15, 15,
        24, 14, 15, 15,
    ])
    cases.append(
        AnchorProbeCase(
            name='contradiction_direct_conflict',
            description='ASSUME_NOT root later drifts into direct-proof style descendants.',
            input_ids=contradiction_direct_conflict,
            target_ids=_make_targets(contradiction_direct_conflict),
            expected_anchor_zone=(0, 11),
            expected_failure_mode='proof_mode_flip',
        )
    )

    const_vs_loop_conflict = t([
        31, 32, 33, 15,
        31, 32, 33, 15,
        31, 32, 33, 15,
        34, 35, 35, 15,
        34, 35, 35, 15,
        34, 35, 35, 15,
    ])
    cases.append(
        AnchorProbeCase(
            name='const_vs_loop_conflict',
            description='Constant-time root later drifts into loop/scan descendants.',
            input_ids=const_vs_loop_conflict,
            target_ids=_make_targets(const_vs_loop_conflict),
            expected_anchor_zone=(0, 11),
            expected_failure_mode='complexity_flip',
        )
    )

    induction_stable = t([
        41, 42, 43, 44,
        41, 42, 43, 44,
        41, 42, 43, 44,
        41, 42, 43, 44,
        41, 42, 43, 44,
        41, 42, 43, 44,
    ])
    cases.append(
        AnchorProbeCase(
            name='induction_stable',
            description='INDUCT root followed by repeated base/step/successor structure.',
            input_ids=induction_stable,
            target_ids=_make_targets(induction_stable),
            expected_anchor_zone=(0, 11),
            expected_failure_mode='stable_induction_tree',
        )
    )

    induction_example_conflict = t([
        41, 42, 43, 44,
        41, 42, 43, 44,
        41, 42, 43, 44,
        45, 45, 45, 15,
        45, 45, 45, 15,
        45, 45, 45, 15,
    ])
    cases.append(
        AnchorProbeCase(
            name='induction_example_conflict',
            description='INDUCT root later collapses into repeated finite examples instead of inductive structure.',
            input_ids=induction_example_conflict,
            target_ids=_make_targets(induction_example_conflict),
            expected_anchor_zone=(0, 11),
            expected_failure_mode='induction_to_examples',
        )
    )

    epsilon_delta_stable = t([
        51, 52, 53, 15,
        51, 52, 53, 15,
        51, 52, 53, 15,
        51, 52, 53, 15,
        51, 52, 53, 15,
        51, 52, 53, 15,
    ])
    cases.append(
        AnchorProbeCase(
            name='epsilon_delta_stable',
            description='EPS root followed by DELTA/BOUND descendants that remain formal.',
            input_ids=epsilon_delta_stable,
            target_ids=_make_targets(epsilon_delta_stable),
            expected_anchor_zone=(0, 11),
            expected_failure_mode='stable_formal_limit_tree',
        )
    )

    epsilon_close_conflict = t([
        51, 52, 53, 15,
        51, 52, 53, 15,
        51, 52, 53, 15,
        54, 54, 54, 15,
        54, 54, 54, 15,
        54, 54, 54, 15,
    ])
    cases.append(
        AnchorProbeCase(
            name='epsilon_close_conflict',
            description='EPS root later drifts into intuitive closeness language instead of formal bound structure.',
            input_ids=epsilon_close_conflict,
            target_ids=_make_targets(epsilon_close_conflict),
            expected_anchor_zone=(0, 11),
            expected_failure_mode='formal_to_intuitive_drift',
        )
    )

    return cases

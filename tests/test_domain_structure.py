import torch

from src.utils.domain_structure import (
    compute_sequence_structure_stats,
    compute_token_structure_stats,
)


def test_compute_token_structure_stats_basic() -> None:
    token_ids = torch.tensor([1, 2, 1, 2, 1, 2, 1, 2], dtype=torch.long)
    stats = compute_token_structure_stats(token_ids, vocab_size=4)
    assert stats.token_count == 8
    assert stats.unique_tokens == 2
    assert stats.repeat_within_4 > 0.5
    assert stats.mean_next_token_peak_prob > 0.9


def test_compute_sequence_structure_stats_basic() -> None:
    sequences = torch.tensor(
        [
            [1, 2, 3],
            [1, 2, 3],
            [3, 2, 1],
            [1, 2, 3],
        ],
        dtype=torch.long,
    )
    stats = compute_sequence_structure_stats(sequences)
    assert stats.num_sequences == 4
    assert stats.unique_sequences == 2
    assert abs(stats.sequence_uniqueness_ratio - 0.5) < 1e-6
    assert abs(stats.top_sequence_mass - 0.75) < 1e-6

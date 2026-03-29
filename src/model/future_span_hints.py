from __future__ import annotations

from typing import Any

import torch

_STOPWORD_HINTS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "to",
    "of",
    "in",
    "on",
    "by",
    "for",
    "that",
    "same",
}


def is_informative_hint_text(text: str) -> bool:
    cleaned = text.strip().lower()
    if not cleaned:
        return False
    if not any(char.isalnum() for char in cleaned):
        return False
    words = [word for word in cleaned.replace("-", " ").split() if word]
    if not words:
        return False
    if len(words) <= 2 and all(word in _STOPWORD_HINTS for word in words):
        return False
    return True


def decode_span_text(tokenizer: Any, token_ids: list[int]) -> str:
    if tokenizer is None:
        return " ".join(str(token_id) for token_id in token_ids)
    try:
        text = tokenizer.decode(token_ids, skip_special_tokens=False)
    except TypeError:
        text = tokenizer.decode(token_ids)
    return text.replace("\n", "\\n")


def safe_decode_token(tokenizer: Any, token_id: int) -> str:
    if tokenizer is None:
        return str(token_id)
    try:
        text = tokenizer.decode([token_id], skip_special_tokens=False)
    except TypeError:
        text = tokenizer.decode([token_id])
    return text.replace("\n", "\\n")


def spans_overlap(span_a: dict[str, Any], span_b: dict[str, Any]) -> bool:
    return not (int(span_a["end"]) < int(span_b["start"]) or int(span_b["end"]) < int(span_a["start"]))


def extract_high_influence_spans(
    scores: torch.Tensor,
    input_ids: torch.Tensor,
    tokenizer: Any,
    min_score: float,
    top_spans: int,
) -> list[dict[str, Any]]:
    selected = [
        idx
        for idx, value in enumerate(scores.tolist())
        if float(value) >= float(min_score)
    ]
    if not selected:
        return []

    spans: list[tuple[int, int]] = []
    start = selected[0]
    prev = selected[0]
    for idx in selected[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        spans.append((start, prev))
        start = idx
        prev = idx
    spans.append((start, prev))

    ranked: list[dict[str, Any]] = []
    for start_idx, end_idx in spans:
        span_scores = scores[start_idx : end_idx + 1]
        token_ids = [int(token.item()) for token in input_ids[start_idx : end_idx + 1]]
        ranked.append(
            {
                "start": int(start_idx),
                "end": int(end_idx),
                "length": int(end_idx - start_idx + 1),
                "mean_score": float(span_scores.mean().item()),
                "max_score": float(span_scores.max().item()),
                "token_ids": token_ids,
                "text": decode_span_text(tokenizer, token_ids),
            }
        )
    ranked.sort(key=lambda item: (item["mean_score"], item["length"], item["max_score"]), reverse=True)
    return ranked[:top_spans]


def compute_span_anchor_overlap(
    future_spans: list[dict[str, Any]],
    active_anchor_spans: list[dict[str, int]],
) -> dict[str, float]:
    if not future_spans:
        return {
            "future_span_overlap_ratio": 0.0,
            "anchor_span_overlap_ratio": 0.0,
        }

    future_overlap = sum(
        1 for span in future_spans if any(spans_overlap(span, anchor) for anchor in active_anchor_spans)
    )
    anchor_overlap = sum(
        1 for anchor in active_anchor_spans if any(spans_overlap(anchor, span) for span in future_spans)
    )
    return {
        "future_span_overlap_ratio": future_overlap / max(len(future_spans), 1),
        "anchor_span_overlap_ratio": anchor_overlap / max(len(active_anchor_spans), 1) if active_anchor_spans else 0.0,
    }


def build_future_hint_candidates(
    future_spans: list[dict[str, Any]],
    active_anchor_spans: list[dict[str, int]],
) -> list[dict[str, Any]]:
    hints: list[dict[str, Any]] = []
    for span in future_spans:
        if any(spans_overlap(span, anchor_span) for anchor_span in active_anchor_spans):
            continue
        if not is_informative_hint_text(str(span["text"])):
            continue
        hints.append(
            {
                "start": int(span["start"]),
                "end": int(span["end"]),
                "text": span["text"],
                "mean_score": float(span["mean_score"]),
                "max_score": float(span["max_score"]),
                "length": int(span["length"]),
            }
        )
    hints.sort(key=lambda item: (item["mean_score"], item["length"], item["max_score"]), reverse=True)
    return hints


def build_auxiliary_future_proposals(
    hidden: torch.Tensor,
    input_ids: torch.Tensor,
    future_hint_candidates: list[dict[str, Any]],
    tokenizer: Any,
    max_candidates: int = 3,
) -> list[dict[str, Any]]:
    proposals: list[dict[str, Any]] = []
    for hint in future_hint_candidates[:max_candidates]:
        start = max(0, min(int(hint["start"]), hidden.size(0) - 1))
        end = max(start, min(int(hint["end"]), hidden.size(0) - 1))
        span_hidden = hidden[start : end + 1]
        span_ids = [int(token.item()) for token in input_ids[start : end + 1]]
        proposals.append(
            {
                "proposal_type": "future_hint_span",
                "proposal_score": float(hint["mean_score"]),
                "proposal_span": (start, end),
                "proposal_root_token": span_ids[-1] if span_ids else None,
                "proposal_text": decode_span_text(tokenizer, span_ids),
                "repr": span_hidden.mean(dim=0).detach(),
            }
        )
    return proposals


def summarize_auxiliary_proposals(
    proposal_batches: list[list[dict[str, Any]]],
) -> dict[str, float]:
    counts = [len(batch) for batch in proposal_batches]
    all_scores = [float(item["proposal_score"]) for batch in proposal_batches for item in batch]
    return {
        "proposal_count": int(sum(counts)),
        "batch_with_proposals_count": int(sum(1 for count in counts if count > 0)),
        "mean_proposal_count_per_batch": float(sum(counts) / max(len(counts), 1)),
        "mean_proposal_score": float(sum(all_scores) / max(len(all_scores), 1)) if all_scores else 0.0,
        "max_proposal_score": max(all_scores) if all_scores else 0.0,
    }

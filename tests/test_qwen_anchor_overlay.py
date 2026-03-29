from __future__ import annotations

from dataclasses import replace

import torch
import torch.nn as nn

from src.model.config import TOY_CONFIG
from src.model.qwen_anchor_overlay import QwenAnchorOverlay
from scripts.run_qwen_anchor_probe import build_markdown_report, summarize_results


class _DummyConfig:
    def __init__(self, hidden_size: int = 32, vocab_size: int = 97, max_position_embeddings: int = 64):
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings


class _DummyOutput:
    def __init__(self, logits: torch.Tensor, hidden_states: tuple[torch.Tensor, ...]):
        self.logits = logits
        self.hidden_states = hidden_states


class _DummyCausalLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = _DummyConfig()
        self.emb = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        self.proj = nn.Linear(self.config.hidden_size, self.config.vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> _DummyOutput:
        del attention_mask, return_dict
        hidden = self.emb(input_ids)
        logits = self.proj(hidden)
        hidden_states = (hidden * 0.5, hidden) if output_hidden_states else (hidden,)
        return _DummyOutput(logits=logits, hidden_states=hidden_states)


class _DummyTokenizer:
    def __call__(
        self,
        texts: list[str],
        padding: bool,
        truncation: bool,
        max_length: int,
        return_tensors: str,
    ) -> dict[str, torch.Tensor]:
        del padding, truncation, return_tensors
        rows = []
        for text in texts:
            ids = [min(ord(ch), 96) % 97 for ch in text[:max_length]]
            rows.append(torch.tensor(ids, dtype=torch.long))
        padded = nn.utils.rnn.pad_sequence(rows, batch_first=True, padding_value=0)
        mask = (padded != 0).long()
        return {"input_ids": padded, "attention_mask": mask}


def test_qwen_anchor_overlay_forward_returns_diagnostics():
    model = _DummyCausalLM()
    cfg = replace(TOY_CONFIG, anchor_threshold=0.10)
    overlay = QwenAnchorOverlay(base_model=model, cfg=cfg)

    input_ids = torch.randint(0, model.config.vocab_size, (2, 12))
    out = overlay(input_ids)

    assert "logits" in out
    assert "anchor_diagnostics" in out
    assert "proposal_diagnostics" in out
    assert out["logits"].shape[:2] == input_ids.shape


def test_qwen_anchor_overlay_analyze_texts_uses_tokenizer():
    model = _DummyCausalLM()
    tokenizer = _DummyTokenizer()
    overlay = QwenAnchorOverlay(base_model=model, cfg=TOY_CONFIG, tokenizer=tokenizer)

    out, batch = overlay.analyze_texts(["alpha beta", "gamma"], max_length=16)

    assert "input_ids" in batch
    assert batch["input_ids"].shape[0] == 2
    assert "anchor_diagnostics" in out


def test_qwen_probe_summary_and_report_include_gaps():
    results = [
        {
            "name": "stable_case",
            "description": "stable",
            "expected_mode": "stable",
            "tokens": 10,
            "num_active": 1,
            "mean_contradiction_pressure": 0.2,
            "mean_viability": 0.6,
            "dead_end_count": 1,
            "proposal_count": 0,
        },
        {
            "name": "conflict_case",
            "description": "conflict",
            "expected_mode": "conflict",
            "tokens": 12,
            "num_active": 2,
            "mean_contradiction_pressure": 0.5,
            "mean_viability": 0.3,
            "dead_end_count": 2,
            "proposal_count": 0,
        },
    ]

    summary = summarize_results(results)
    report = build_markdown_report(
        model_name="Qwen/Qwen2.5-1.5B",
        device="cpu",
        max_length=64,
        seed=7,
        results=results,
        summary=summary,
    )

    assert summary["pressure_gap_conflict_minus_stable"] > 0
    assert "Qwen Anchor Probe Report" in report
    assert "conflict_case" in report

from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch
import torch.nn as nn

from src.model.anchor_detector import AnchorDetector
from src.model.anchor_memory import AnchorMemory
from src.model.anchor_monitor import ContradictionMonitor
from src.model.anchor_revision import RevisionController
from src.model.anchor_viability import ViabilityTracker
from src.model.config import ModelConfig, TOY_CONFIG


class QwenAnchorOverlay(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        cfg: ModelConfig | None = None,
        tokenizer: Any | None = None,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.tokenizer = tokenizer

        model_cfg = getattr(base_model, "config", None)
        if model_cfg is None:
            raise ValueError("base_model must expose a `.config` object")

        hidden_size = int(getattr(model_cfg, "hidden_size"))
        vocab_size = int(getattr(model_cfg, "vocab_size"))
        max_seq_len = int(
            getattr(
                model_cfg,
                "max_position_embeddings",
                getattr(model_cfg, "max_seq_len", 32768),
            )
        )

        anchor_cfg = replace(
            cfg or TOY_CONFIG,
            d_model=hidden_size,
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
        )
        self.cfg = anchor_cfg
        self.anchor_detector = AnchorDetector(anchor_cfg)
        self.anchor_memory = AnchorMemory(anchor_cfg)
        self.contradiction_monitor = ContradictionMonitor(anchor_cfg)
        self.viability_tracker = ViabilityTracker(anchor_cfg)
        self.revision_controller = RevisionController(anchor_cfg)

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "Qwen/Qwen2.5-1.5B",
        cfg: ModelConfig | None = None,
        device: str | torch.device | None = None,
        torch_dtype: torch.dtype | None = None,
        **kwargs: Any,
    ) -> "QwenAnchorOverlay":
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "transformers is required for QwenAnchorOverlay.from_pretrained"
            ) from exc

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            **kwargs,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        overlay = cls(base_model=model, cfg=cfg, tokenizer=tokenizer)
        if device is not None:
            overlay = overlay.to(device)
        return overlay

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden = outputs.hidden_states[-1]
        history = torch.roll(hidden.detach(), shifts=1, dims=1)
        history[:, 0] = hidden[:, 0].detach()

        detector_out = self.anchor_detector(hidden, history, attention_mask=attention_mask)
        anchors = self.anchor_memory.add_candidates(detector_out["candidates"])
        anchors = self.anchor_memory.update_support(anchors, detector_out["scores"])
        anchors = self.anchor_memory.update_ttl(anchors)

        contradiction_out = self.contradiction_monitor(
            hidden,
            anchors,
            aux={"input_ids": input_ids},
        )
        viability_out = self.viability_tracker(anchors, contradiction_out)
        revision_events = self.revision_controller(anchors, viability_out, arbiter={})
        anchors = self.anchor_memory.apply_revision(anchors, revision_events)
        active_anchors = self.anchor_memory.get_active_anchors(anchors)
        diagnostics = self.anchor_memory.export_diagnostics(anchors)
        diagnostics["revision_event_count"] = len(revision_events)

        return {
            "logits": outputs.logits,
            "hidden": hidden,
            "anchor_candidates": detector_out["candidates"],
            "active_anchors": active_anchors,
            "anchor_states": [[anchor.state for anchor in batch] for batch in anchors],
            "contradiction_pressure": contradiction_out["contradiction_pressure"],
            "viability": viability_out["viability"],
            "revision_events": revision_events,
            "anchor_diagnostics": diagnostics,
            "detector_scores": detector_out["scores"],
            "proposal_diagnostics": {
                "proposal_count": 0,
                "regime_shift_count": 0,
                "anchors_with_proposal_influence": 0,
                "mean_proposal_score": 0.0,
                "mean_blend_ratio": 0.0,
            },
        }

    def analyze_texts(
        self,
        texts: list[str],
        max_length: int = 256,
    ) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
        if self.tokenizer is None:
            raise ValueError("tokenizer is required for analyze_texts")

        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        device = next(self.parameters()).device
        batch = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            out = self(**batch)
        return out, batch

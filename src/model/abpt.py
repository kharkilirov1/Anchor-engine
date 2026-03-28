import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.config import ModelConfig
from src.model.backbone import Backbone
from src.model.plastic import PlasticLayer
from src.model.branches import BranchRouter
from src.model.verifier import Verifier


class ABPTModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.backbone = Backbone(cfg)

        if cfg.use_plastic:
            self.plastic = PlasticLayer(cfg)

        if cfg.use_branches:
            self.branch_router = BranchRouter(cfg)
        else:
            self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.use_verifier and cfg.use_branches:
            self.verifier = Verifier(cfg)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> dict:
        result = {}

        backbone_out = self.backbone(input_ids)
        hidden = backbone_out["hidden"]

        if self.cfg.use_plastic:
            hidden = self.plastic(hidden)

        if self.cfg.use_branches:
            branch_out = self.branch_router(hidden)
            result["diversity_loss"] = branch_out["diversity_loss"]
            result["branch_logits"] = branch_out["branch_logits"]

            if self.cfg.use_verifier:
                verifier_out = self.verifier(branch_out["branch_logits"])
                logits = verifier_out["logits"]
                result["confidence"] = verifier_out["confidence"]
                result["branch_weights"] = verifier_out["branch_weights"]
            else:
                logits = branch_out["logits"]
        else:
            logits = self.lm_head(hidden)

        result["logits"] = logits

        if targets is not None:
            B, T, V = logits.shape
            ce_loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))
            total_loss = ce_loss

            if self.cfg.use_branches:
                total_loss = total_loss + self.cfg.diversity_weight * result["diversity_loss"]

            result["loss"] = total_loss
            result["ce_loss"] = ce_loss

        return result

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def param_count_str(self) -> str:
        n = self.param_count()
        if n >= 1_000_000:
            return f"{n / 1_000_000:.1f}M"
        return f"{n / 1_000:.1f}K"

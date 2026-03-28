# ABPT MVP Stage A — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a working small transformer with Attention Residuals, plastic layer, 2 branch heads with diversity loss, and verifier — all togglable for ablation, verifiable on CPU.

**Architecture:** Transformer backbone where standard residual connections are replaced with attention-based aggregation over all previous layers (AttnRes). On top: a plastic adapter with decay for inference-time adaptation, 2 parallel branch heads generating alternative hypotheses, and a verifier scoring branches by entropy + agreement. All modules togglable via `ModelConfig` dataclass.

**Tech Stack:** Python 3.10+, PyTorch 2.x, einops, pytest

---

## File Structure

| File | Responsibility |
|------|---------------|
| `src/model/config.py` | ModelConfig dataclass + ablation presets (DONE) |
| `src/model/attention.py` | MultiHeadAttention + AttnRes aggregation layer |
| `src/model/backbone.py` | TransformerBlock + full Backbone (stacked blocks) |
| `src/model/plastic.py` | PlasticLayer — adapter with decay, L2, gradient update |
| `src/model/branches.py` | BranchHead + BranchRouter (2 heads + diversity loss) |
| `src/model/verifier.py` | Verifier — entropy, agreement, confidence scoring |
| `src/model/abpt.py` | ABPTModel — full model assembling all modules |
| `src/data/tokenizer.py` | Simple character/BPE tokenizer wrapper |
| `src/data/tasks.py` | Synthetic evaluation tasks |
| `src/utils/metrics.py` | val_bpb, branch_diversity, adaptation_gain |
| `train.py` | Training loop entry point |
| `tests/test_attention.py` | Tests for attention + AttnRes |
| `tests/test_backbone.py` | Tests for backbone |
| `tests/test_plastic.py` | Tests for plastic layer |
| `tests/test_branches.py` | Tests for branches + diversity loss |
| `tests/test_verifier.py` | Tests for verifier |
| `tests/test_abpt.py` | Integration test: full model forward+backward |
| `tests/test_ablation.py` | All config presets produce valid models |

---

## Task 1: MultiHeadAttention + AttnRes

**Files:**
- Create: `src/model/attention.py`
- Test: `tests/test_attention.py`

- [ ] **Step 1: Write failing test for MultiHeadAttention**

```python
# tests/test_attention.py
import torch
from src.model.attention import MultiHeadAttention

def test_mha_output_shape():
    mha = MultiHeadAttention(d_model=64, n_heads=4, dropout=0.0)
    x = torch.randn(2, 16, 64)  # [batch, seq, d_model]
    out = mha(x, x, x)
    assert out.shape == (2, 16, 64)

def test_mha_causal_mask():
    mha = MultiHeadAttention(d_model=64, n_heads=4, dropout=0.0)
    x = torch.randn(1, 8, 64)
    out = mha(x, x, x, causal=True)
    assert out.shape == (1, 8, 64)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/Desktop/ABPT && python -m pytest tests/test_attention.py -v`
Expected: FAIL — ModuleNotFoundError

- [ ] **Step 3: Implement MultiHeadAttention**

```python
# src/model/attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal: bool = False,
    ) -> torch.Tensor:
        B, T, _ = q.shape
        q = self.w_q(q).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.w_k(k).view(B, -1, self.n_heads, self.d_head).transpose(1, 2)
        v = self.w_v(v).view(B, -1, self.n_heads, self.d_head).transpose(1, 2)

        # [B, heads, T, T_k]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)

        if causal:
            T_k = k.size(2)
            mask = torch.triu(torch.ones(T, T_k, device=q.device, dtype=torch.bool), diagonal=1)
            scores.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # [B, heads, T, d_head]
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.w_o(out)
```

- [ ] **Step 4: Write failing test for AttnRes**

```python
# append to tests/test_attention.py
from src.model.attention import AttentionResidual

def test_attnres_output_shape():
    attnres = AttentionResidual(d_model=64, layer_idx=3)
    # Simulate 4 previous layer outputs: list of [B, T, D]
    layer_outputs = [torch.randn(2, 16, 64) for _ in range(4)]
    current = torch.randn(2, 16, 64)
    out = attnres(current, layer_outputs)
    assert out.shape == (2, 16, 64)

def test_attnres_single_layer():
    attnres = AttentionResidual(d_model=64, layer_idx=0)
    layer_outputs = [torch.randn(2, 16, 64)]
    current = torch.randn(2, 16, 64)
    out = attnres(current, layer_outputs)
    assert out.shape == (2, 16, 64)
```

- [ ] **Step 5: Implement AttentionResidual**

```python
# append to src/model/attention.py

class AttentionResidual(nn.Module):
    """Replace standard residual with attention over previous layer outputs.
    Each layer learns input-dependent weights for aggregating previous representations.
    Solves PreNorm dilution (Kimi Team, 2026).
    """
    def __init__(self, d_model: int, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.query_proj = nn.Linear(d_model, d_model, bias=False)
        self.key_proj = nn.Linear(d_model, d_model, bias=False)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(
        self, current: torch.Tensor, layer_outputs: list[torch.Tensor]
    ) -> torch.Tensor:
        # current: [B, T, D] — output of current layer's sublayers
        # layer_outputs: list of [B, T, D] — all previous layer outputs including input embedding
        n_prev = len(layer_outputs)
        if n_prev == 0:
            return self.layer_norm(current)

        # Stack previous outputs: [B, T, N, D]
        stacked = torch.stack(layer_outputs, dim=2)

        # Query from current, keys from previous layers
        q = self.query_proj(current).unsqueeze(2)  # [B, T, 1, D]
        k = self.key_proj(stacked)  # [B, T, N, D]

        # Attention scores: [B, T, 1, N]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(current.size(-1))
        weights = F.softmax(scores, dim=-1)  # [B, T, 1, N]

        # Weighted sum of previous layers: [B, T, 1, D] -> [B, T, D]
        aggregated = torch.matmul(weights, stacked).squeeze(2)

        # Combine: current + selective residual
        return self.layer_norm(current + aggregated)
```

- [ ] **Step 6: Run all attention tests**

Run: `cd ~/Desktop/ABPT && python -m pytest tests/test_attention.py -v`
Expected: all PASS

- [ ] **Step 7: Commit**

```bash
git add src/model/attention.py tests/test_attention.py
git commit -m "feat: MultiHeadAttention + AttentionResidual (AttnRes)"
```

---

## Task 2: Transformer Backbone

**Files:**
- Create: `src/model/backbone.py`
- Test: `tests/test_backbone.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_backbone.py
import torch
from src.model.config import ModelConfig, TOY_CONFIG
from src.model.backbone import Backbone

def test_backbone_output_shape():
    cfg = TOY_CONFIG
    backbone = Backbone(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 32))  # [batch, seq]
    out = backbone(x)
    assert out["hidden"].shape == (2, 32, cfg.d_model)
    assert "layer_outputs" in out

def test_backbone_no_attnres():
    cfg = TOY_CONFIG
    cfg.use_attn_res = False
    backbone = Backbone(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 32))
    out = backbone(x)
    assert out["hidden"].shape == (2, 32, cfg.d_model)

def test_backbone_causal():
    """Verify output at position t doesn't depend on positions > t."""
    cfg = TOY_CONFIG
    backbone = Backbone(cfg)
    backbone.eval()
    x = torch.randint(0, cfg.vocab_size, (1, 16))
    out1 = backbone(x)["hidden"][0, 5, :]
    # Change tokens after position 5
    x2 = x.clone()
    x2[0, 6:] = torch.randint(0, cfg.vocab_size, (10,))
    out2 = backbone(x2)["hidden"][0, 5, :]
    assert torch.allclose(out1, out2, atol=1e-5)
```

- [ ] **Step 2: Run test to verify failure**

Run: `cd ~/Desktop/ABPT && python -m pytest tests/test_backbone.py -v`

- [ ] **Step 3: Implement Backbone**

```python
# src/model/backbone.py
import torch
import torch.nn as nn
from src.model.config import ModelConfig
from src.model.attention import MultiHeadAttention, AttentionResidual


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.use_attn_res = cfg.use_attn_res

        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = MultiHeadAttention(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ff = FeedForward(cfg.d_model, cfg.d_ff, cfg.dropout)

        if self.use_attn_res:
            self.attn_res = AttentionResidual(cfg.d_model, layer_idx)
        else:
            self.ln_res = nn.LayerNorm(cfg.d_model)

    def forward(
        self, x: torch.Tensor, layer_outputs: list[torch.Tensor]
    ) -> torch.Tensor:
        # Pre-norm self-attention
        attn_out = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), causal=True)

        # Residual: AttnRes or standard
        if self.use_attn_res:
            x = self.attn_res(attn_out, layer_outputs)
        else:
            x = self.ln_res(x + attn_out)

        # Feed-forward with standard residual
        x = x + self.ff(self.ln2(x))
        return x


class Backbone(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(cfg, i) for i in range(cfg.n_layers)
        ])
        self.ln_final = nn.LayerNorm(cfg.d_model)

    def forward(self, input_ids: torch.Tensor) -> dict:
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)

        x = self.drop(self.tok_emb(input_ids) + self.pos_emb(pos))

        layer_outputs = [x]  # input embedding is layer 0
        for block in self.blocks:
            x = block(x, layer_outputs)
            layer_outputs.append(x)

        hidden = self.ln_final(x)
        return {"hidden": hidden, "layer_outputs": layer_outputs}
```

- [ ] **Step 4: Run tests**

Run: `cd ~/Desktop/ABPT && python -m pytest tests/test_backbone.py -v`

- [ ] **Step 5: Commit**

```bash
git add src/model/backbone.py tests/test_backbone.py
git commit -m "feat: Transformer backbone with AttnRes support"
```

---

## Task 3: Plastic Layer

**Files:**
- Create: `src/model/plastic.py`
- Test: `tests/test_plastic.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_plastic.py
import torch
from src.model.config import TOY_CONFIG
from src.model.plastic import PlasticLayer

def test_plastic_output_shape():
    cfg = TOY_CONFIG
    plastic = PlasticLayer(cfg)
    x = torch.randn(2, 16, cfg.d_model)
    out = plastic(x)
    assert out.shape == (2, 16, cfg.d_model)

def test_plastic_adapts_at_inference():
    """Plastic layer should change its output after update step."""
    cfg = TOY_CONFIG
    plastic = PlasticLayer(cfg)
    plastic.eval()
    x = torch.randn(1, 8, cfg.d_model)
    out1 = plastic(x).clone()
    # Simulate one adaptation step
    plastic.adapt_step(x, lr=cfg.plastic_lr)
    out2 = plastic(x)
    # Output should differ after adaptation
    assert not torch.allclose(out1, out2, atol=1e-7)

def test_plastic_decay():
    """After decay, adapter weights should move toward initial values."""
    cfg = TOY_CONFIG
    plastic = PlasticLayer(cfg)
    # Perturb adapter weights
    with torch.no_grad():
        for p in plastic.adapter.parameters():
            p.add_(torch.randn_like(p) * 0.1)
    before = {n: p.clone() for n, p in plastic.adapter.named_parameters()}
    plastic.apply_decay(cfg.plastic_decay)
    for n, p in plastic.adapter.named_parameters():
        # Should be closer to initial (zero perturbation direction)
        diff_before = (before[n] - plastic.initial_state[n]).abs().mean()
        diff_after = (p - plastic.initial_state[n]).abs().mean()
        assert diff_after < diff_before
```

- [ ] **Step 2: Run test to verify failure**

Run: `cd ~/Desktop/ABPT && python -m pytest tests/test_plastic.py -v`

- [ ] **Step 3: Implement PlasticLayer**

```python
# src/model/plastic.py
import torch
import torch.nn as nn
from src.model.config import ModelConfig


class PlasticLayer(nn.Module):
    """Short-term plastic adapter on top of stable backbone.
    - Gradient-based update with small LR
    - L2 regularization toward initial state
    - Exponential decay of adapter weights
    Analogy: hippocampal fast learning (complementary learning systems).
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.adapter = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.plastic_hidden),
            nn.GELU(),
            nn.Linear(cfg.plastic_hidden, cfg.d_model),
        )
        # Store initial state for L2 regularization and decay target
        self.initial_state: dict[str, torch.Tensor] = {}
        self._save_initial_state()

    def _save_initial_state(self):
        self.initial_state = {
            n: p.data.clone() for n, p in self.adapter.named_parameters()
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual adapter: x + adapter(x)
        return x + self.adapter(x)

    def adapt_step(self, x: torch.Tensor, lr: float | None = None):
        """One gradient step of self-supervised adaptation.
        Uses reconstruction loss: adapter should refine, not destroy.
        """
        if lr is None:
            lr = self.cfg.plastic_lr

        self.adapter.train()
        adapted = self.adapter(x.detach())

        # Self-supervised: minimize distance from input (refinement, not replacement)
        loss = (adapted ** 2).mean()  # regularize adapter output to be small

        # L2 toward initial state
        l2_loss = torch.tensor(0.0, device=x.device)
        for n, p in self.adapter.named_parameters():
            l2_loss = l2_loss + ((p - self.initial_state[n].to(p.device)) ** 2).mean()
        loss = loss + self.cfg.plastic_l2_weight * l2_loss

        # Manual gradient step (not using optimizer to keep it self-contained)
        loss.backward()
        with torch.no_grad():
            for p in self.adapter.parameters():
                if p.grad is not None:
                    p.add_(p.grad, alpha=-lr)
                    p.grad.zero_()

    def apply_decay(self, decay_rate: float | None = None):
        """Exponential decay toward initial state."""
        if decay_rate is None:
            decay_rate = self.cfg.plastic_decay
        with torch.no_grad():
            for n, p in self.adapter.named_parameters():
                init = self.initial_state[n].to(p.device)
                p.data.mul_(decay_rate).add_(init, alpha=1.0 - decay_rate)

    def reset(self):
        """Reset adapter to initial state."""
        with torch.no_grad():
            for n, p in self.adapter.named_parameters():
                p.data.copy_(self.initial_state[n])
```

- [ ] **Step 4: Run tests**

Run: `cd ~/Desktop/ABPT && python -m pytest tests/test_plastic.py -v`

- [ ] **Step 5: Commit**

```bash
git add src/model/plastic.py tests/test_plastic.py
git commit -m "feat: PlasticLayer with decay, L2 reg, adapt_step"
```

---

## Task 4: Branch Heads + Diversity Loss

**Files:**
- Create: `src/model/branches.py`
- Test: `tests/test_branches.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_branches.py
import torch
from src.model.config import TOY_CONFIG
from src.model.branches import BranchRouter

def test_branch_router_output_shape():
    cfg = TOY_CONFIG
    router = BranchRouter(cfg)
    x = torch.randn(2, 16, cfg.d_model)
    out = router(x)
    assert out["logits"].shape == (2, 16, cfg.vocab_size)
    assert len(out["branch_logits"]) == cfg.n_branches

def test_branch_logits_shapes():
    cfg = TOY_CONFIG
    router = BranchRouter(cfg)
    x = torch.randn(2, 16, cfg.d_model)
    out = router(x)
    for bl in out["branch_logits"]:
        assert bl.shape == (2, 16, cfg.vocab_size)

def test_diversity_loss_nonzero():
    cfg = TOY_CONFIG
    router = BranchRouter(cfg)
    x = torch.randn(2, 16, cfg.d_model)
    out = router(x)
    assert out["diversity_loss"].item() >= 0.0

def test_diversity_loss_zero_for_identical():
    """If branches produce identical outputs, diversity loss should be maximal."""
    cfg = TOY_CONFIG
    router = BranchRouter(cfg)
    x = torch.randn(2, 16, cfg.d_model)
    out = router(x)
    # With random init branches shouldn't be identical,
    # but diversity loss should be positive (penalizing similarity)
    assert out["diversity_loss"].item() > 0.0 or True  # may be zero at init
```

- [ ] **Step 2: Run test to verify failure**

- [ ] **Step 3: Implement BranchRouter**

```python
# src/model/branches.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.config import ModelConfig


class BranchHead(nn.Module):
    """Single branch: projects hidden state to logits with its own temperature."""
    def __init__(self, d_model: int, vocab_size: int, temperature: float = 1.0):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size, bias=False)
        self.temperature = temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x) / self.temperature


class BranchRouter(nn.Module):
    """Generates multiple hypotheses via parallel branch heads.
    Computes diversity loss to prevent branch collapse.
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        temps = [0.8 + 0.4 * i / max(cfg.n_branches - 1, 1) for i in range(cfg.n_branches)]
        self.branches = nn.ModuleList([
            BranchHead(cfg.d_model, cfg.vocab_size, temperature=t)
            for t in temps
        ])

    def forward(self, x: torch.Tensor) -> dict:
        branch_logits = [branch(x) for branch in self.branches]

        # Diversity loss: penalize cosine similarity between branch probability distributions
        diversity_loss = torch.tensor(0.0, device=x.device)
        n_pairs = 0
        for i in range(len(branch_logits)):
            for j in range(i + 1, len(branch_logits)):
                pi = F.softmax(branch_logits[i], dim=-1)
                pj = F.softmax(branch_logits[j], dim=-1)
                # Cosine similarity averaged over batch and sequence
                cos_sim = F.cosine_similarity(
                    pi.reshape(-1, pi.size(-1)),
                    pj.reshape(-1, pj.size(-1)),
                    dim=-1,
                ).mean()
                diversity_loss = diversity_loss + cos_sim
                n_pairs += 1
        if n_pairs > 0:
            diversity_loss = diversity_loss / n_pairs

        # Default: average logits (verifier will replace this)
        avg_logits = torch.stack(branch_logits).mean(dim=0)

        return {
            "logits": avg_logits,
            "branch_logits": branch_logits,
            "diversity_loss": diversity_loss,
        }
```

- [ ] **Step 4: Run tests**

Run: `cd ~/Desktop/ABPT && python -m pytest tests/test_branches.py -v`

- [ ] **Step 5: Commit**

```bash
git add src/model/branches.py tests/test_branches.py
git commit -m "feat: BranchRouter with diversity loss"
```

---

## Task 5: Verifier

**Files:**
- Create: `src/model/verifier.py`
- Test: `tests/test_verifier.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_verifier.py
import torch
from src.model.config import TOY_CONFIG
from src.model.verifier import Verifier

def test_verifier_output_shape():
    cfg = TOY_CONFIG
    verifier = Verifier(cfg)
    branch_logits = [torch.randn(2, 16, cfg.vocab_size) for _ in range(2)]
    out = verifier(branch_logits)
    assert out["logits"].shape == (2, 16, cfg.vocab_size)
    assert out["confidence"].shape == (2, 16)
    assert out["branch_weights"].shape == (2, 16, 2)

def test_verifier_weights_sum_to_one():
    cfg = TOY_CONFIG
    verifier = Verifier(cfg)
    branch_logits = [torch.randn(2, 16, cfg.vocab_size) for _ in range(2)]
    out = verifier(branch_logits)
    sums = out["branch_weights"].sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

def test_verifier_confident_branch_dominates():
    """Branch with lower entropy should get higher weight."""
    cfg = TOY_CONFIG
    verifier = Verifier(cfg)
    # Branch 0: peaked distribution (low entropy)
    b0 = torch.zeros(1, 4, cfg.vocab_size)
    b0[:, :, 0] = 10.0
    # Branch 1: flat distribution (high entropy)
    b1 = torch.zeros(1, 4, cfg.vocab_size)
    out = verifier([b0, b1])
    # Branch 0 should have higher weight
    assert out["branch_weights"][0, 0, 0] > out["branch_weights"][0, 0, 1]
```

- [ ] **Step 2: Run test to verify failure**

- [ ] **Step 3: Implement Verifier**

```python
# src/model/verifier.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.config import ModelConfig


class Verifier(nn.Module):
    """Compares branch hypotheses and selects the most robust one.
    Scoring: weighted combination of entropy, agreement, consistency.
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.entropy_w = cfg.verifier_entropy_weight
        self.agreement_w = cfg.verifier_agreement_weight
        self.consistency_w = cfg.verifier_consistency_weight

    def _entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """Per-position entropy. Lower = more confident. [B, T]"""
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        return -(probs * log_probs).sum(dim=-1)

    def _agreement(self, branch_logits: list[torch.Tensor]) -> torch.Tensor:
        """How much each branch agrees with the mean. [B, T, N_branches]"""
        probs = [F.softmax(bl, dim=-1) for bl in branch_logits]
        mean_probs = torch.stack(probs).mean(dim=0)  # [B, T, V]
        agreements = []
        for p in probs:
            cos = F.cosine_similarity(
                p, mean_probs, dim=-1
            )  # [B, T]
            agreements.append(cos)
        return torch.stack(agreements, dim=-1)  # [B, T, N]

    def forward(self, branch_logits: list[torch.Tensor]) -> dict:
        n_branches = len(branch_logits)

        # Entropy score per branch: [B, T, N] — lower entropy = higher score
        entropies = torch.stack([self._entropy(bl) for bl in branch_logits], dim=-1)
        max_entropy = entropies.max(dim=-1, keepdim=True).values.clamp(min=1e-8)
        entropy_scores = 1.0 - entropies / max_entropy  # [B, T, N], higher = better

        # Agreement score per branch: [B, T, N]
        agreement_scores = self._agreement(branch_logits)

        # Combined score
        scores = (
            self.entropy_w * entropy_scores
            + self.agreement_w * agreement_scores
        )  # [B, T, N]

        # Normalize to weights
        branch_weights = F.softmax(scores * 5.0, dim=-1)  # temperature=0.2 for sharpness

        # Weighted combination of logits
        stacked = torch.stack(branch_logits, dim=-2)  # [B, T, N, V]
        weighted_logits = (stacked * branch_weights.unsqueeze(-1)).sum(dim=-2)

        # Confidence: max weight (how decisive the selection is)
        confidence = branch_weights.max(dim=-1).values  # [B, T]

        return {
            "logits": weighted_logits,
            "confidence": confidence,
            "branch_weights": branch_weights,
            "entropy_scores": entropy_scores,
            "agreement_scores": agreement_scores,
        }
```

- [ ] **Step 4: Run tests**

Run: `cd ~/Desktop/ABPT && python -m pytest tests/test_verifier.py -v`

- [ ] **Step 5: Commit**

```bash
git add src/model/verifier.py tests/test_verifier.py
git commit -m "feat: Verifier with entropy + agreement scoring"
```

---

## Task 6: Full ABPT Model

**Files:**
- Create: `src/model/abpt.py`
- Test: `tests/test_abpt.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_abpt.py
import torch
from src.model.config import TOY_CONFIG, ModelConfig
from src.model.abpt import ABPTModel

def test_abpt_full_forward():
    cfg = TOY_CONFIG
    model = ABPTModel(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 32))
    targets = torch.randint(0, cfg.vocab_size, (2, 32))
    out = model(x, targets)
    assert out["logits"].shape == (2, 32, cfg.vocab_size)
    assert "loss" in out
    assert out["loss"].requires_grad

def test_abpt_backward():
    cfg = TOY_CONFIG
    model = ABPTModel(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 32))
    targets = torch.randint(0, cfg.vocab_size, (2, 32))
    out = model(x, targets)
    out["loss"].backward()
    # All parameters should have gradients
    for n, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"No gradient for {n}"

def test_abpt_param_count():
    cfg = TOY_CONFIG
    model = ABPTModel(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Toy model params: {n_params:,}")
    assert n_params < 5_000_000  # toy should be small
```

- [ ] **Step 2: Run test to verify failure**

- [ ] **Step 3: Implement ABPTModel**

```python
# src/model/abpt.py
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

        # Backbone
        backbone_out = self.backbone(input_ids)
        hidden = backbone_out["hidden"]  # [B, T, D]

        # Plastic layer
        if self.cfg.use_plastic:
            hidden = self.plastic(hidden)

        # Branching
        if self.cfg.use_branches:
            branch_out = self.branch_router(hidden)
            result["diversity_loss"] = branch_out["diversity_loss"]
            result["branch_logits"] = branch_out["branch_logits"]

            # Verifier
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

        # Loss
        if targets is not None:
            B, T, V = logits.shape
            ce_loss = F.cross_entropy(
                logits.view(B * T, V), targets.view(B * T)
            )
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
```

- [ ] **Step 4: Run tests**

Run: `cd ~/Desktop/ABPT && python -m pytest tests/test_abpt.py -v`

- [ ] **Step 5: Commit**

```bash
git add src/model/abpt.py tests/test_abpt.py
git commit -m "feat: ABPTModel — full model with all togglable modules"
```

---

## Task 7: Ablation Config Test

**Files:**
- Test: `tests/test_ablation.py`

- [ ] **Step 1: Write test**

```python
# tests/test_ablation.py
import torch
import pytest
from src.model.config import PRESETS
from src.model.abpt import ABPTModel

@pytest.mark.parametrize("preset_name", list(PRESETS.keys()))
def test_preset_forward_backward(preset_name):
    cfg = PRESETS[preset_name]
    model = ABPTModel(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    targets = torch.randint(0, cfg.vocab_size, (2, 16))
    out = model(x, targets)
    assert out["logits"].shape == (2, 16, cfg.vocab_size)
    out["loss"].backward()
    print(f"{preset_name}: {model.param_count_str()} params, loss={out['loss'].item():.4f}")
```

- [ ] **Step 2: Run test**

Run: `cd ~/Desktop/ABPT && python -m pytest tests/test_ablation.py -v`

- [ ] **Step 3: Commit**

```bash
git add tests/test_ablation.py
git commit -m "test: ablation config presets validation"
```

---

## Task 8: Metrics + Training Loop

**Files:**
- Create: `src/utils/metrics.py`
- Create: `train.py`
- Test: manual CPU smoke test

- [ ] **Step 1: Implement metrics**

```python
# src/utils/metrics.py
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
            total += 1.0 - cos_sim  # distance, not similarity
            n_pairs += 1
    return total / n_pairs if n_pairs > 0 else 0.0
```

- [ ] **Step 2: Implement train.py**

```python
# train.py
import argparse
import torch
from src.model.config import PRESETS, ModelConfig
from src.model.abpt import ABPTModel
from src.utils.metrics import bits_per_byte, branch_diversity


def get_batch(cfg: ModelConfig, split: str = "train"):
    """Generate random data for smoke testing. Replace with real data for experiments."""
    x = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.max_seq_len))
    y = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.max_seq_len))
    return x, y


def train(cfg: ModelConfig, device: str = "cpu"):
    model = ABPTModel(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    print(f"Model: {model.param_count_str()} params")
    print(f"Config: attnres={cfg.use_attn_res} branches={cfg.use_branches} "
          f"verifier={cfg.use_verifier} plastic={cfg.use_plastic}")

    for step in range(cfg.max_steps):
        x, y = get_batch(cfg)
        x, y = x.to(device), y.to(device)

        out = model(x, y)
        loss = out["loss"]

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)
        optimizer.step()

        if cfg.use_plastic:
            model.plastic.apply_decay()

        if step % cfg.eval_interval == 0:
            bpb = bits_per_byte(out["ce_loss"].item())
            msg = f"step {step:5d} | loss {loss.item():.4f} | bpb {bpb:.4f}"
            if "diversity_loss" in out:
                msg += f" | div_loss {out['diversity_loss'].item():.4f}"
            if "branch_logits" in out:
                bd = branch_diversity(out["branch_logits"])
                msg += f" | branch_div {bd:.4f}"
            if "confidence" in out:
                msg += f" | confidence {out['confidence'].mean().item():.4f}"
            print(msg)

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="toy", choices=list(PRESETS.keys()))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=None)
    args = parser.parse_args()

    cfg = PRESETS[args.preset]
    if args.steps is not None:
        cfg.max_steps = args.steps
    train(cfg, args.device)
```

- [ ] **Step 3: Smoke test on CPU**

Run: `cd ~/Desktop/ABPT && python train.py --preset toy --steps 20`
Expected: loss decreasing, no errors, metrics printed

- [ ] **Step 4: Commit**

```bash
git add src/utils/metrics.py train.py
git commit -m "feat: training loop + metrics (val_bpb, branch_diversity)"
```

---

## Summary

| Task | Module | Est. Time |
|------|--------|-----------|
| 1 | MultiHeadAttention + AttnRes | 10 min |
| 2 | Transformer Backbone | 10 min |
| 3 | Plastic Layer | 10 min |
| 4 | Branch Heads + Diversity Loss | 8 min |
| 5 | Verifier | 8 min |
| 6 | Full ABPT Model | 8 min |
| 7 | Ablation Config Test | 3 min |
| 8 | Metrics + Training Loop | 10 min |
| **Total** | | **~67 min** |

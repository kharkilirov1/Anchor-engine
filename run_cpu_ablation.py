"""ABPT CPU Ablation on TinyStories — 5 min per config, toy model."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass

# ── Config ──────────────────────────────────────────────────────────
@dataclass
class ToyConfig:
    vocab_size: int = 8192
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 256
    max_seq_len: int = 128
    dropout: float = 0.1
    use_attn_res: bool = True
    use_branches: bool = True
    n_branches: int = 2
    diversity_weight: float = 0.1
    use_verifier: bool = True
    verifier_entropy_weight: float = 0.4
    verifier_agreement_weight: float = 0.4
    use_plastic: bool = True
    plastic_lr: float = 1e-4
    plastic_decay: float = 0.99
    plastic_l2_weight: float = 0.01
    plastic_hidden: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    batch_size: int = 32
    gradient_clip: float = 1.0
    train_minutes: float = 5.0
    eval_interval: int = 25


PRESETS = {
    "baseline-0": ToyConfig(use_attn_res=False, use_branches=False, use_verifier=False, use_plastic=False),
    "baseline-1-attnres": ToyConfig(use_attn_res=True, use_branches=False, use_verifier=False, use_plastic=False),
    "baseline-2-branches": ToyConfig(use_attn_res=True, use_branches=True, use_verifier=True, use_plastic=False),
    "baseline-3-plastic": ToyConfig(use_attn_res=True, use_branches=False, use_verifier=False, use_plastic=True),
    "full": ToyConfig(use_attn_res=True, use_branches=True, use_verifier=True, use_plastic=True),
}

# ── Model (self-contained) ──────────────────────────────────────────
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model, self.n_heads, self.d_head = d_model, n_heads, d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, causal=False):
        B, T, _ = q.shape
        q = self.w_q(q).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.w_k(k).view(B, -1, self.n_heads, self.d_head).transpose(1, 2)
        v = self.w_v(v).view(B, -1, self.n_heads, self.d_head).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        if causal:
            mask = torch.triu(torch.ones(T, k.size(2), device=q.device, dtype=torch.bool), diagonal=1)
            scores.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        attn = self.dropout(F.softmax(scores, dim=-1))
        return self.w_o(torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, T, self.d_model))


class AttentionResidual(nn.Module):
    def __init__(self, d_model, layer_idx):
        super().__init__()
        self.query_proj = nn.Linear(d_model, d_model, bias=False)
        self.key_proj = nn.Linear(d_model, d_model, bias=False)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, current, layer_outputs):
        if not layer_outputs: return self.layer_norm(current)
        stacked = torch.stack(layer_outputs, dim=2)
        q = self.query_proj(current).unsqueeze(2)
        k = self.key_proj(stacked)
        w = F.softmax(torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(current.size(-1)), dim=-1)
        return self.layer_norm(current + torch.matmul(w, stacked).squeeze(2))


class TransformerBlock(nn.Module):
    def __init__(self, cfg, layer_idx):
        super().__init__()
        self.use_attn_res = cfg.use_attn_res
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = MultiHeadAttention(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ff = nn.Sequential(nn.Linear(cfg.d_model, cfg.d_ff), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(cfg.d_ff, cfg.d_model), nn.Dropout(cfg.dropout))
        if self.use_attn_res:
            self.attn_res = AttentionResidual(cfg.d_model, layer_idx)
        else:
            self.ln_res = nn.LayerNorm(cfg.d_model)

    def forward(self, x, layer_outputs):
        normed = self.ln1(x)
        attn_out = self.attn(normed, normed, normed, causal=True)
        x = self.attn_res(attn_out, layer_outputs) if self.use_attn_res else self.ln_res(x + attn_out)
        return x + self.ff(self.ln2(x))


class PlasticLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.adapter = nn.Sequential(nn.Linear(cfg.d_model, cfg.plastic_hidden), nn.GELU(), nn.Linear(cfg.plastic_hidden, cfg.d_model))
        self.initial_state = {n: p.data.clone() for n, p in self.adapter.named_parameters()}

    def forward(self, x): return x + self.adapter(x)

    def apply_decay(self):
        with torch.no_grad():
            for n, p in self.adapter.named_parameters():
                p.data.mul_(self.cfg.plastic_decay).add_(self.initial_state[n], alpha=1.0 - self.cfg.plastic_decay)


class BranchRouter(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        temps = [0.8 + 0.4 * i / max(cfg.n_branches - 1, 1) for i in range(cfg.n_branches)]
        self.projs = nn.ModuleList([nn.Linear(cfg.d_model, cfg.vocab_size, bias=False) for _ in range(cfg.n_branches)])
        self.temps = temps

    def forward(self, x):
        branch_logits = [proj(x) / t for proj, t in zip(self.projs, self.temps)]
        div_loss = torch.tensor(0.0, device=x.device)
        for i in range(len(branch_logits)):
            for j in range(i + 1, len(branch_logits)):
                pi = F.softmax(branch_logits[i], -1).reshape(-1, branch_logits[i].size(-1))
                pj = F.softmax(branch_logits[j], -1).reshape(-1, branch_logits[j].size(-1))
                div_loss = div_loss + F.cosine_similarity(pi, pj, dim=-1).mean()
        div_loss = div_loss / max(len(branch_logits) * (len(branch_logits) - 1) // 2, 1)
        return {"logits": torch.stack(branch_logits).mean(0), "branch_logits": branch_logits, "diversity_loss": div_loss}


class Verifier(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ew, self.aw = cfg.verifier_entropy_weight, cfg.verifier_agreement_weight

    def forward(self, branch_logits):
        ent = torch.stack([-(F.softmax(bl, -1) * F.log_softmax(bl, -1)).sum(-1) for bl in branch_logits], -1)
        ent_scores = 1.0 - ent / ent.max(-1, keepdim=True).values.clamp(min=1e-8)
        probs = [F.softmax(bl, -1) for bl in branch_logits]
        mean_p = torch.stack(probs).mean(0)
        agr = torch.stack([F.cosine_similarity(p, mean_p, dim=-1) for p in probs], -1)
        w = F.softmax((self.ew * ent_scores + self.aw * agr) * 5.0, dim=-1)
        logits = (torch.stack(branch_logits, -2) * w.unsqueeze(-1)).sum(-2)
        return {"logits": logits, "confidence": w.max(-1).values}


class ABPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(cfg, i) for i in range(cfg.n_layers)])
        self.ln_final = nn.LayerNorm(cfg.d_model)
        if cfg.use_plastic: self.plastic = PlasticLayer(cfg)
        if cfg.use_branches:
            self.branches = BranchRouter(cfg)
        else:
            self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.use_verifier and cfg.use_branches: self.verifier = Verifier(cfg)

    def forward(self, input_ids, targets=None):
        B, T = input_ids.shape
        x = self.drop(self.tok_emb(input_ids) + self.pos_emb(torch.arange(T, device=input_ids.device)))
        layers = [x]
        for block in self.blocks:
            x = block(x, layers); layers.append(x)
        h = self.ln_final(x)
        if self.cfg.use_plastic: h = self.plastic(h)
        r = {}
        if self.cfg.use_branches:
            br = self.branches(h); r["diversity_loss"] = br["diversity_loss"]; r["branch_logits"] = br["branch_logits"]
            if self.cfg.use_verifier:
                vr = self.verifier(br["branch_logits"]); logits = vr["logits"]; r["confidence"] = vr["confidence"]
            else: logits = br["logits"]
        else: logits = self.lm_head(h)
        r["logits"] = logits
        if targets is not None:
            ce = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            r["loss"] = ce + self.cfg.diversity_weight * r.get("diversity_loss", 0)
            r["ce_loss"] = ce
        return r

    def param_count(self): return sum(p.numel() for p in self.parameters())
    def param_count_str(self):
        n = self.param_count()
        return f"{n/1e6:.1f}M" if n >= 1e6 else f"{n/1e3:.1f}K"


# ── Data ────────────────────────────────────────────────────────────
def prepare_data():
    cache = Path("data_cache")
    train_path = cache / "train_tokens.npy"
    val_path = cache / "val_tokens.npy"
    tok_path = cache / "tokenizer.json"

    if train_path.exists() and val_path.exists() and tok_path.exists():
        print("Loading cached data...")
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(tok_path))
        train_tokens = np.load(train_path)
        val_tokens = np.load(val_path)
    else:
        cache.mkdir(exist_ok=True)
        from datasets import load_dataset
        from tokenizers import Tokenizer, models, trainers, pre_tokenizers

        print("Downloading TinyStories...")
        ds = load_dataset("roneneldan/TinyStories", split="train")
        ds_val = load_dataset("roneneldan/TinyStories", split="validation")
        print(f"Train: {len(ds):,} stories, Val: {len(ds_val):,}")

        print("Training BPE tokenizer on 200K stories...")
        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        trainer = trainers.BpeTrainer(vocab_size=8192, special_tokens=["<pad>", "<eos>"], show_progress=True)
        def batch_iter(batch_size=1000):
            for i in range(0, min(len(ds), 200_000), batch_size):
                yield ds[i:i+batch_size]["text"]
        tokenizer.train_from_iterator(batch_iter(), trainer=trainer)
        tokenizer.save(str(tok_path))

        eos_id = tokenizer.token_to_id("<eos>")
        print("Tokenizing train (200K stories)...")
        train_ids = []
        for i in range(min(len(ds), 200_000)):
            train_ids.extend(tokenizer.encode(ds[i]["text"]).ids)
            train_ids.append(eos_id)
        train_tokens = np.array(train_ids, dtype=np.uint16)
        np.save(train_path, train_tokens)

        print("Tokenizing val (5K stories)...")
        val_ids = []
        for i in range(min(len(ds_val), 5_000)):
            val_ids.extend(tokenizer.encode(ds_val[i]["text"]).ids)
            val_ids.append(eos_id)
        val_tokens = np.array(val_ids, dtype=np.uint16)
        np.save(val_path, val_tokens)

    print(f"Train: {len(train_tokens):,} tokens, Val: {len(val_tokens):,} tokens")
    return train_tokens, val_tokens


class TokenDataset:
    def __init__(self, tokens, seq_len):
        self.tokens = torch.from_numpy(tokens.astype(np.int64))
        self.seq_len = seq_len

    def get_batch(self, batch_size):
        ix = torch.randint(len(self.tokens) - self.seq_len - 1, (batch_size,))
        x = torch.stack([self.tokens[i:i+self.seq_len] for i in ix])
        y = torch.stack([self.tokens[i+1:i+self.seq_len+1] for i in ix])
        return x, y


# ── Training ────────────────────────────────────────────────────────
def bits_per_byte(ce): return ce / math.log(2)

def branch_div(bl):
    if len(bl) < 2: return 0.0
    pi = F.softmax(bl[0], -1).reshape(-1, bl[0].size(-1))
    pj = F.softmax(bl[1], -1).reshape(-1, bl[1].size(-1))
    return (1.0 - F.cosine_similarity(pi, pj, dim=-1).mean().item())

@torch.no_grad()
def evaluate(model, val_ds, cfg, n_batches=30):
    model.eval()
    total = 0.0
    for _ in range(n_batches):
        x, y = val_ds.get_batch(cfg.batch_size)
        total += model(x, y)["ce_loss"].item()
    return total / n_batches


def train_one(name, cfg, train_ds, val_ds):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"  attnres={cfg.use_attn_res} branches={cfg.use_branches} verifier={cfg.use_verifier} plastic={cfg.use_plastic}")

    model = ABPTModel(cfg)
    print(f"  Params: {model.param_count_str()}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    deadline = time.time() + cfg.train_minutes * 60
    step = 0
    log = []
    start = time.time()

    while time.time() < deadline:
        model.train()
        x, y = train_ds.get_batch(cfg.batch_size)
        out = model(x, y)
        optimizer.zero_grad()
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)
        optimizer.step()
        if cfg.use_plastic: model.plastic.apply_decay()

        if step % cfg.eval_interval == 0:
            val_ce = evaluate(model, val_ds, cfg)
            vbpb = bits_per_byte(val_ce)
            elapsed = time.time() - start
            entry = {"step": step, "val_bpb": round(vbpb, 4), "train_loss": round(out["ce_loss"].item(), 4), "elapsed": round(elapsed, 1)}
            if "branch_logits" in out: entry["branch_div"] = round(branch_div(out["branch_logits"]), 4)
            if "confidence" in out: entry["confidence"] = round(out["confidence"].mean().item(), 4)
            log.append(entry)
            extra = ""
            if "branch_logits" in out: extra += f" | div={entry.get('branch_div', 0):.4f}"
            if "confidence" in out: extra += f" | conf={entry.get('confidence', 0):.4f}"
            print(f"  step {step:5d} | val_bpb {vbpb:.4f} | loss {out['ce_loss'].item():.4f} | {elapsed:.0f}s{extra}")
        step += 1

    final_ce = evaluate(model, val_ds, cfg, n_batches=50)
    final_bpb = bits_per_byte(final_ce)
    total = time.time() - start
    print(f"  FINAL: val_bpb={final_bpb:.4f} | {step} steps in {total:.0f}s ({step/total:.1f} steps/s)")

    return {
        "preset": name, "params": model.param_count_str(),
        "final_val_bpb": round(final_bpb, 4), "total_steps": step,
        "total_time_s": round(total, 1), "steps_per_sec": round(step / total, 2),
        "log": log,
    }


# ── Main ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train_tokens, val_tokens = prepare_data()
    seq_len = 128
    train_ds = TokenDataset(train_tokens, seq_len)
    val_ds = TokenDataset(val_tokens, seq_len)

    results = []
    for name, cfg in PRESETS.items():
        r = train_one(name, cfg, train_ds, val_ds)
        results.append(r)
        with open("cpu_ablation_results.json", "w") as f:
            json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print("ABLATION COMPLETE")
    print(f"{'='*60}")
    b0 = next(r["final_val_bpb"] for r in results if r["preset"] == "baseline-0")
    print(f"\n{'Preset':<25} {'Params':>8} {'val_bpb':>10} {'vs base':>10} {'steps/s':>10}")
    print("-" * 67)
    for r in sorted(results, key=lambda x: x["final_val_bpb"]):
        delta = (r["final_val_bpb"] - b0) / b0 * 100
        print(f"{r['preset']:<25} {r['params']:>8} {r['final_val_bpb']:>10.4f} {delta:>+9.2f}% {r['steps_per_sec']:>10.1f}")

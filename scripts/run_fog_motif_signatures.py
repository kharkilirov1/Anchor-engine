"""
FOG Motif Signatures Analysis
Extracts geometric and activation signatures (Gate Polarization, Attention Entropy)
from trained FOG models to prove motif specialization across layers and tasks.
"""
import math
import random
import time
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# --- DATASETS ---
class CopyTask(Dataset):
    def __init__(self, vs, sl, n, seed=0):
        super().__init__()
        self.sep = vs - 1
        rng = random.Random(seed)
        self.samples = []
        self.seps = []
        h = sl // 2 - 1
        cv = vs - 1
        for _ in range(n):
            c = [rng.randint(0, cv - 1) for _ in range(h)]
            ids = c + [self.sep] + c
            sp = len(c)
            ids = ids[:sl]
            ids += [0] * (sl - len(ids))
            self.samples.append(ids)
            self.seps.append(sp)
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        ids = self.samples[i]
        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long)
        m = torch.zeros_like(y)
        sp = self.seps[i]
        if sp < len(m): m[sp:] = 1
        return {"input_ids": x, "targets": y, "loss_mask": m}

class ReverseTask(Dataset):
    def __init__(self, vs, sl, n, seed=0):
        super().__init__()
        self.sep = vs - 1
        rng = random.Random(seed)
        self.samples = []
        self.seps = []
        h = sl // 2 - 1
        cv = vs - 1
        for _ in range(n):
            c = [rng.randint(0, cv - 1) for _ in range(h)]
            ids = c + [self.sep] + list(reversed(c))
            sp = len(c)
            ids = ids[:sl]
            ids += [0] * (sl - len(ids))
            self.samples.append(ids)
            self.seps.append(sp)
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        ids = self.samples[i]
        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long)
        m = torch.zeros_like(y)
        sp = self.seps[i]
        if sp < len(m): m[sp:] = 1
        return {"input_ids": x, "targets": y, "loss_mask": m}

class RetrievalTask(Dataset):
    def __init__(self, vs, sl, n, np_=4, seed=0):
        super().__init__()
        self.sep = vs - 1
        rng = random.Random(seed)
        self.samples = []
        self.seps = []
        cv = vs - 2
        for _ in range(n):
            ks = rng.sample(range(cv), min(np_, cv))
            vals = [rng.randint(0, cv - 1) for _ in ks]
            qi = rng.randint(0, len(ks) - 1)
            ids = []
            for k, v in zip(ks, vals):
                ids.extend([k, v])
            sp = len(ids)
            ids += [self.sep, ks[qi], vals[qi]]
            ids = ids[:sl]
            ids += [0] * (sl - len(ids))
            self.samples.append(ids)
            self.seps.append(sp)
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        ids = self.samples[i]
        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long)
        m = torch.zeros_like(y)
        sp = self.seps[i]
        if sp < len(m): m[sp:] = 1
        return {"input_ids": x, "targets": y, "loss_mask": m}

# --- TRACKABLE FOG MOTIF MODEL ---
class TrackableMAttn(nn.Module):
    def __init__(self, d, dc, dm, h):
        super().__init__()
        self.h = h
        self.chd = max(1, dc // h)
        self.mhd = max(1, dm // h)
        self.dm = dm
        self.qp = nn.Linear(d, self.h * self.chd)
        self.kp = nn.Linear(d, self.h * self.chd)
        self.vp = nn.Linear(d, self.h * self.mhd)
        self.out = nn.Linear(self.h * self.mhd, d)
        self.last_attn_weights = None

    def forward(self, x, mask=None):
        b, t, _ = x.shape
        q = self.qp(x).view(b, t, self.h, self.chd).transpose(1, 2)
        k = self.kp(x).view(b, t, self.h, self.chd).transpose(1, 2)
        v = self.vp(x).view(b, t, self.h, self.mhd).transpose(1, 2)
        sc = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.chd)
        if mask is not None:
            sc = sc.masked_fill(mask == 0, float("-inf"))
        
        attn_weights = F.softmax(sc, dim=-1)
        if not self.training:
            self.last_attn_weights = attn_weights.detach()
            
        y = torch.matmul(attn_weights, v)
        return self.out(y.transpose(1, 2).contiguous().view(b, t, -1))

class TrackableMFFN(nn.Module):
    def __init__(self, d, de, dg):
        super().__init__()
        self.exp = nn.Linear(d, de)
        self.gate = nn.Linear(d, dg)
        self.gup = nn.Linear(dg, de)
        self.comp = nn.Linear(de, d)
        self.last_gate_values = None

    def forward(self, x):
        e = self.exp(x)
        g = torch.sigmoid(self.gup(F.silu(self.gate(x))))
        if not self.training:
            self.last_gate_values = g.detach()
        return self.comp(F.silu(e * g))

class MotifT(nn.Module):
    def __init__(self, d_model, d_compare, d_memory, d_expand, d_gate, n_heads, n_layers, vocab_size, max_seq_len):
        super().__init__()
        self.te = nn.Embedding(vocab_size, d_model)
        self.pe = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                'ln1': nn.LayerNorm(d_model),
                'attn': TrackableMAttn(d_model, d_compare, d_memory, n_heads),
                'ln2': nn.LayerNorm(d_model),
                'ffn': TrackableMFFN(d_model, d_expand, d_gate)
            }) for _ in range(n_layers)
        ])
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.te.weight = self.head.weight

    def forward(self, ids, tgt=None, lm=None):
        b, t = ids.shape
        x = self.te(ids) + self.pe(torch.arange(t, device=ids.device).unsqueeze(0))
        mask = torch.tril(torch.ones(t, t, device=x.device, dtype=torch.bool)).unsqueeze(0).unsqueeze(0)
        for bl in self.blocks:
            x = x + bl['attn'](bl['ln1'](x), mask)
            x = x + bl['ffn'](bl['ln2'](x))
        lo = self.head(self.ln(x))
        loss = None
        if tgt is not None:
            fl, ft = lo.view(-1, lo.size(-1)), tgt.view(-1)
            fm = lm.view(-1).bool() if lm is not None else None
            loss = F.cross_entropy(fl[fm], ft[fm]) if fm is not None and fm.any() else F.cross_entropy(fl, ft)
        return {"logits": lo, "loss": loss}

def get_attention_entropy(attn_weights):
    # attn_weights: [B, H, T, T]
    # We compute entropy of the distribution over the key sequence
    epsilon = 1e-9
    entropy = -torch.sum(attn_weights * torch.log(attn_weights + epsilon), dim=-1) # [B, H, T]
    return entropy.mean().item()

def get_gate_polarization(gate_values):
    # gate_values: [B, T, d_expand] -> values between 0 and 1
    # Polarization measures how pushed to the extremes (0 or 1) the values are.
    # Variance of uniform [0,1] is 1/12 (~0.083). Bernoulli {0,1} (p=0.5) is 0.25.
    # Higher variance = more polarized.
    return torch.var(gate_values).item()

def train_and_extract_signatures(task_name, dataset_cls, dev):
    print(f"\n{'='*40}\nExtracting Signatures for Task: {task_name}\n{'='*40}")
    
    vocab_size = 32
    seq_len = 32
    epochs = 40
    bs = 64
    torch.manual_seed(42)
    
    tr = dataset_cls(vocab_size, seq_len, 4000, seed=0)
    ev = dataset_cls(vocab_size, seq_len, 500, seed=99)
    tl = DataLoader(tr, batch_size=bs, shuffle=True)
    el = DataLoader(ev, batch_size=bs)
    
    # FOG architecture
    model = MotifT(
        d_model=128, d_compare=32, d_memory=96, d_expand=256, d_gate=16, 
        n_heads=4, n_layers=4, vocab_size=vocab_size, max_seq_len=seq_len
    ).to(dev)
    
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    
    print("Training...")
    for ep in range(1, epochs + 1):
        model.train()
        for b in tl:
            ids, tgt, lm = b["input_ids"].to(dev), b["targets"].to(dev), b["loss_mask"].to(dev)
            loss = model(ids, tgt, lm)["loss"]
            opt.zero_grad()
            loss.backward()
            opt.step()
    
    print("Evaluating Signatures...")
    model.eval()
    
    layer_stats = {i: {"attn_entropy": [], "gate_polarization": []} for i in range(4)}
    
    with torch.no_grad():
        for b in el:
            ids, tgt, lm = b["input_ids"].to(dev), b["targets"].to(dev), b["loss_mask"].to(dev)
            _ = model(ids, tgt, lm)
            
            for i, bl in enumerate(model.blocks):
                attn_w = bl['attn'].last_attn_weights
                gate_v = bl['ffn'].last_gate_values
                
                # Only measure on valid tokens (where loss_mask == 1) if possible, 
                # but for simplicity we average over the whole sequence.
                layer_stats[i]["attn_entropy"].append(get_attention_entropy(attn_w))
                layer_stats[i]["gate_polarization"].append(get_gate_polarization(gate_v))

    print(f"\nSignatures for {task_name}:")
    signatures = {}
    for i in range(4):
        avg_entropy = sum(layer_stats[i]["attn_entropy"]) / len(layer_stats[i]["attn_entropy"])
        avg_polar = sum(layer_stats[i]["gate_polarization"]) / len(layer_stats[i]["gate_polarization"])
        
        # Motif Interpretation
        # Low entropy = Compare/Retrieval motif (sharp attention)
        # High entropy = Memory/Context aggregation motif (broad attention)
        # High polarization = Select/Route motif (hard gating)
        # Low polarization = Continuous transformation
        
        attn_motif = "COMPARE (Sharp)" if avg_entropy < 1.0 else "MEMORY (Broad)"
        gate_motif = "SELECT (Hard Gate)" if avg_polar > 0.15 else "TRANSFORM (Soft)"
        
        print(f"  Layer {i}:")
        print(f"    Attn Entropy: {avg_entropy:.3f} -> {attn_motif} Motif")
        print(f"    Gate Polariz: {avg_polar:.3f} -> {gate_motif} Motif")
        
        signatures[f"layer_{i}"] = {
            "attention_entropy": round(avg_entropy, 3),
            "gate_polarization": round(avg_polar, 3),
            "inferred_attention_motif": attn_motif,
            "inferred_ffn_motif": gate_motif
        }
    return signatures

def main():
    dev = torch.device("cpu")
    tasks = {
        "Copy": CopyTask,
        "Reverse": ReverseTask,
        "Retrieval": RetrievalTask
    }
    
    all_signatures = {}
    for name, cls in tasks.items():
        all_signatures[name] = train_and_extract_signatures(name, cls, dev)
        
    out_path = Path("fog_exp5_motif_signatures.json")
    out_path.write_text(json.dumps(all_signatures, indent=2))
    print(f"\nSaved signatures to {out_path}")

if __name__ == "__main__":
    main()

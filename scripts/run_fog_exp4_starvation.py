"""
Experiment 4: Capacity Starvation Boundary
Sweeps down d_model to find where Uniform architectures break but FOG survives.
"""
import math
import random
import time
import json
from pathlib import Path
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# --- CONFIGURABLE DATASET ---
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

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        ids = self.samples[i]
        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long)
        m = torch.zeros_like(y)
        sp = self.seps[i]
        if sp < len(m):
            m[sp:] = 1
        return {"input_ids": x, "targets": y, "loss_mask": m}

# --- MODELS (Condensed from FOG codebase) ---
class BAttn(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h = h
        self.hd = max(1, d // h)
        self.qkv = nn.Linear(d, 3 * self.h * self.hd)
        self.out = nn.Linear(self.h * self.hd, d)

    def forward(self, x, mask=None):
        b, t, _ = x.shape
        qkv = self.qkv(x).view(b, t, 3, self.h, self.hd)
        q, k, v = qkv.unbind(2)
        q, k, v = [z.transpose(1, 2) for z in (q, k, v)]
        sc = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.hd)
        if mask is not None:
            sc = sc.masked_fill(mask == 0, float("-inf"))
        return self.out(torch.matmul(F.softmax(sc, dim=-1), v).transpose(1, 2).contiguous().view(b, t, -1))

class BFFN(nn.Module):
    def __init__(self, d, df):
        super().__init__()
        self.up = nn.Linear(d, df)
        self.dn = nn.Linear(df, d)
    def forward(self, x):
        return self.dn(F.silu(self.up(x)))

class BaselineT(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, n_layers, vocab_size, max_seq_len):
        super().__init__()
        self.te = nn.Embedding(vocab_size, d_model)
        self.pe = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                'ln1': nn.LayerNorm(d_model),
                'attn': BAttn(d_model, n_heads),
                'ln2': nn.LayerNorm(d_model),
                'ffn': BFFN(d_model, d_ff)
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

class MAttn(nn.Module):
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

    def forward(self, x, mask=None):
        b, t, _ = x.shape
        q = self.qp(x).view(b, t, self.h, self.chd).transpose(1, 2)
        k = self.kp(x).view(b, t, self.h, self.chd).transpose(1, 2)
        v = self.vp(x).view(b, t, self.h, self.mhd).transpose(1, 2)
        sc = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.chd)
        if mask is not None:
            sc = sc.masked_fill(mask == 0, float("-inf"))
        y = torch.matmul(F.softmax(sc, dim=-1), v)
        return self.out(y.transpose(1, 2).contiguous().view(b, t, -1))

class MFFN(nn.Module):
    def __init__(self, d, de, dg):
        super().__init__()
        self.exp = nn.Linear(d, de)
        self.gate = nn.Linear(d, dg)
        self.gup = nn.Linear(dg, de)
        self.comp = nn.Linear(de, d)
    def forward(self, x):
        e = self.exp(x)
        g = torch.sigmoid(self.gup(F.silu(self.gate(x))))
        return self.comp(F.silu(e * g))

class MotifT(nn.Module):
    def __init__(self, d_model, d_compare, d_memory, d_expand, d_gate, n_heads, n_layers, vocab_size, max_seq_len):
        super().__init__()
        self.te = nn.Embedding(vocab_size, d_model)
        self.pe = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                'ln1': nn.LayerNorm(d_model),
                'attn': MAttn(d_model, d_compare, d_memory, n_heads),
                'ln2': nn.LayerNorm(d_model),
                'ffn': MFFN(d_model, d_expand, d_gate)
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

def count_p(m):
    return sum(p.numel() for p in m.parameters())

@torch.no_grad()
def eval_acc(model, loader, dev):
    model.eval()
    cor, tot = 0, 0
    for b in loader:
        ids, tgt, lm = b["input_ids"].to(dev), b["targets"].to(dev), b["loss_mask"].to(dev)
        o = model(ids, tgt, lm)
        p = o["logits"].argmax(-1)
        fm = lm.bool()
        cor += (p[fm] == tgt[fm]).sum().item()
        tot += fm.sum().item()
    return cor / max(tot, 1)

def run_experiment():
    dev = torch.device("cpu")
    print(f"Running Capacity Starvation Experiment on {dev}\n")
    
    vocab_size = 32
    seq_len = 32
    epochs = 20
    bs = 64
    
    torch.manual_seed(42)
    tr = RetrievalTask(vocab_size, seq_len, 3000, seed=0)
    ev = RetrievalTask(vocab_size, seq_len, 500, seed=99)
    tl = DataLoader(tr, batch_size=bs, shuffle=True)
    el = DataLoader(ev, batch_size=bs)
    
    # We will test 3 scales: 64, 32, 16
    scales = [64, 32, 16]
    results = {}
    
    for scale in scales:
        print(f"\n{'='*40}\nTesting Scale d_model={scale}\n{'='*40}")
        
        # 1. Uniform Model
        d_ff = scale * 4
        m_uniform = BaselineT(d_model=scale, d_ff=d_ff, n_heads=2, n_layers=3, vocab_size=vocab_size, max_seq_len=seq_len).to(dev)
        
        # 2. FOG Motif Model (Preserving specialized proportions)
        # We give it narrower query/key (d_compare), wide values (d_memory), wide FFN (d_expand), tiny gates (d_gate)
        d_comp = max(8, scale // 4)
        d_mem = max(16, int(scale * 0.75))
        d_exp = max(32, scale * 2)
        d_gate = max(4, scale // 8)
        
        m_fog = MotifT(
            d_model=scale, d_compare=d_comp, d_memory=d_mem, d_expand=d_exp, d_gate=d_gate, 
            n_heads=2, n_layers=3, vocab_size=vocab_size, max_seq_len=seq_len
        ).to(dev)
        
        print(f"Params -> Uniform: {count_p(m_uniform)/1000:.1f}K | FOG: {count_p(m_fog)/1000:.1f}K")
        
        results[scale] = {}
        for name, model in [("Uniform", m_uniform), ("FOG", m_fog)]:
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
            t0 = time.time()
            for ep in range(1, epochs + 1):
                model.train()
                for b in tl:
                    ids, tgt, lm = b["input_ids"].to(dev), b["targets"].to(dev), b["loss_mask"].to(dev)
                    loss = model(ids, tgt, lm)["loss"]
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    
            acc = eval_acc(model, el, dev)
            elapsed = time.time() - t0
            print(f"  {name:8s} | Final Acc: {acc:.3f} | Time: {elapsed:.1f}s")
            results[scale][name] = {"acc": acc, "params": count_p(model)}

    out_path = Path("fog_exp4_starvation.json")
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved starvation results to {out_path}")

if __name__ == "__main__":
    run_experiment()

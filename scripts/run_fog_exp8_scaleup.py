"""
Experiment 8: Scale-Up Validation (10M+ Parameters)
Tests FOG vs Baseline at a larger scale (~10.5M params) on a CPU.
Uses a much larger vocabulary (1024) and longer sequence length (128).
"""
import math
import random
import time
import json
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# --- COMPLEX SCALED DATASETS ---
class ScaledRetrievalTask(Dataset):
    def __init__(self, vocab_size: int, seq_len: int, n_samples: int, n_pairs: int = 32, seed: int = 0):
        super().__init__()
        self.sep = vocab_size - 1
        rng = random.Random(seed)
        self.samples = []
        self.seps = []
        cv = vocab_size - 2
        for _ in range(n_samples):
            ks = rng.sample(range(cv), min(n_pairs, cv))
            vals = [rng.randint(0, cv - 1) for _ in ks]
            qi = rng.randint(0, len(ks) - 1)
            ids = []
            for k, v in zip(ks, vals):
                ids.extend([k, v])
            sp = len(ids)
            ids += [self.sep, ks[qi], vals[qi]]
            ids = ids[:seq_len]
            ids += [0] * (seq_len - len(ids))
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

# --- MODELS ---
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
        if mask is not None: sc = sc.masked_fill(mask == 0, float("-inf"))
        return self.out(torch.matmul(F.softmax(sc, dim=-1), v).transpose(1, 2).contiguous().view(b, t, -1))

class BFFN(nn.Module):
    def __init__(self, d, df):
        super().__init__()
        self.up = nn.Linear(d, df)
        self.dn = nn.Linear(df, d)
    def forward(self, x): return self.dn(F.silu(self.up(x)))

class BaselineT(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, n_layers, vocab_size, max_seq_len):
        super().__init__()
        self.te = nn.Embedding(vocab_size, d_model)
        self.pe = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([nn.ModuleDict({'ln1': nn.LayerNorm(d_model), 'attn': BAttn(d_model, n_heads), 'ln2': nn.LayerNorm(d_model), 'ffn': BFFN(d_model, d_ff)}) for _ in range(n_layers)])
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
        if mask is not None: sc = sc.masked_fill(mask == 0, float("-inf"))
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
        self.blocks = nn.ModuleList([nn.ModuleDict({'ln1': nn.LayerNorm(d_model), 'attn': MAttn(d_model, d_compare, d_memory, n_heads), 'ln2': nn.LayerNorm(d_model), 'ffn': MFFN(d_model, d_expand, d_gate)}) for _ in range(n_layers)])
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

def count_p(m): return sum(p.numel() for p in m.parameters())

@torch.no_grad()
def eval_acc(model, loader, dev):
    model.eval()
    cor, tot = 0, 0
    for b in loader:
        ids, tgt, lm = b["input_ids"].to(dev), b["targets"].to(dev), b["loss_mask"].to(dev)
        o = model(ids, tgt, lm)
        p = o["logits"].argmax(-1)
        fm = lm.bool()
        if fm.any():
            cor += (p[fm] == tgt[fm]).sum().item()
            tot += fm.sum().item()
    return cor / max(tot, 1)

def run_experiment():
    dev = torch.device("cpu")
    print(f"Running Scale-Up Experiment on {dev}\n")
    
    vocab_size = 1024
    seq_len = 128
    epochs = 8
    bs = 64
    
    torch.manual_seed(42)
    tr = ScaledRetrievalTask(vocab_size, seq_len, 1000, n_pairs=32, seed=0)
    ev = ScaledRetrievalTask(vocab_size, seq_len, 250, n_pairs=32, seed=99)
    tl = DataLoader(tr, batch_size=bs, shuffle=True)
    el = DataLoader(ev, batch_size=bs)
    
    # Target Parameter Budget: ~10.5M Params
    
    # 1. Uniform Baseline
    # d_model=384, d_ff=1536, h=6, l=6 -> ~10.6M params
    m_uniform = BaselineT(d_model=384, d_ff=1536, n_heads=6, n_layers=6, vocab_size=vocab_size, max_seq_len=seq_len).to(dev)
    
    # 2. FOG Motif Model (Parameter-matched to Uniform)
    # We constrain compare and gate, and enormously widen memory and expand
    # d_model=384, d_compare=64, d_memory=512, d_expand=1536, d_gate=64, h=6, l=6 -> ~10.4M params
    m_fog = MotifT(
        d_model=384, d_compare=64, d_memory=512, d_expand=1536, d_gate=64, 
        n_heads=6, n_layers=6, vocab_size=vocab_size, max_seq_len=seq_len
    ).to(dev)
    
    print(f"Params -> Uniform: {count_p(m_uniform)/1e6:.2f}M | FOG: {count_p(m_fog)/1e6:.2f}M")
    
    results = {}
    for name, model in [("Uniform", m_uniform), ("FOG", m_fog)]:
        print(f"\nTraining {name}...")
        opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
        
        t0 = time.time()
        for ep in range(1, epochs + 1):
            model.train()
            ep_loss = 0
            for b in tl:
                ids, tgt, lm = b["input_ids"].to(dev), b["targets"].to(dev), b["loss_mask"].to(dev)
                loss = model(ids, tgt, lm)["loss"]
                opt.zero_grad()
                loss.backward()
                opt.step()
                ep_loss += loss.item()
                
            if ep % 3 == 0 or ep == 1:
                acc = eval_acc(model, el, dev)
                print(f"  Ep {ep:2d} | Train Loss: {ep_loss/len(tl):.3f} | Eval Acc: {acc:.3f}")
                
        elapsed = time.time() - t0
        final_acc = eval_acc(model, el, dev)
        print(f"  {name:8s} | Final Acc: {final_acc:.3f} | Time: {elapsed:.1f}s")
        results[name] = {"acc": final_acc, "params_m": count_p(model)/1e6, "time_s": elapsed}

    out_path = Path("fog_exp8_scaleup.json")
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved scale-up results to {out_path}")

if __name__ == "__main__":
    run_experiment()

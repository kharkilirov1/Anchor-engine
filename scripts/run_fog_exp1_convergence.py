"""
Experiment 1 & 3: FOG Validation (Convergence & eRank Collapse)
Runs on local CPU. Tests parameter-matched Baseline vs Motif models on Retrieval task.
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

# --- CONFIGS ---
@dataclass(frozen=True)
class C:
    vocab_size: int = 32
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    max_seq_len: int = 32
    dropout: float = 0.1
    d_ff: int = 512
    d_compare: int = 32
    d_memory: int = 96
    d_expand: int = 256
    d_gate: int = 16

BASELINE = C()
# Adjusted to roughly match the parameter count of Motif (~350K params)
UNIFORM = C(d_model=96, n_heads=2, d_ff=384) 
MOTIF = C()

# --- DATASET ---
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

# --- MODELS ---
class BAttn(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h = h
        self.hd = d // h
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)

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
        self.dn = nn.Linear(df, d) # W2 matrix

    def forward(self, x):
        return self.dn(F.silu(self.up(x)))

class BBlock(nn.Module):
    def __init__(self, d, df, h, dr):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.attn = BAttn(d, h)
        self.ffn = BFFN(d, df)
        self.drop = nn.Dropout(dr)

    def forward(self, x, mask=None):
        x = x + self.drop(self.attn(self.ln1(x), mask))
        return x + self.drop(self.ffn(self.ln2(x)))

class BaselineT(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.te = nn.Embedding(c.vocab_size, c.d_model)
        self.pe = nn.Embedding(c.max_seq_len, c.d_model)
        self.blocks = nn.ModuleList([BBlock(c.d_model, c.d_ff, c.n_heads, c.dropout) for _ in range(c.n_layers)])
        self.ln = nn.LayerNorm(c.d_model)
        self.head = nn.Linear(c.d_model, c.vocab_size, bias=False)
        self.te.weight = self.head.weight

    def forward(self, ids, tgt=None, lm=None):
        b, t = ids.shape
        x = self.te(ids) + self.pe(torch.arange(t, device=ids.device).unsqueeze(0))
        mask = torch.tril(torch.ones(t, t, device=x.device, dtype=torch.bool)).unsqueeze(0).unsqueeze(0)
        for bl in self.blocks:
            x = bl(x, mask)
        lo = self.head(self.ln(x))
        loss = None
        if tgt is not None:
            fl, ft = lo.view(-1, lo.size(-1)), tgt.view(-1)
            if lm is not None:
                fm = lm.view(-1).bool()
                loss = F.cross_entropy(fl[fm], ft[fm]) if fm.any() else torch.tensor(0.0, device=lo.device)
            else:
                loss = F.cross_entropy(fl, ft)
        return {"logits": lo, "loss": loss}

class MAttn(nn.Module):
    def __init__(self, d, dc, dm, h):
        super().__init__()
        self.h = h
        self.chd = dc // h
        self.mhd = dm // h
        self.dm = dm
        self.qp = nn.Linear(d, dc)
        self.kp = nn.Linear(d, dc)
        self.vp = nn.Linear(d, dm)
        self.out = nn.Linear(dm, d)

    def forward(self, x, mask=None):
        b, t, _ = x.shape
        q = self.qp(x).view(b, t, self.h, self.chd).transpose(1, 2)
        k = self.kp(x).view(b, t, self.h, self.chd).transpose(1, 2)
        v = self.vp(x).view(b, t, self.h, self.mhd).transpose(1, 2)
        sc = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.chd)
        if mask is not None:
            sc = sc.masked_fill(mask == 0, float("-inf"))
        y = torch.matmul(F.softmax(sc, dim=-1), v)
        return self.out(y.transpose(1, 2).contiguous().view(b, t, self.dm))

class MFFN(nn.Module):
    def __init__(self, d, de, dg):
        super().__init__()
        self.exp = nn.Linear(d, de)
        self.gate = nn.Linear(d, dg)
        self.gup = nn.Linear(dg, de)
        self.comp = nn.Linear(de, d) # W2 matrix

    def forward(self, x):
        e = self.exp(x)
        g = torch.sigmoid(self.gup(F.silu(self.gate(x))))
        return self.comp(F.silu(e * g))

class MBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln1 = nn.LayerNorm(c.d_model)
        self.ln2 = nn.LayerNorm(c.d_model)
        self.attn = MAttn(c.d_model, c.d_compare, c.d_memory, c.n_heads)
        self.ffn = MFFN(c.d_model, c.d_expand, c.d_gate)
        self.drop = nn.Dropout(c.dropout)

    def forward(self, x, mask=None):
        x = x + self.drop(self.attn(self.ln1(x), mask))
        return x + self.drop(self.ffn(self.ln2(x)))

class MotifT(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.te = nn.Embedding(c.vocab_size, c.d_model)
        self.pe = nn.Embedding(c.max_seq_len, c.d_model)
        self.blocks = nn.ModuleList([MBlock(c) for _ in range(c.n_layers)])
        self.ln = nn.LayerNorm(c.d_model)
        self.head = nn.Linear(c.d_model, c.vocab_size, bias=False)
        self.te.weight = self.head.weight

    def forward(self, ids, tgt=None, lm=None):
        b, t = ids.shape
        x = self.te(ids) + self.pe(torch.arange(t, device=ids.device).unsqueeze(0))
        mask = torch.tril(torch.ones(t, t, device=x.device, dtype=torch.bool)).unsqueeze(0).unsqueeze(0)
        for bl in self.blocks:
            x = bl(x, mask)
        lo = self.head(self.ln(x))
        loss = None
        if tgt is not None:
            fl, ft = lo.view(-1, lo.size(-1)), tgt.view(-1)
            if lm is not None:
                fm = lm.view(-1).bool()
                loss = F.cross_entropy(fl[fm], ft[fm]) if fm.any() else torch.tensor(0.0, device=lo.device)
            else:
                loss = F.cross_entropy(fl, ft)
        return {"logits": lo, "loss": loss}

# --- ERANK CALCULATION ---
def get_erank(W):
    with torch.no_grad():
        W = W.detach().float()
        if W.dim() > 2:
            W = W.view(W.size(0), -1)
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        S = S[S > 1e-9]
        if len(S) == 0: return 0.0, 0
        p = S / S.sum()
        entropy = -torch.sum(p * torch.log(p))
        erank = torch.exp(entropy).item()
        return erank, len(S)

def extract_w2_matrices(model):
    matrices = []
    for name, param in model.named_parameters():
        if "ffn.dn.weight" in name or "ffn.comp.weight" in name:
            matrices.append((name, param))
    return matrices

def count_p(m):
    return sum(p.numel() for p in m.parameters())

# --- TRAINING LOOP ---
def train_ep(model, loader, opt, dev):
    model.train()
    tl = 0
    nb = 0
    for b in loader:
        ids = b["input_ids"].to(dev)
        tgt = b["targets"].to(dev)
        lm = b["loss_mask"].to(dev)
        o = model(ids, tgt, lm)
        loss = o["loss"]
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tl += loss.item()
        nb += 1
    return tl / max(nb, 1)

@torch.no_grad()
def eval_acc(model, loader, dev):
    model.eval()
    tl = 0
    cor = 0
    tot = 0
    nb = 0
    for b in loader:
        ids = b["input_ids"].to(dev)
        tgt = b["targets"].to(dev)
        lm = b["loss_mask"].to(dev)
        o = model(ids, tgt, lm)
        tl += o["loss"].item()
        nb += 1
        p = o["logits"].argmax(-1)
        fm = lm.bool()
        cor += (p[fm] == tgt[fm]).sum().item()
        tot += fm.sum().item()
    return {"loss": tl / max(nb, 1), "accuracy": cor / max(tot, 1)}

def run_experiment():
    dev = torch.device("cpu")
    print(f"Running FOG Validation on {dev}")
    
    configs = [
        ("baseline_large", BASELINE), 
        ("uniform_small", UNIFORM), 
        ("fog_motif", MOTIF)
    ]
    
    epochs = 40
    bs = 64
    lr = 5e-4
    seed = 42
    
    torch.manual_seed(seed)
    
    # 1. Dataset
    # We use Retrieval Task as it heavily relies on pure memory and comparison
    tr = RetrievalTask(BASELINE.vocab_size, BASELINE.max_seq_len, 5000, seed=0)
    ev = RetrievalTask(BASELINE.vocab_size, BASELINE.max_seq_len, 500, seed=99)
    tl = DataLoader(tr, batch_size=bs, shuffle=True)
    el = DataLoader(ev, batch_size=bs)
    
    results = {}
    
    for mtype, cfg in configs:
        print(f"\n{'='*50}\nTraining {mtype}\n{'='*50}")
        model_class = BaselineT if mtype in ("baseline_large", "uniform_small") else MotifT
        model = model_class(cfg).to(dev)
        np_ = count_p(model)
        print(f"Parameters: {np_/1000:.1f}K")
        
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        
        history = []
        w2_mats = extract_w2_matrices(model)
        
        # Initial eRank
        init_ranks = [get_erank(W)[0] for n, W in w2_mats]
        print(f"Init Mean eRank: {sum(init_ranks)/len(init_ranks):.1f}")
        
        t0 = time.time()
        for ep in range(1, epochs + 1):
            trl = train_ep(model, tl, opt, dev)
            
            if ep % 5 == 0 or ep == 1:
                m = eval_acc(model, el, dev)
                
                # Measure eRank periodically
                ranks = [get_erank(W)[0] for n, W in w2_mats]
                mean_erank = sum(ranks) / len(ranks) if ranks else 0.0
                
                print(f"Ep {ep:>3d} | Loss: {trl:.3f} | Eval Acc: {m['accuracy']:.3f} | Mean eRank: {mean_erank:.1f}")
                
                history.append({
                    "epoch": ep,
                    "train_loss": trl,
                    "eval_acc": m['accuracy'],
                    "mean_erank": mean_erank
                })
        
        elapsed = time.time() - t0
        print(f"-> {mtype} finished in {elapsed:.1f}s")
        results[mtype] = {
            "params": np_,
            "history": history
        }
        
    out_path = Path("fog_exp1_results.json")
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    run_experiment()

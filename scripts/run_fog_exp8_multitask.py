"""
Experiment 8: Multi-Task Interference
Trains models on a mixture of tasks (Copy, Reverse, Retrieval) to see if
FOG's heterogeneous geometry prevents catastrophic interference better than Uniform Baseline.
"""
import math
import random
import time
import json
from pathlib import Path
from dataclasses import dataclass
import numpy as np

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
        return {"input_ids": x, "targets": y, "loss_mask": m, "task_id": 0}

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
        return {"input_ids": x, "targets": y, "loss_mask": m, "task_id": 1}

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
        return {"input_ids": x, "targets": y, "loss_mask": m, "task_id": 2}

class MixedTask(Dataset):
    def __init__(self, datasets, seed=42):
        super().__init__()
        self.items = []
        for ds in datasets:
            for i in range(len(ds)):
                self.items.append(ds[i])
        rng = random.Random(seed)
        rng.shuffle(self.items)
    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]

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
    task_stats = {0: [0, 0], 1: [0, 0], 2: [0, 0]} # {task_id: [correct, total]}
    cor, tot = 0, 0
    for b in loader:
        ids, tgt, lm, t_ids = b["input_ids"].to(dev), b["targets"].to(dev), b["loss_mask"].to(dev), b["task_id"]
        o = model(ids, tgt, lm)
        p = o["logits"].argmax(-1)
        fm = lm.bool()
        
        # Calculate per-item accuracy to group by task
        for i in range(ids.size(0)):
            item_mask = fm[i]
            if item_mask.any():
                item_cor = (p[i][item_mask] == tgt[i][item_mask]).sum().item()
                item_tot = item_mask.sum().item()
                t_id = t_ids[i].item()
                task_stats[t_id][0] += item_cor
                task_stats[t_id][1] += item_tot
                cor += item_cor
                tot += item_tot
                
    accs = {
        "Copy": task_stats[0][0] / max(task_stats[0][1], 1),
        "Reverse": task_stats[1][0] / max(task_stats[1][1], 1),
        "Retrieval": task_stats[2][0] / max(task_stats[2][1], 1),
        "Overall": cor / max(tot, 1)
    }
    return accs

def run_experiment():
    dev = torch.device("cpu")
    print(f"Running Multi-Task Interference Experiment on {dev}\n")
    
    vocab_size = 32
    seq_len = 32
    epochs = 40
    bs = 64
    seeds = [42, 43, 44]
    
    # Approx 450K params budget
    d_model = 96
    
    results = {}
    for model_name in ["Uniform", "FOG"]:
        results[model_name] = []
        
    for seed in seeds:
        print(f"\n{'='*40}\nSeed: {seed}\n{'='*40}")
        torch.manual_seed(seed)
        
        # Build mixed datasets
        tr_copy = CopyTask(vocab_size, seq_len, 2000, seed=seed)
        tr_rev = ReverseTask(vocab_size, seq_len, 2000, seed=seed)
        tr_ret = RetrievalTask(vocab_size, seq_len, 2000, seed=seed)
        
        ev_copy = CopyTask(vocab_size, seq_len, 300, seed=seed+99)
        ev_rev = ReverseTask(vocab_size, seq_len, 300, seed=seed+99)
        ev_ret = RetrievalTask(vocab_size, seq_len, 300, seed=seed+99)
        
        tr_mixed = MixedTask([tr_copy, tr_rev, tr_ret], seed=seed)
        ev_mixed = MixedTask([ev_copy, ev_rev, ev_ret], seed=seed+99)
        
        tl = DataLoader(tr_mixed, batch_size=bs, shuffle=True)
        el = DataLoader(ev_mixed, batch_size=bs)
        
        # 1. Uniform Baseline (~450K)
        m_uniform = BaselineT(d_model=d_model, d_ff=d_model*4, n_heads=2, n_layers=4, vocab_size=vocab_size, max_seq_len=seq_len).to(dev)
        
        # 2. FOG Motif (~430K)
        m_fog = MotifT(
            d_model=d_model, d_compare=24, d_memory=72, d_expand=192, d_gate=12, 
            n_heads=2, n_layers=4, vocab_size=vocab_size, max_seq_len=seq_len
        ).to(dev)
        
        if seed == seeds[0]:
            print(f"Params -> Uniform: {count_p(m_uniform)/1000:.1f}K | FOG: {count_p(m_fog)/1000:.1f}K")
            
        for name, model in [("Uniform", m_uniform), ("FOG", m_fog)]:
            print(f"  Training {name}...")
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
                    
            accs = eval_acc(model, el, dev)
            elapsed = time.time() - t0
            print(f"    Final Accuracies: Overall={accs['Overall']:.3f} | Copy={accs['Copy']:.3f} | Reverse={accs['Reverse']:.3f} | Retrieval={accs['Retrieval']:.3f}")
            results[name].append({
                "seed": seed,
                "acc_copy": accs['Copy'],
                "acc_reverse": accs['Reverse'],
                "acc_retrieval": accs['Retrieval'],
                "acc_overall": accs['Overall'],
                "time": elapsed
            })

    # Summary
    print(f"\n{'='*40}\nSUMMARY ACROSS {len(seeds)} SEEDS\n{'='*40}")
    final_output = {}
    for name in ["Uniform", "FOG"]:
        runs = results[name]
        mean_copy = np.mean([r['acc_copy'] for r in runs])
        mean_rev = np.mean([r['acc_reverse'] for r in runs])
        mean_ret = np.mean([r['acc_retrieval'] for r in runs])
        mean_all = np.mean([r['acc_overall'] for r in runs])
        
        final_output[name] = {
            "mean_copy": round(float(mean_copy), 3),
            "mean_reverse": round(float(mean_rev), 3),
            "mean_retrieval": round(float(mean_ret), 3),
            "mean_overall": round(float(mean_all), 3)
        }
        
        print(f"{name:8s} | Overall: {mean_all:.3f} | Copy: {mean_copy:.3f} | Rev: {mean_rev:.3f} | Ret: {mean_ret:.3f}")

    out_path = Path("fog_exp8_multitask.json")
    out_path.write_text(json.dumps(final_output, indent=2))
    print(f"\nSaved multi-task results to {out_path}")

if __name__ == "__main__":
    run_experiment()

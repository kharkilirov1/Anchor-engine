"""
Experiment 6: Complex Motif Composition Stress Test
Tests FOG vs Baseline on complex compositional tasks (ChainedRetrieval, NoisyRetrieval).
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

# --- HELPERS ---
def _build_item(ids: list[int], sp: int, sl: int) -> dict[str, torch.Tensor]:
    ids = ids[:sl]
    ids += [0] * (sl - len(ids))
    x = torch.tensor(ids[:-1], dtype=torch.long)
    y = torch.tensor(ids[1:], dtype=torch.long)
    m = torch.zeros_like(y)
    if sp < len(m):
        m[sp:] = 1
    return {"input_ids": x, "targets": y, "loss_mask": m}

# --- COMPLEX DATASETS ---
class DistractorRetrieval(Dataset):
    """Keys differ by +/-1 from query. Forces precise compare."""
    def __init__(self, vocab_size: int, seq_len: int, n_samples: int, n_pairs: int = 4, seed: int = 42):
        super().__init__()
        sep = vocab_size - 1
        rng = random.Random(seed)
        cv = vocab_size - 2
        self.items = []
        for _ in range(n_samples):
            q_k = rng.randint(1, cv - 2)
            keys = [q_k]
            while len(keys) < n_pairs:
                cand = rng.choice([q_k - 1, q_k + 1])
                if cand not in keys and 0 <= cand < cv:
                    keys.append(cand)
                else:
                    cand = rng.randint(0, cv - 1)
                    if cand not in keys:
                        keys.append(cand)
            rng.shuffle(keys)
            values = [rng.randint(0, cv - 1) for _ in keys]
            ids = []
            for k, v in zip(keys, values):
                ids.extend([k, v])
            sp = len(ids)
            ids += [sep, q_k, values[keys.index(q_k)]]
            self.items.append(_build_item(ids, sp, seq_len))
    def __len__(self): return len(self.items)
    def __getitem__(self, idx): return self.items[idx]

class NoisyRetrieval(Dataset):
    """Noise tokens between KV pairs. Forces select to filter, memory to retrieve."""
    def __init__(self, vocab_size: int, seq_len: int, n_samples: int, n_pairs: int = 3, noise_len: int = 2, seed: int = 42):
        super().__init__()
        sep = vocab_size - 1
        rng = random.Random(seed)
        cv = vocab_size - 2
        self.items = []
        for _ in range(n_samples):
            keys = rng.sample(range(cv), min(n_pairs, cv))
            values = [rng.randint(0, cv - 1) for _ in keys]
            qi = rng.randint(0, len(keys) - 1)
            ids = []
            for k, v in zip(keys, values):
                ids.extend([k, v])
                ids.extend([rng.randint(0, cv - 1) for _ in range(noise_len)])
            sp = len(ids)
            ids += [sep, keys[qi], values[qi]]
            self.items.append(_build_item(ids, sp, seq_len))
    def __len__(self): return len(self.items)
    def __getitem__(self, idx): return self.items[idx]

class ChainedRetrieval(Dataset):
    """Two-hop lookup: find value for query key, use that value as key for second lookup."""
    def __init__(self, vocab_size: int, seq_len: int, n_samples: int, n_pairs: int = 5, seed: int = 42):
        super().__init__()
        sep = vocab_size - 1
        rng = random.Random(seed)
        cv = vocab_size - 2
        self.items = []
        attempts = 0
        while len(self.items) < n_samples and attempts < n_samples * 20:
            attempts += 1
            keys = rng.sample(range(cv), min(n_pairs, cv))
            values = [rng.randint(0, cv - 1) for _ in keys]
            chain_found = False
            for qi in range(len(keys)):
                v1 = values[qi]
                for hop2 in range(len(keys)):
                    if hop2 != qi and keys[hop2] == v1:
                        ids = []
                        for k, v in zip(keys, values):
                            ids.extend([k, v])
                        sp = len(ids)
                        ids += [sep, keys[qi], values[hop2]]
                        self.items.append(_build_item(ids, sp, seq_len))
                        chain_found = True
                        break
                if chain_found: break
    def __len__(self): return len(self.items)
    def __getitem__(self, idx): return self.items[idx]

class ComposeArithmetic(Dataset):
    """Retrieve two values by keys, output (v1 + v2) mod M."""
    def __init__(self, vocab_size: int, seq_len: int, n_samples: int, n_pairs: int = 5, seed: int = 42):
        super().__init__()
        sep = vocab_size - 1
        rng = random.Random(seed)
        cv = vocab_size - 2
        modulus = cv
        self.items = []
        for _ in range(n_samples):
            keys = rng.sample(range(cv), min(n_pairs, cv))
            values = [rng.randint(0, cv - 1) for _ in keys]
            qi1, qi2 = rng.sample(range(len(keys)), 2)
            answer = (values[qi1] + values[qi2]) % modulus
            ids = []
            for k, v in zip(keys, values):
                ids.extend([k, v])
            sp = len(ids)
            ids += [sep, keys[qi1], keys[qi2], answer]
            self.items.append(_build_item(ids, sp, seq_len))
    def __len__(self): return len(self.items)
    def __getitem__(self, idx): return self.items[idx]

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
        cor += (p[fm] == tgt[fm]).sum().item()
        tot += fm.sum().item()
    return cor / max(tot, 1)

def run_experiment():
    dev = torch.device("cpu")
    print(f"Running Complex Motif Composition Stress Test on {dev}\n")
    
    vocab_size = 48
    seq_len = 48
    epochs = 40
    bs = 64
    
    tasks = {
        "DistractorRetrieval": DistractorRetrieval,
        "NoisyRetrieval": NoisyRetrieval,
        "ChainedRetrieval": ChainedRetrieval,
        "ComposeArithmetic": ComposeArithmetic
    }
    
    results = {}
    
    # We test on ~400K param budget
    d_model = 96
    
    for task_name, task_cls in tasks.items():
        print(f"\n{'='*40}\nTask: {task_name}\n{'='*40}")
        torch.manual_seed(42)
        
        tr = task_cls(vocab_size, seq_len, 4000, seed=0)
        ev = task_cls(vocab_size, seq_len, 500, seed=99)
        tl = DataLoader(tr, batch_size=bs, shuffle=True)
        el = DataLoader(ev, batch_size=bs)
        
        # 1. Uniform Baseline (~450K)
        m_uniform = BaselineT(d_model=d_model, d_ff=d_model*4, n_heads=2, n_layers=4, vocab_size=vocab_size, max_seq_len=seq_len).to(dev)
        
        # 2. FOG Motif (~430K)
        m_fog = MotifT(
            d_model=d_model, d_compare=24, d_memory=72, d_expand=192, d_gate=12, 
            n_heads=2, n_layers=4, vocab_size=vocab_size, max_seq_len=seq_len
        ).to(dev)
        
        if task_name == "DistractorRetrieval": # Print params only once
            print(f"Params -> Uniform: {count_p(m_uniform)/1000:.1f}K | FOG: {count_p(m_fog)/1000:.1f}K")
            
        results[task_name] = {}
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
            results[task_name][name] = {"acc": acc, "params": count_p(model)}

    out_path = Path("fog_exp6_stress_test.json")
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved stress test results to {out_path}")

if __name__ == "__main__":
    run_experiment()

"""
Experiment 5: Motif Signatures Extraction (Revised)
Extracts normalized attention entropy and activation polarization
across multiple seeds to compare Uniform Baseline vs FOG Motif.
"""
import math
import random
import time
import json
from pathlib import Path
from collections import defaultdict
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

# --- TRACKABLE MODELS ---
class TrackableBAttn(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h = h
        self.hd = max(1, d // h)
        self.qkv = nn.Linear(d, 3 * self.h * self.hd)
        self.out = nn.Linear(self.h * self.hd, d)
        self.last_attn_weights = None

    def forward(self, x, mask=None):
        b, t, _ = x.shape
        qkv = self.qkv(x).view(b, t, 3, self.h, self.hd)
        q, k, v = qkv.unbind(2)
        q, k, v = [z.transpose(1, 2) for z in (q, k, v)]
        sc = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.hd)
        if mask is not None: sc = sc.masked_fill(mask == 0, float("-inf"))
        
        attn_weights = F.softmax(sc, dim=-1)
        if not self.training:
            self.last_attn_weights = attn_weights.detach()
            
        y = torch.matmul(attn_weights, v)
        return self.out(y.transpose(1, 2).contiguous().view(b, t, -1))

class TrackableBFFN(nn.Module):
    def __init__(self, d, df):
        super().__init__()
        self.up = nn.Linear(d, df)
        self.dn = nn.Linear(df, d)
        self.last_acts = None

    def forward(self, x):
        acts = F.silu(self.up(x))
        if not self.training:
            self.last_acts = acts.detach()
        return self.dn(acts)

class BaselineT(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, n_layers, vocab_size, max_seq_len):
        super().__init__()
        self.te = nn.Embedding(vocab_size, d_model)
        self.pe = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                'ln1': nn.LayerNorm(d_model),
                'attn': TrackableBAttn(d_model, n_heads),
                'ln2': nn.LayerNorm(d_model),
                'ffn': TrackableBFFN(d_model, d_ff)
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

# --- METRICS ---
def get_normalized_attention_entropy(attn_weights, loss_mask):
    """
    Computes normalized attention entropy per head on valid output tokens.
    H_norm = H / log(t+1)
    Returns: list of floats (entropy per head)
    """
    B, H, T, _ = attn_weights.shape
    epsilon = 1e-9
    
    # Entropy per query token: [B, H, T]
    entropy = -torch.sum(attn_weights * torch.log(attn_weights + epsilon), dim=-1)
    
    # Max entropy at position t is log(t+1) due to causal masking
    max_entropy = torch.log(torch.arange(1, T + 1, device=attn_weights.device, dtype=torch.float32))
    max_entropy[0] = 1.0 # avoid div by zero, entropy at t=0 is 0 anyway
    
    norm_entropy = entropy / max_entropy.view(1, 1, T) # [B, H, T]
    
    # Average only over valid tokens according to loss_mask
    valid_tokens = loss_mask.sum().item()
    if valid_tokens > 0:
        mask_expanded = loss_mask.view(B, 1, T).expand(B, H, T)
        avg_norm_entropy = (norm_entropy * mask_expanded).sum(dim=(0, 2)) / valid_tokens # [H]
    else:
        avg_norm_entropy = norm_entropy.mean(dim=(0, 2)) # [H]
        
    return avg_norm_entropy.tolist()

def get_activation_metrics(model_type, bl, loss_mask):
    """
    For FOG, measures gate polarization (variance).
    For Uniform, measures activation sparsity (fraction near 0).
    Averages over valid output tokens.
    """
    if model_type == "FOG":
        acts = bl['ffn'].last_gate_values # [B, T, D_gate]
    else:
        acts = bl['ffn'].last_acts # [B, T, D_ff]
        
    B, T, D = acts.shape
    valid_mask = loss_mask.bool().view(B, T)
    
    if valid_mask.any():
        valid_acts = acts[valid_mask] # [N_valid, D]
    else:
        valid_acts = acts.view(-1, D)
        
    if model_type == "FOG":
        # Variance of gates in [0, 1]
        metric = torch.var(valid_acts).item()
    else:
        # Sparsity of SiLU activations
        metric = (valid_acts < 0.01).float().mean().item()
        
    return metric

def train_and_extract(model_name, model_cls, model_kwargs, task_name, dataset_cls, dev, seed):
    vocab_size = 32
    seq_len = 32
    epochs = 20
    bs = 64
    torch.manual_seed(seed)
    
    tr = dataset_cls(vocab_size, seq_len, 1500, seed=seed)
    ev = dataset_cls(vocab_size, seq_len, 500, seed=seed+99)
    tl = DataLoader(tr, batch_size=bs, shuffle=True)
    el = DataLoader(ev, batch_size=bs)
    
    model = model_cls(**model_kwargs).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    
    for ep in range(1, epochs + 1):
        model.train()
        for b in tl:
            ids, tgt, lm = b["input_ids"].to(dev), b["targets"].to(dev), b["loss_mask"].to(dev)
            loss = model(ids, tgt, lm)["loss"]
            opt.zero_grad()
            loss.backward()
            opt.step()
    
    model.eval()
    
    layer_stats = defaultdict(lambda: {"attn_entropy_heads": [], "activation_metric": []})
    
    with torch.no_grad():
        for b in el:
            ids, tgt, lm = b["input_ids"].to(dev), b["targets"].to(dev), b["loss_mask"].to(dev)
            _ = model(ids, tgt, lm)
            
            for i, bl in enumerate(model.blocks):
                attn_w = bl['attn'].last_attn_weights
                
                head_entropies = get_normalized_attention_entropy(attn_w, lm)
                act_metric = get_activation_metrics(model_name, bl, lm)
                
                layer_stats[i]["attn_entropy_heads"].append(head_entropies)
                layer_stats[i]["activation_metric"].append(act_metric)

    # Average over batches
    final_stats = {}
    for i in range(len(model.blocks)):
        # attn_entropy_heads is a list of lists: [batch][head]
        all_batches_heads = np.array(layer_stats[i]["attn_entropy_heads"]) # [n_batches, n_heads]
        avg_head_entropy = all_batches_heads.mean(axis=0).tolist()
        avg_act_metric = np.mean(layer_stats[i]["activation_metric"])
        
        final_stats[f"layer_{i}"] = {
            "attn_entropy_per_head": [round(x, 3) for x in avg_head_entropy],
            "attn_entropy_mean": round(np.mean(avg_head_entropy), 3),
            "activation_metric": round(float(avg_act_metric), 3) # FOG=Polarization, Uniform=Sparsity
        }
        
    return final_stats

def main():
    dev = torch.device("cpu")
    print(f"Running Revised Motif Signatures Analysis on {dev}\n")
    
    tasks = {
        "Copy": CopyTask,
        "Reverse": ReverseTask,
        "Retrieval": RetrievalTask
    }
    
    # Approx 450K params for both
    uniform_kwargs = dict(d_model=96, d_ff=384, n_heads=4, n_layers=4, vocab_size=32, max_seq_len=32)
    fog_kwargs = dict(d_model=96, d_compare=24, d_memory=72, d_expand=192, d_gate=12, n_heads=4, n_layers=4, vocab_size=32, max_seq_len=32)
    
    models = {
        "Uniform": (BaselineT, uniform_kwargs),
        "FOG": (MotifT, fog_kwargs)
    }
    
    seeds = [42, 43, 44]
    
    all_results = {}
    
    for task_name, task_cls in tasks.items():
        all_results[task_name] = {}
        print(f"\n{'='*40}\nTask: {task_name}\n{'='*40}")
        
        for model_name, (model_cls, model_kwargs) in models.items():
            print(f"  Evaluating {model_name}...")
            
            seed_results = []
            for seed in seeds:
                stats = train_and_extract(model_name, model_cls, model_kwargs, task_name, task_cls, dev, seed)
                seed_results.append(stats)
                
            # Average across seeds
            avg_stats = {}
            for layer in seed_results[0].keys():
                mean_entropies = np.mean([s[layer]["attn_entropy_per_head"] for s in seed_results], axis=0)
                mean_overall = np.mean([s[layer]["attn_entropy_mean"] for s in seed_results])
                mean_act = np.mean([s[layer]["activation_metric"] for s in seed_results])
                
                avg_stats[layer] = {
                    "attn_entropy_per_head": [round(x, 3) for x in mean_entropies],
                    "attn_entropy_mean": round(float(mean_overall), 3),
                    "activation_metric": round(float(mean_act), 3)
                }
            
            all_results[task_name][model_name] = avg_stats
            
            print(f"    {model_name} Averages across {len(seeds)} seeds:")
            for l_idx, l_stats in avg_stats.items():
                print(f"      {l_idx}: Attn Entropy Mean: {l_stats['attn_entropy_mean']:.3f} | Per-Head: {l_stats['attn_entropy_per_head']} | Act Metric: {l_stats['activation_metric']:.3f}")

    out_path = Path("fog_exp5_signatures_revised.json")
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved revised signatures to {out_path}")

if __name__ == "__main__":
    main()

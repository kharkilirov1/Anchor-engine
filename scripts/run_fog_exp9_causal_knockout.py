"""
Experiment 9: Causal Motif Intervention (Knockout Analysis)
Causally proves that specific heads/layers are responsible for specific motifs
by selectively zeroing out attention heads or gate activations.
Now compares Uniform vs FOG across 3 seeds and 3 tasks (Copy, Reverse, Retrieval).
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
        return {"input_ids": x, "targets": y, "loss_mask": m, "task_id": 0, "task_name": "Copy"}

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
        return {"input_ids": x, "targets": y, "loss_mask": m, "task_id": 1, "task_name": "Reverse"}

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
        return {"input_ids": x, "targets": y, "loss_mask": m, "task_id": 2, "task_name": "Retrieval"}

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

# --- MODELS (With Ablation Hooks) ---
class AblatableBAttn(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h = h
        self.hd = max(1, d // h)
        self.qkv = nn.Linear(d, 3 * self.h * self.hd)
        self.out = nn.Linear(self.h * self.hd, d)
        self.ablate_head = None 
    def forward(self, x, mask=None):
        b, t, _ = x.shape
        qkv = self.qkv(x).view(b, t, 3, self.h, self.hd)
        q, k, v = qkv.unbind(2)
        q, k, v = [z.transpose(1, 2) for z in (q, k, v)]
        sc = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.hd)
        if mask is not None: sc = sc.masked_fill(mask == 0, float("-inf"))
        attn_weights = F.softmax(sc, dim=-1)
        
        if self.ablate_head is not None and not self.training:
            attn_weights[:, self.ablate_head, :, :] = 0.0
            
        y = torch.matmul(attn_weights, v)
        return self.out(y.transpose(1, 2).contiguous().view(b, t, -1))

class AblatableBFFN(nn.Module):
    def __init__(self, d, df):
        super().__init__()
        self.up = nn.Linear(d, df)
        self.dn = nn.Linear(df, d)
        self.ablate_gate = False # Uniform doesn't have gates, but we can ablate the whole hidden state
    def forward(self, x): 
        h = F.silu(self.up(x))
        if self.ablate_gate and not self.training:
            h = torch.zeros_like(h) # Severe! Ablates entire FFN
        return self.dn(h)

class BaselineT(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, n_layers, vocab_size, max_seq_len):
        super().__init__()
        self.te = nn.Embedding(vocab_size, d_model)
        self.pe = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([nn.ModuleDict({'ln1': nn.LayerNorm(d_model), 'attn': AblatableBAttn(d_model, n_heads), 'ln2': nn.LayerNorm(d_model), 'ffn': AblatableBFFN(d_model, d_ff)}) for _ in range(n_layers)])
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

class AblatableMAttn(nn.Module):
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
        self.ablate_head = None
    def forward(self, x, mask=None):
        b, t, _ = x.shape
        q = self.qp(x).view(b, t, self.h, self.chd).transpose(1, 2)
        k = self.kp(x).view(b, t, self.h, self.chd).transpose(1, 2)
        v = self.vp(x).view(b, t, self.h, self.mhd).transpose(1, 2)
        sc = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.chd)
        if mask is not None:
            sc = sc.masked_fill(mask == 0, float("-inf"))
        attn_weights = F.softmax(sc, dim=-1)
        if self.ablate_head is not None and not self.training:
            attn_weights[:, self.ablate_head, :, :] = 0.0
        y = torch.matmul(attn_weights, v)
        return self.out(y.transpose(1, 2).contiguous().view(b, t, -1))

class AblatableMFFN(nn.Module):
    def __init__(self, d, de, dg):
        super().__init__()
        self.exp = nn.Linear(d, de)
        self.gate = nn.Linear(d, dg)
        self.gup = nn.Linear(dg, de)
        self.comp = nn.Linear(de, d)
        self.ablate_gate = False
    def forward(self, x):
        e = self.exp(x)
        g = torch.sigmoid(self.gup(F.silu(self.gate(x))))
        if self.ablate_gate and not self.training:
            g = torch.zeros_like(g) # Only ablates the gate, memory expansion still happens
        return self.comp(F.silu(e * g))

class MotifT(nn.Module):
    def __init__(self, d_model, d_compare, d_memory, d_expand, d_gate, n_heads, n_layers, vocab_size, max_seq_len):
        super().__init__()
        self.te = nn.Embedding(vocab_size, d_model)
        self.pe = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                'ln1': nn.LayerNorm(d_model),
                'attn': AblatableMAttn(d_model, d_compare, d_memory, n_heads),
                'ln2': nn.LayerNorm(d_model),
                'ffn': AblatableMFFN(d_model, d_expand, d_gate)
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

def reset_ablations(model):
    for bl in model.blocks:
        bl['attn'].ablate_head = None
        bl['ffn'].ablate_gate = False

def set_head_ablation(model, layer_idx, head_idx):
    reset_ablations(model)
    model.blocks[layer_idx]['attn'].ablate_head = head_idx

def set_gate_ablation(model, layer_idx):
    reset_ablations(model)
    model.blocks[layer_idx]['ffn'].ablate_gate = True

def run_experiment():
    dev = torch.device("cpu")
    print(f"Running Enhanced Causal Knockout Analysis on {dev}\n")
    
    vocab_size = 32
    seq_len = 32
    epochs = 20
    bs = 64
    seeds = [42, 43, 44]
    
    # Target Parameter Budget: ~250K Params (For both to be fair)
    d_model = 96
    
    models = {
        "Uniform": lambda: BaselineT(d_model=d_model, d_ff=d_model*4, n_heads=4, n_layers=4, vocab_size=vocab_size, max_seq_len=seq_len),
        "FOG": lambda: MotifT(d_model=d_model, d_compare=24, d_memory=72, d_expand=192, d_gate=12, n_heads=4, n_layers=4, vocab_size=vocab_size, max_seq_len=seq_len)
    }
    
    all_results = {}
    
    for model_name, model_fn in models.items():
        all_results[model_name] = []
        print(f"\n{'='*40}\nEvaluating Model: {model_name}\n{'='*40}")
        
        for seed in seeds:
            print(f"\n  --- Seed {seed} ---")
            torch.manual_seed(seed)
            
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
            
            model = model_fn().to(dev)
            if seed == seeds[0] and model_name == "Uniform":
                print(f"  Params -> Uniform: {sum(p.numel() for p in model.parameters())/1000:.1f}K")
            elif seed == seeds[0] and model_name == "FOG":
                print(f"  Params -> FOG: {sum(p.numel() for p in model.parameters())/1000:.1f}K")
            
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
            for ep in range(1, epochs + 1):
                model.train()
                for b in tl:
                    ids, tgt, lm = b["input_ids"].to(dev), b["targets"].to(dev), b["loss_mask"].to(dev)
                    loss = model(ids, tgt, lm)["loss"]
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    
            reset_ablations(model)
            base_acc = eval_acc(model, el, dev)
            print(f"    Baseline Acc: Copy={base_acc['Copy']:.3f} | Rev={base_acc['Reverse']:.3f} | Ret={base_acc['Retrieval']:.3f}")
            
            seed_ablations = {"base_acc": base_acc, "head_drops": [], "gate_drops": []}
            
            # Head ablations
            for l_idx in range(4):
                for h_idx in range(4):
                    set_head_ablation(model, l_idx, h_idx)
                    acc = eval_acc(model, el, dev)
                    
                    drop_c = base_acc['Copy'] - acc['Copy']
                    drop_rev = base_acc['Reverse'] - acc['Reverse']
                    drop_ret = base_acc['Retrieval'] - acc['Retrieval']
                    
                    seed_ablations["head_drops"].append({
                        "layer": l_idx, "head": h_idx,
                        "drop_copy": drop_c, "drop_reverse": drop_rev, "drop_retrieval": drop_ret
                    })
                    
            # Gate ablations
            for l_idx in range(4):
                set_gate_ablation(model, l_idx)
                acc = eval_acc(model, el, dev)
                
                drop_c = base_acc['Copy'] - acc['Copy']
                drop_rev = base_acc['Reverse'] - acc['Reverse']
                drop_ret = base_acc['Retrieval'] - acc['Retrieval']
                
                seed_ablations["gate_drops"].append({
                    "layer": l_idx,
                    "drop_copy": drop_c, "drop_reverse": drop_rev, "drop_retrieval": drop_ret
                })
                
            all_results[model_name].append(seed_ablations)

    # Average results
    final_output = {}
    for m_name in all_results:
        final_output[m_name] = {"head_drops": [], "gate_drops": []}
        
        # Heads
        for i in range(16): # 4 layers * 4 heads
            l_idx = i // 4
            h_idx = i % 4
            dc = np.mean([s["head_drops"][i]["drop_copy"] for s in all_results[m_name]])
            drev = np.mean([s["head_drops"][i]["drop_reverse"] for s in all_results[m_name]])
            dret = np.mean([s["head_drops"][i]["drop_retrieval"] for s in all_results[m_name]])
            
            final_output[m_name]["head_drops"].append({
                "layer": l_idx, "head": h_idx,
                "drop_copy": round(float(dc), 4),
                "drop_reverse": round(float(drev), 4),
                "drop_retrieval": round(float(dret), 4)
            })
            
        # Gates
        for i in range(4):
            dc = np.mean([s["gate_drops"][i]["drop_copy"] for s in all_results[m_name]])
            drev = np.mean([s["gate_drops"][i]["drop_reverse"] for s in all_results[m_name]])
            dret = np.mean([s["gate_drops"][i]["drop_retrieval"] for s in all_results[m_name]])
            
            final_output[m_name]["gate_drops"].append({
                "layer": i,
                "drop_copy": round(float(dc), 4),
                "drop_reverse": round(float(drev), 4),
                "drop_retrieval": round(float(dret), 4)
            })

    out_path = Path("fog_exp9_causal_knockout_enhanced.json")
    out_path.write_text(json.dumps(final_output, indent=2))
    print(f"\nSaved enhanced causal knockout results to {out_path}")
    
    # Print top specific heads for FOG
    print("\n--- FOG Motif Specialized Heads (Average over 3 seeds) ---")
    fog_heads = final_output["FOG"]["head_drops"]
    for task in ["copy", "reverse", "retrieval"]:
        key = f"drop_{task}"
        sorted_heads = sorted(fog_heads, key=lambda x: x[key], reverse=True)
        top = sorted_heads[0]
        print(f"Top head for {task.capitalize()}: Layer {top['layer']} Head {top['head']} (Drop: {top[key]:.3f})")
        # How much did this head affect other tasks?
        other1 = "reverse" if task == "copy" else "copy"
        other2 = "retrieval" if task in ["copy", "reverse"] else "reverse"
        print(f"  -> Effect on {other1.capitalize()}: {top[f'drop_{other1}']:.3f}")
        print(f"  -> Effect on {other2.capitalize()}: {top[f'drop_{other2}']:.3f}")

if __name__ == "__main__":
    run_experiment()

"""
ABPT v2 Ablation — Single Cell for Colab/Kaggle
1. Colab: Runtime -> Change runtime type -> T4 GPU -> Save
2. Paste this entire file into one cell
3. Run
"""
import torch, torch.nn as nn, torch.nn.functional as F
import math, time, json, glob, os, gc
import numpy as np
from dataclasses import dataclass
from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════
@dataclass
class C:
    vocab_size:int=4096; d_model:int=512; n_heads:int=8; n_layers:int=8
    d_ff:int=1024; max_seq_len:int=256; dropout:float=0.1
    use_attn_res:bool=True; use_branches:bool=True; n_branches:int=2
    diversity_weight:float=0.1; use_verifier:bool=True
    verifier_entropy_weight:float=0.4; verifier_agreement_weight:float=0.4
    use_plastic:bool=True; plastic_decay:float=0.9; plastic_hidden:int=128
    learning_rate:float=3e-4; weight_decay:float=0.01
    max_steps:int=5000; batch_size:int=64; micro_batch:int=16; accum_steps:int=4
    eval_interval:int=50; gradient_clip:float=1.0
    div_loss_interval:int=5

PRESETS = {
    "baseline-0":          C(use_attn_res=False,use_branches=False,use_verifier=False,use_plastic=False),
    "baseline-1-attnres":  C(use_attn_res=True, use_branches=False,use_verifier=False,use_plastic=False),
    "baseline-2-branches": C(use_attn_res=True, use_branches=True, use_verifier=True, use_plastic=False),
    "baseline-3-plastic":  C(use_attn_res=True, use_branches=False,use_verifier=False,use_plastic=True),
    "full-A":              C(use_attn_res=True, use_branches=True, use_verifier=True, use_plastic=True),
}

# ═══════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════
print("\n=== Loading Data ===")
train_texts, val_texts = None, None

if os.path.exists("/kaggle/input"):
    for d in sorted(os.listdir("/kaggle/input")):
        dp = f"/kaggle/input/{d}"
        files = glob.glob(f"{dp}/**/*", recursive=True)
        parquets = [f for f in files if f.endswith(".parquet")]
        txts = [f for f in files if f.endswith(".txt")]
        csvs = [f for f in files if f.endswith(".csv")]
        print(f"Found: {dp} ({len(parquets)} parquet, {len(txts)} txt, {len(csvs)} csv)")
        if parquets:
            import pandas as pd
            dfs = [pd.read_parquet(f) for f in sorted(parquets)]
            df = pd.concat(dfs, ignore_index=True)
            tcol = [c for c in df.columns if c.lower() in ["text","story","content"]]
            if tcol:
                all_t = df[tcol[0]].dropna().tolist()
                vs = min(5000, len(all_t)//20)
                train_texts, val_texts = all_t[:-vs], all_t[-vs:]
                break
        elif txts:
            all_t = []
            for f in sorted(txts):
                content = open(f).read()
                if "<|endoftext|>" in content:
                    all_t.extend([t.strip() for t in content.split("<|endoftext|>") if t.strip()])
                else:
                    all_t.extend([t.strip() for t in content.split("\n\n") if len(t.strip())>50])
            if all_t:
                vs = min(5000, len(all_t)//20)
                train_texts, val_texts = all_t[:-vs], all_t[-vs:]
                break

if train_texts is None:
    print("No local data found, trying HuggingFace...")
    from datasets import load_dataset
    ds = load_dataset("roneneldan/TinyStories", split="train")
    ds_val = load_dataset("roneneldan/TinyStories", split="validation")
    train_texts = [ds[i]["text"] for i in range(min(len(ds),500000))]
    val_texts = [ds_val[i]["text"] for i in range(min(len(ds_val),10000))]
    del ds, ds_val
    gc.collect()

print(f"Train: {len(train_texts):,} stories, Val: {len(val_texts):,} stories")

# Tokenizer
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
print("Training BPE tokenizer...")
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tr = trainers.BpeTrainer(vocab_size=4096, special_tokens=["<pad>","<eos>"], show_progress=True)
def batch_it(bs=1000):
    for i in range(0,min(len(train_texts),500000),bs): yield train_texts[i:i+bs]
tokenizer.train_from_iterator(batch_it(), trainer=tr)
print(f"Tokenizer: {tokenizer.get_vocab_size()} tokens")

# Tokenize
print("Tokenizing...")
eos = tokenizer.token_to_id("<eos>")
def tok(texts, mx=None):
    ids=[]
    n=min(len(texts),mx) if mx else len(texts)
    for i in range(n):
        ids.extend(tokenizer.encode(texts[i]).ids); ids.append(eos)
    return np.array(ids, dtype=np.uint16)
train_tok = tok(train_texts, 500000)
val_tok = tok(val_texts)
del train_texts, val_texts
gc.collect()
print(f"Train: {len(train_tok):,} tokens, Val: {len(val_tok):,} tokens")

class DS:
    def __init__(s,t,sl): s.t=torch.from_numpy(t.astype(np.int64)); s.sl=sl
    def batch(s,bs,dev="cpu"):
        ix=torch.randint(len(s.t)-s.sl-1,(bs,))
        return torch.stack([s.t[i:i+s.sl] for i in ix]).to(dev), torch.stack([s.t[i+1:i+s.sl+1] for i in ix]).to(dev)

train_ds = DS(train_tok, 256)
val_ds = DS(val_tok, 256)

# ═══════════════════════════════════════════════════════════════
# MODEL
# ═══════════════════════════════════════════════════════════════
class MHA(nn.Module):
    def __init__(s,dm,nh,do=0.1):
        super().__init__()
        s.dm,s.nh,s.dh=dm,nh,dm//nh
        s.wq=nn.Linear(dm,dm,bias=False);s.wk=nn.Linear(dm,dm,bias=False)
        s.wv=nn.Linear(dm,dm,bias=False);s.wo=nn.Linear(dm,dm,bias=False)
        s.do=nn.Dropout(do)
    def forward(s,q,k,v,causal=False):
        B,T,_=q.shape
        q=s.wq(q).view(B,T,s.nh,s.dh).transpose(1,2)
        k=s.wk(k).view(B,-1,s.nh,s.dh).transpose(1,2)
        v=s.wv(v).view(B,-1,s.nh,s.dh).transpose(1,2)
        o=F.scaled_dot_product_attention(q,k,v,is_causal=causal,dropout_p=s.do.p if s.training else 0.0)
        return s.wo(o.transpose(1,2).contiguous().view(B,T,s.dm))

class AR(nn.Module):
    def __init__(s,dm,li):
        super().__init__()
        s.qp=nn.Linear(dm,dm,bias=False);s.kp=nn.Linear(dm,dm,bias=False);s.ln=nn.LayerNorm(dm)
    def forward(s,cur,prev):
        if not prev: return s.ln(cur)
        st=torch.stack(prev,dim=2)
        w=F.softmax(torch.matmul(s.qp(cur).unsqueeze(2),s.kp(st).transpose(-2,-1))/math.sqrt(cur.size(-1)),dim=-1)
        return s.ln(cur+torch.matmul(w,st).squeeze(2))

class TB(nn.Module):
    def __init__(s,c,li):
        super().__init__()
        s.ar=c.use_attn_res
        s.ln1=nn.LayerNorm(c.d_model);s.attn=MHA(c.d_model,c.n_heads,c.dropout)
        s.ln2=nn.LayerNorm(c.d_model)
        s.ff=nn.Sequential(nn.Linear(c.d_model,c.d_ff),nn.GELU(),nn.Dropout(c.dropout),nn.Linear(c.d_ff,c.d_model),nn.Dropout(c.dropout))
        if s.ar: s.attn_res=AR(c.d_model,li)
        else: s.lnr=nn.LayerNorm(c.d_model)
    def forward(s,x,lo):
        n=s.ln1(x);a=s.attn(n,n,n,causal=True)
        x=s.attn_res(a,lo) if s.ar else s.lnr(x+a)
        return x+s.ff(s.ln2(x))

class PL(nn.Module):
    def __init__(s,c):
        super().__init__()
        s.c=c;s.ad=nn.Sequential(nn.Linear(c.d_model,c.plastic_hidden),nn.GELU(),nn.Linear(c.plastic_hidden,c.d_model))
        s.init={n:p.data.clone() for n,p in s.ad.named_parameters()}
    def forward(s,x): return x+s.ad(x)
    def decay(s):
        with torch.no_grad():
            for n,p in s.ad.named_parameters(): p.data.mul_(s.c.plastic_decay).add_(s.init[n].to(p.device),alpha=1-s.c.plastic_decay)

class BR(nn.Module):
    def __init__(s,c):
        super().__init__()
        ts=[0.8+0.4*i/max(c.n_branches-1,1) for i in range(c.n_branches)]
        s.ps=nn.ModuleList([nn.Linear(c.d_model,c.vocab_size,bias=False) for _ in range(c.n_branches)])
        s.ts=ts
    def forward(s,x,compute_div=True):
        bl=[p(x)/t for p,t in zip(s.ps,s.ts)]
        dl=torch.tensor(0.0,device=x.device)
        if compute_div:
            for i in range(len(bl)):
                for j in range(i+1,len(bl)):
                    # Use smaller sample for diversity to save compute
                    pi=F.log_softmax(bl[i],-1);pj=F.log_softmax(bl[j],-1)
                    dl=dl+F.cosine_similarity(pi.reshape(-1,pi.size(-1)),pj.reshape(-1,pj.size(-1)),dim=-1).mean()
            dl=dl/max(len(bl)*(len(bl)-1)//2,1)
        return {"logits":torch.stack(bl).mean(0),"branch_logits":bl,"diversity_loss":dl}

class VR(nn.Module):
    def __init__(s,c): super().__init__();s.ew=c.verifier_entropy_weight;s.aw=c.verifier_agreement_weight
    def forward(s,bl):
        ent=torch.stack([-(F.softmax(b,-1)*F.log_softmax(b,-1)).sum(-1) for b in bl],-1)
        es=1-ent/ent.max(-1,keepdim=True).values.clamp(min=1e-8)
        ps=[F.softmax(b,-1) for b in bl];mp=torch.stack(ps).mean(0)
        ag=torch.stack([F.cosine_similarity(p,mp,dim=-1) for p in ps],-1)
        w=F.softmax((s.ew*es+s.aw*ag)*5,dim=-1)
        return {"logits":(torch.stack(bl,-2)*w.unsqueeze(-1)).sum(-2),"confidence":w.max(-1).values}

class ABPT(nn.Module):
    def __init__(s,c):
        super().__init__();s.c=c
        s.te=nn.Embedding(c.vocab_size,c.d_model);s.pe=nn.Embedding(c.max_seq_len,c.d_model)
        s.dr=nn.Dropout(c.dropout)
        s.bl=nn.ModuleList([TB(c,i) for i in range(c.n_layers)])
        s.ln=nn.LayerNorm(c.d_model)
        if c.use_plastic: s.pl=PL(c)
        if c.use_branches:
            s.br=BR(c)  # branches keep their own weights for diversity
        else:
            s.lm=nn.Linear(c.d_model,c.vocab_size,bias=False)
        if c.use_verifier and c.use_branches: s.vr=VR(c)
    def forward(s,ids,tgt=None,compute_div=True):
        B,T=ids.shape
        x=s.dr(s.te(ids)+s.pe(torch.arange(T,device=ids.device)))
        lo=[x]
        for b in s.bl: x=b(x,lo);lo.append(x)
        h=s.ln(x)
        if s.c.use_plastic: h=s.pl(h)
        r={}
        if s.c.use_branches:
            br=s.br(h,compute_div=compute_div);r["dl"]=br["diversity_loss"];r["bl"]=br["branch_logits"]
            if s.c.use_verifier: vr=s.vr(br["branch_logits"]);lg=vr["logits"];r["conf"]=vr["confidence"]
            else: lg=br["logits"]
        else: lg=s.lm(h)
        r["logits"]=lg
        if tgt is not None:
            ce=F.cross_entropy(lg.view(-1,lg.size(-1)),tgt.view(-1))
            r["loss"]=ce+s.c.diversity_weight*r.get("dl",0);r["ce"]=ce
        return r
    def pc(s): return sum(p.numel() for p in s.parameters())
    def pcs(s): n=s.pc(); return f"{n/1e6:.1f}M" if n>=1e6 else f"{n/1e3:.1f}K"

# ═══════════════════════════════════════════════════════════════
# TRAINING — with GPU memory cleanup
# ═══════════════════════════════════════════════════════════════
def bpb(ce): return ce/math.log(2)

@torch.no_grad()
def ev(m,vds,c,dev,nb=30):
    m.eval();t=0;mb=getattr(c,'micro_batch',c.batch_size)
    for _ in range(nb):
        x,y=vds.batch(mb,dev)
        with torch.amp.autocast(device_type="cuda",dtype=dtype,enabled=dev=="cuda"):
            t+=m(x,y)["ce"].item()
    return t/nb

def free_gpu():
    """Free GPU memory between configs."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        used = torch.cuda.memory_allocated() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  GPU memory: {used:.2f} / {total:.1f} GB")

def train1(name,c,tds,vds,dev):
    print(f"\n{'='*60}\n  {name}\n  attnres={c.use_attn_res} branches={c.use_branches} verifier={c.use_verifier} plastic={c.use_plastic}")
    m=ABPT(c).to(dev);print(f"  Params: {m.pcs()}")
    if dev=="cuda":
        print(f"  GPU mem after model load: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    opt=torch.optim.AdamW(m.parameters(),lr=c.learning_rate,weight_decay=c.weight_decay)
    sc=torch.amp.GradScaler(enabled=dev=="cuda")
    accum=getattr(c,'accum_steps',1);mb=getattr(c,'micro_batch',c.batch_size)
    max_st=getattr(c,'max_steps',2000);st=0;log=[];t0=time.time()
    while st<max_st:
        m.train();opt.zero_grad()
        div_int=getattr(c,'div_loss_interval',1);do_div=(st%div_int==0)
        o=None
        for _acc in range(accum):
            x,y=tds.batch(mb,dev)
            with torch.amp.autocast(device_type="cuda",dtype=dtype,enabled=dev=="cuda"):
                o=m(x,y,compute_div=do_div);loss=o["loss"]/accum
            sc.scale(loss).backward()
        sc.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(m.parameters(),c.gradient_clip);sc.step(opt);sc.update()
        if c.use_plastic: m.pl.decay()
        if st%c.eval_interval==0:
            vc=ev(m,vds,c,dev);vb=bpb(vc);el=time.time()-t0
            e={"step":st,"val_bpb":round(vb,4),"loss":round(o["ce"].item(),4),"time":round(el,1)}
            if "bl" in o:
                pi=F.softmax(o["bl"][0],-1).reshape(-1,o["bl"][0].size(-1))
                pj=F.softmax(o["bl"][1],-1).reshape(-1,o["bl"][1].size(-1))
                e["bdiv"]=round((1-F.cosine_similarity(pi,pj,dim=-1).mean().item()),4)
            if "conf" in o: e["conf"]=round(o["conf"].mean().item(),4)
            log.append(e)
            ex=""
            if "bdiv" in e: ex+=f" | div={e['bdiv']:.4f}"
            if "conf" in e: ex+=f" | conf={e['conf']:.4f}"
            print(f"  step {st:5d} | val_bpb {vb:.4f} | loss {o['ce'].item():.4f} | {el:.0f}s{ex}")
        st+=1
    fc=ev(m,vds,c,dev,50);fb=bpb(fc);tt=time.time()-t0
    print(f"  FINAL: val_bpb={fb:.4f} | {st} steps in {tt:.0f}s ({st/tt:.1f} st/s)")
    result = {"preset":name,"params":m.pcs(),"final_val_bpb":round(fb,4),"steps":st,"time":round(tt,1),"sps":round(st/tt,2),"log":log}
    # Free model from GPU
    del m, opt, sc
    free_gpu()
    return result

# ═══════════════════════════════════════════════════════════════
# RUN ABLATION
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*60}\nStarting ablation: {list(PRESETS.keys())}\n{'='*60}")
results=[]
for name,c in PRESETS.items():
    free_gpu()
    r=train1(name,c,train_ds,val_ds,device)
    results.append(r)
    with open("ablation_results.json","w") as f: json.dump(results,f,indent=2)

# ═══════════════════════════════════════════════════════════════
# RESULTS
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*60}\nABLATION COMPLETE\n{'='*60}")
b0=next(r["final_val_bpb"] for r in results if r["preset"]=="baseline-0")
print(f"\n{'Preset':<25} {'Params':>8} {'val_bpb':>10} {'vs base':>10} {'steps/s':>10}")
print("-"*67)
for r in sorted(results, key=lambda x:x["final_val_bpb"]):
    d=(r["final_val_bpb"]-b0)/b0*100
    print(f"{r['preset']:<25} {r['params']:>8} {r['final_val_bpb']:>10.4f} {d:>+9.2f}% {r['sps']:>10.1f}")

# Plot
try:
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,1,figsize=(12,6))
    for r in results:
        ss=[e["step"] for e in r["log"]];bb=[e["val_bpb"] for e in r["log"]]
        ax.plot(ss,bb,label=f"{r['preset']} ({r['params']})")
    ax.set_xlabel("Step");ax.set_ylabel("val_bpb (lower=better)")
    ax.set_title("ABPT v2 Ablation");ax.legend();ax.grid(True,alpha=0.3)
    plt.tight_layout();plt.savefig("ablation.png",dpi=150);plt.show()
    # Branch diversity
    fig,ax=plt.subplots(1,1,figsize=(12,4))
    has=False
    for r in results:
        dv=[(e["step"],e["bdiv"]) for e in r["log"] if "bdiv" in e]
        if dv: ss,vv=zip(*dv);ax.plot(ss,vv,label=r["preset"]);has=True
    if has:
        ax.set_xlabel("Step");ax.set_ylabel("Branch Diversity");ax.set_title("Branch Diversity");ax.legend();ax.grid(True,alpha=0.3)
        plt.tight_layout();plt.savefig("branch_div.png",dpi=150);plt.show()
except: print("(matplotlib plots skipped)")

print("\nDone! Results saved to ablation_results.json")

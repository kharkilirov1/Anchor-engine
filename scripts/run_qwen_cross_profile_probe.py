"""Cross-profile tail_retention_ratio probe вЂ” robust single-run."""
from __future__ import annotations
import argparse, json, sys, traceback
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.qwen_anchor_geometry_cases import make_qwen_anchor_geometry_cases
from src.model.config import TOY_CONFIG
from src.model.qwen_anchor_overlay import QwenAnchorOverlay
from src.utils.anchor_geometry import (
    compute_geometry_metrics, extract_delta_vectors, match_anchor_span,
)

CS, CE = 4, 8
PROP_S, INTEG_E, HAND_S = 9, 23, 24
MAX_LEN, MAX_NEW = 160, 120
SEED = 7

KW = {
    "strictly_vegan_meal_plan_policy": {"pos": ["vegan","plant-based","dairy-free"], "neg": ["meat","chicken","beef","fish","dairy","milk","cheese","egg"]},
    "async_fastapi_service_architecture_policy": {"pos": ["async","await","fastapi"], "neg": ["flask","django"]},
    "json_only_response_format_policy": {"pos": ["json","{","}"], "neg": ["here is","sure"]},
    "proof_by_contradiction_reasoning_steps": {"pos": ["assume","contradiction","therefore","suppose"], "neg": []},
    "binary_search_update_loop_procedure": {"pos": ["low","high","mid","while"], "neg": []},
    "dependency_injection_request_flow_sequence": {"pos": ["inject","dependency","container","provider"], "neg": []},
}

def auc(r1, s, e):
    vs = [r1.get(str(l), 0.0) or 0.0 for l in range(s, e+1)]
    return float(np.trapezoid(vs)) if len(vs) >= 2 else 0.0

def tail_retention(r1, nl):
    ea = auc(r1, CS, CE)
    return auc(r1, PROP_S, INTEG_E) / ea if ea > 0 else None

def score(text, grp):
    sp = KW.get(grp, {"pos":[], "neg":[]})
    lo = text.lower()
    ph = sum(1 for t in sp["pos"] if t in lo)
    nh = sum(1 for t in sp["neg"] if t in lo)
    return 1.0 if (ph >= 2 and nh == 0) else 0.0

def spearman(xs, ys):
    if len(xs) < 3: return None
    def rank(v):
        s = sorted(enumerate(v), key=lambda x: x[1])
        r = [0.0]*len(v)
        for i,(j,_) in enumerate(s,1): r[j]=float(i)
        return r
    n=len(xs); rx=rank(xs); ry=rank(ys)
    d2=sum((a-b)**2 for a,b in zip(rx,ry))
    return 1.0-(6*d2)/(n*(n**2-1))

def run_profile(overlay, profile, nl, device):
    cases = make_qwen_anchor_geometry_cases(anchor_span_profile=profile)
    trs, css = [], []
    for case in cases:
        try:
            enc = overlay.tokenizer(case.prompt, truncation=True, max_length=MAX_LEN, return_tensors="pt")
            batch = {k:v.to(device) for k,v in enc.items() if isinstance(v, torch.Tensor)}
            ids = [int(t) for t in batch["input_ids"][0].tolist()]
            sm = match_anchor_span(
                text=case.prompt,
                anchor_text=case.anchor_text,
                input_ids=ids,
                tokenizer=overlay.tokenizer,
                offsets=None,
            )
            if sm is None: continue
            with torch.no_grad():
                out = overlay.base_model(input_ids=batch["input_ids"], attention_mask=batch.get("attention_mask"), output_hidden_states=True, return_dict=True)
            hs = out.hidden_states
            r1 = {}
            for l in range(nl):
                dv = extract_delta_vectors(hs[l+1][0], sm.token_start, sm.token_end)
                gm = compute_geometry_metrics(dv)
                r1[str(l)] = float(gm.get("rank1_explained_variance") or 0)
            tr = tail_retention(r1, nl)
            if tr is None: continue
            inp = overlay.tokenizer([case.prompt], truncation=True, max_length=MAX_LEN, return_tensors="pt", padding=False)
            iids = inp["input_ids"].to(device)
            am = inp.get("attention_mask")
            if am is not None: am = am.to(device)
            np_tok = int(iids.shape[1])
            with torch.no_grad():
                gen = overlay.base_model.generate(iids, attention_mask=am, max_new_tokens=MAX_NEW, do_sample=False)
            txt = overlay.tokenizer.decode(gen[0][np_tok:], skip_special_tokens=True)
            cs = score(txt, case.anchor_group)
            trs.append(tr); css.append(cs)
            print(json.dumps({"case": case.name, "profile": profile, "tr": round(tr,4), "cs": cs}))
        except Exception as e:
            print(json.dumps({"case_error": case.name, "profile": profile, "err": str(e)[:200]}))
    rho = spearman(trs, css)
    return {"n": len(trs), "tail_retention_rho": rho, "mean_tr": float(np.mean(trs)) if trs else None, "mean_cs": float(np.mean(css)) if css else None}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args, _ = parser.parse_known_args()
    torch.manual_seed(SEED)
    dev = torch.device(args.device)
    print(f"[CrossProbe] Loading {args.model_name}...")
    ov = QwenAnchorOverlay.from_pretrained(args.model_name, config=TOY_CONFIG)
    ov.to(dev); ov.eval()
    nl = int(ov.model_num_hidden_layers)
    print(f"[CrossProbe] layers={nl}")
    results = {}
    for prof in ["short", "medium", "long"]:
        print(f"\n[CrossProbe] === Profile: {prof} ===")
        try:
            results[prof] = run_profile(ov, prof, nl, dev)
            print(json.dumps({"profile_done": prof, **results[prof]}))
        except Exception as e:
            results[prof] = {"error": str(e)[:300]}
            print(json.dumps({"profile_error": prof, "err": str(e)[:300]}))
            traceback.print_exc()
    out = {"cross_profile": results, "model": args.model_name, "seed": SEED}
    archive = ROOT / "archive"
    archive.mkdir(exist_ok=True)
    fp = archive / "qwen35_4b_cross_profile_probe.json"
    fp.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n[CrossProbe] Saved: {fp}")
    print("\n===FINAL_RESULT===")
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()

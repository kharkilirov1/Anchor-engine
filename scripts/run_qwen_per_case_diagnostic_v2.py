"""Per-case tail_retention diagnostic — parameterized profile."""
from __future__ import annotations
import argparse, json, sys, traceback
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.qwen_anchor_geometry_cases import make_qwen_anchor_geometry_cases
from src.model.config import TOY_CONFIG
from src.model.qwen_anchor_overlay import QwenAnchorOverlay
from src.utils.anchor_geometry import extract_delta_vectors, compute_geometry_metrics, match_anchor_span

CS, CE = 4, 8
PROP_S, INTEG_E = 9, 23
KW = {
    "strictly_vegan_meal_plan_policy": {"pos": ["vegan","plant-based","dairy-free"], "neg": ["meat","chicken","beef","fish","dairy","milk","cheese","egg"]},
    "async_fastapi_service_architecture_policy": {"pos": ["async","await","fastapi"], "neg": ["flask","django"]},
    "json_only_response_format_policy": {"pos": ["json","{","}"], "neg": ["here is","sure"]},
    "proof_by_contradiction_reasoning_steps": {"pos": ["assume","contradiction","therefore","suppose"], "neg": []},
    "binary_search_update_loop_procedure": {"pos": ["low","high","mid","while"], "neg": []},
    "dependency_injection_request_flow_sequence": {"pos": ["inject","dependency","container","provider"], "neg": []},
}

def auc(r1, s, e):
    vs = [r1.get(str(l), 0.0) for l in range(s, e+1)]
    return float(np.trapezoid(vs)) if len(vs) >= 2 else 0.0

def tail_ret(r1):
    ea = auc(r1, CS, CE)
    return auc(r1, PROP_S, INTEG_E) / ea if ea > 1e-9 else None

def cscore(text, grp):
    sp = KW.get(grp, {"pos":[], "neg":[]})
    lo = text.lower()
    return 1.0 if (sum(1 for t in sp["pos"] if t in lo) >= 2 and sum(1 for t in sp["neg"] if t in lo) == 0) else 0.0

def spearman(xs, ys):
    n = len(xs)
    if n < 3: return None
    def rank(v):
        s = sorted(range(n), key=lambda i: v[i])
        r = [0.0]*n
        for i,j in enumerate(s,1): r[j]=float(i)
        return r
    rx, ry = rank(xs), rank(ys)
    d2 = sum((a-b)**2 for a,b in zip(rx,ry))
    return round(1.0 - 6*d2/(n*(n**2-1)), 6)

def select_cases(cases, group_case_cap):
    if group_case_cap <= 0:
        return list(cases)
    selected = []
    seen = defaultdict(int)
    for case in cases:
        if seen[case.anchor_group] >= group_case_cap:
            continue
        selected.append(case)
        seen[case.anchor_group] += 1
    return selected

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--model-name", default="Qwen/Qwen3.5-4B")
    pa.add_argument("--profile", default="short", choices=["short","medium","long"])
    pa.add_argument("--max-new-tokens", type=int, default=48)
    pa.add_argument("--group-case-cap", type=int, default=1)
    pa.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args, _ = pa.parse_known_args()
    dev = torch.device(args.device)
    torch.manual_seed(7)
    print(f"[Diag] Loading {args.model_name} on {dev}, profile={args.profile}")
    ov = QwenAnchorOverlay.from_pretrained(args.model_name, config=TOY_CONFIG)
    ov.to(dev); ov.eval()
    nl = int(ov.model_num_hidden_layers)
    all_cases = make_qwen_anchor_geometry_cases(anchor_span_profile=args.profile)
    cases = select_cases(all_cases, args.group_case_cap)
    rows, trs, css = [], [], []
    for c in cases:
        row = {"name": c.name, "group": c.anchor_group, "tr": None, "cs": None, "err": None}
        try:
            enc = ov.tokenizer(c.prompt, truncation=True, max_length=160, return_tensors="pt")
            b = {k:v.to(dev) for k,v in enc.items() if isinstance(v, torch.Tensor)}
            ids = b["input_ids"][0].tolist()
            sm = match_anchor_span(
                text=c.prompt,
                anchor_text=c.anchor_text,
                input_ids=ids,
                tokenizer=ov.tokenizer,
                offsets=None,
            )
            if sm is None:
                row["err"] = "no_span_match"; rows.append(row); continue
            with torch.no_grad():
                out = ov.base_model(input_ids=b["input_ids"], attention_mask=b.get("attention_mask"), output_hidden_states=True, return_dict=True)
            r1 = {}
            for l in range(nl):
                dv = extract_delta_vectors(out.hidden_states[l+1][0], sm.token_start, sm.token_end)
                gm = compute_geometry_metrics(dv)
                r1[str(l)] = float(gm.get("rank1_explained_variance") or 0)
            tr = tail_ret(r1)
            if tr is None:
                row["err"] = "zero_early_auc"; rows.append(row); continue
            with torch.no_grad():
                gen = ov.base_model.generate(
                    b["input_ids"],
                    attention_mask=b.get("attention_mask"),
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                )
            txt = ov.tokenizer.decode(gen[0][b["input_ids"].shape[1]:], skip_special_tokens=True)
            cs = cscore(txt, c.anchor_group)
            row["tr"] = round(tr, 6); row["cs"] = cs
            trs.append(tr); css.append(cs)
        except Exception as e:
            row["err"] = str(e)[:200]
            traceback.print_exc()
        rows.append(row)
        print(json.dumps(row))
    rho = spearman(trs, css)
    result = {"spearman_rho": rho, "n_valid": len(trs), "n_total": len(cases), "n_source_cases": len(all_cases), "mean_tr": round(float(np.mean(trs)),4) if trs else None, "mean_cs": round(float(np.mean(css)),4) if css else None, "model": args.model_name, "profile": args.profile, "max_new_tokens": args.max_new_tokens, "group_case_cap": args.group_case_cap, "cases": rows}
    Path(ROOT/"archive").mkdir(exist_ok=True)
    Path(ROOT/f"archive/qwen35_4b_per_case_diagnostic_{args.profile}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print("\n===FINAL_RESULT===")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()

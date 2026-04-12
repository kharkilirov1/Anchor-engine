import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from collections import defaultdict

# --- CONFIGURATION ---
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
OUTPUT_DIR = "fog_cartographer_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/figures", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/notes", exist_ok=True)

# --- TASKS ---
def generate_tasks(num_samples=20):
    tasks = {
        "context_revision": [],
        "retrieval": [],
        "sequential_local": [],
        "multi_step": []
    }
    
    # 1. Context Revision
    # "The password is {A}. Wait, no, the password is {B}. What is the password? {B}"
    for i in range(num_samples):
        a = np.random.randint(1000, 9999)
        b = np.random.randint(1000, 9999)
        while a == b: b = np.random.randint(1000, 9999)
        prompt = f"The secret code is {a}. Wait, I made a mistake, the secret code is actually {b}. The correct secret code is"
        tasks["context_revision"].append(prompt)
        
    # 2. Retrieval
    for i in range(num_samples):
        k1, k2, k3 = np.random.choice(["Alpha", "Beta", "Gamma", "Delta", "Echo"], 3, replace=False)
        v1, v2, v3 = np.random.randint(10, 99, 3)
        prompt = f"Data: {k1}={v1}, {k2}={v2}, {k3}={v3}. Question: What is the value of {k2}? Answer:"
        tasks["retrieval"].append(prompt)
        
    # 3. Sequential Local (Copy)
    words = ["apple", "banana", "cherry", "date", "elderberry", "fig", "grape"]
    for i in range(num_samples):
        seq = np.random.choice(words, 4, replace=False)
        prompt = f"Sequence: {seq[0]} {seq[1]} {seq[2]} {seq[3]}. Repeat the sequence exactly: {seq[0]} {seq[1]}"
        tasks["sequential_local"].append(prompt)
        
    # 4. Multi-step reasoning
    names = ["Alice", "Bob", "Charlie", "David"]
    cities = ["Paris", "London", "Tokyo", "Berlin"]
    for i in range(num_samples):
        n1, n2 = np.random.choice(names, 2, replace=False)
        c1, c2 = np.random.choice(cities, 2, replace=False)
        prompt = f"Fact 1: {n1} lives in the same city as {n2}. Fact 2: {n2} lives in {c1}. Question: Where does {n1} live? Answer:"
        tasks["multi_step"].append(prompt)
        
    return tasks

# --- METRICS ---
def compute_normalized_entropy(attn_weights):
    # attn_weights: [B, H, T, T]
    B, H, T, _ = attn_weights.shape
    epsilon = 1e-9
    entropy = -torch.sum(attn_weights * torch.log(attn_weights + epsilon), dim=-1) # [B, H, T]
    max_entropy = torch.log(torch.arange(1, T + 1, device=attn_weights.device, dtype=torch.float32))
    max_entropy[0] = 1.0 # avoid div by zero
    norm_entropy = entropy / max_entropy.view(1, 1, T)
    # We care about the entropy of the last token (the one making the prediction)
    last_token_entropy = norm_entropy[:, :, -1].mean(dim=0) # [H]
    return last_token_entropy.cpu().numpy()

def compute_locality(attn_weights, window=2):
    # How much attention is focused on the last 'window' tokens
    B, H, T, _ = attn_weights.shape
    last_token_attn = attn_weights[:, :, -1, :] # [B, H, T]
    # sum of attention weights on the last `window` tokens
    start_idx = max(0, T - window)
    local_attn = last_token_attn[:, :, start_idx:T].sum(dim=-1) # [B, H]
    return local_attn.mean(dim=0).cpu().numpy()

def compute_clad(hidden_states):
    # hidden_states is a tuple of (layers+1) tensors of shape [B, T, D]
    clad_per_layer = []
    # We look at the representation of the last token
    for l in range(1, len(hidden_states)):
        h_prev = hidden_states[l-1][:, -1, :] # [B, D]
        h_curr = hidden_states[l][:, -1, :]   # [B, D]
        sim = torch.nn.functional.cosine_similarity(h_prev, h_curr, dim=-1)
        drift = 1.0 - sim.mean().item()
        clad_per_layer.append(drift)
    return clad_per_layer

# --- HOOKS FOR FFN ---
def attach_ffn_hooks(model):
    hooks = []
    ffn_stats = defaultdict(list)
    
    def create_hook(layer_idx):
        def hook(module, args):
            # args is a tuple, we want args[0]
            acts = args[0] # [B, T, Intermediate_Dim]
            last_token_acts = acts[:, -1, :]
            # Measure sparsity (fraction of near-zero activations)
            sparsity = (last_token_acts.abs() < 1e-3).float().mean().item()
            # Measure variance (polarization proxy)
            variance = last_token_acts.var(dim=-1).mean().item()
            ffn_stats[layer_idx].append({"sparsity": sparsity, "variance": variance})
        return hook

    for i, layer in enumerate(model.model.layers):
        # Hook into down_proj to see what it receives (the activated state)
        h = layer.mlp.down_proj.register_forward_pre_hook(create_hook(i))
        hooks.append(h)
    return hooks, ffn_stats

# --- MAIN ENGINE ---
def run_cartographer(smoke_test=False):
    print(f"\nLoading tokenizer and model: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        device_map="cpu", 
        torch_dtype=torch.float32,
        trust_remote_code=True,
        attn_implementation="eager"
    )
    model.eval()
    
    # Save model info
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    d_model = model.config.hidden_size
    model_info = {
        "model_name": MODEL_NAME,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "d_model": d_model,
        "vocab_size": model.config.vocab_size
    }
    with open(f"{OUTPUT_DIR}/notes/model_info.json", "w") as f:
        json.dump(model_info, f, indent=2)
        
    num_samples = 2 if smoke_test else 30
    tasks = generate_tasks(num_samples)
    
    results = defaultdict(lambda: {"entropy": [], "locality": [], "clad": [], "ffn_var": [], "ffn_sparse": []})
    
    hooks, ffn_stats = attach_ffn_hooks(model)
    
    print(f"Running {'SMOKE TEST' if smoke_test else 'FULL TEST'}...")
    
    for task_name, prompts in tasks.items():
        print(f"Processing task: {task_name}")
        for prompt in tqdm(prompts):
            inputs = tokenizer(prompt, return_tensors="pt")
            
            with torch.no_grad():
                # Clear ffn_stats for this run
                for k in list(ffn_stats.keys()):
                    ffn_stats[k].clear()
                    
                outputs = model(**inputs, output_attentions=True, output_hidden_states=True)
                
                # 1. Attentions [layers, B, H, T, T]
                attentions = outputs.attentions
                layer_entropy = []
                layer_locality = []
                for l_idx in range(num_layers):
                    ent = compute_normalized_entropy(attentions[l_idx])
                    loc = compute_locality(attentions[l_idx], window=3)
                    layer_entropy.append(ent)
                    layer_locality.append(loc)
                    
                results[task_name]["entropy"].append(layer_entropy)
                results[task_name]["locality"].append(layer_locality)
                
                # 2. Hidden States (CLAD)
                clad = compute_clad(outputs.hidden_states)
                results[task_name]["clad"].append(clad)
                
                # 3. FFN Stats
                var_per_layer = []
                sparse_per_layer = []
                for l_idx in range(num_layers):
                    var_per_layer.append(ffn_stats[l_idx][0]["variance"])
                    sparse_per_layer.append(ffn_stats[l_idx][0]["sparsity"])
                results[task_name]["ffn_var"].append(var_per_layer)
                results[task_name]["ffn_sparse"].append(sparse_per_layer)

    for h in hooks:
        h.remove()
        
    # Aggregate results
    aggregated = {}
    for task_name in tasks.keys():
        aggregated[task_name] = {
            "mean_entropy": np.mean(results[task_name]["entropy"], axis=0).tolist(), # [Layers, Heads]
            "mean_locality": np.mean(results[task_name]["locality"], axis=0).tolist(),
            "mean_clad": np.mean(results[task_name]["clad"], axis=0).tolist(), # [Layers]
            "mean_ffn_var": np.mean(results[task_name]["ffn_var"], axis=0).tolist(), # [Layers]
            "mean_ffn_sparse": np.mean(results[task_name]["ffn_sparse"], axis=0).tolist()
        }
        
    with open(f"{OUTPUT_DIR}/motif_map.json", "w") as f:
        json.dump(aggregated, f, indent=2)
        
    return aggregated, num_layers, num_heads

def plot_heatmaps(aggregated, num_layers, num_heads):
    print("Generating heatmaps...")
    tasks = list(aggregated.keys())
    
    # 1. Entropy Heatmaps (Layer vs Head)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    for i, task in enumerate(tasks):
        data = np.array(aggregated[task]["mean_entropy"])
        im = axes[i].imshow(data, cmap='viridis', aspect='auto')
        axes[i].set_title(f"{task} - Attn Entropy")
        axes[i].set_xlabel("Head")
        axes[i].set_ylabel("Layer")
        plt.colorbar(im, ax=axes[i])
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/figures/attn_entropy_heatmaps.png", dpi=200)
    plt.close()
    
    # 2. CLAD and FFN Stats (Layer-wise)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for task in tasks:
        clad = aggregated[task]["mean_clad"]
        ffn_var = aggregated[task]["mean_ffn_var"]
        ffn_sparse = aggregated[task]["mean_ffn_sparse"]
        layers = range(num_layers)
        
        axes[0].plot(layers, clad, label=task, marker='o', markersize=4)
        axes[1].plot(layers, ffn_var, label=task, marker='o', markersize=4)
        axes[2].plot(layers, ffn_sparse, label=task, marker='o', markersize=4)
        
    axes[0].set_title("Cross-Layer Angular Drift (CLAD)")
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("Drift (1 - CosSim)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    axes[1].set_title("FFN Activation Variance")
    axes[1].set_xlabel("Layer")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    axes[2].set_title("FFN Activation Sparsity")
    axes[2].set_xlabel("Layer")
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/figures/layerwise_stats.png", dpi=200)
    plt.close()
    
def generate_reports(aggregated, num_layers, num_heads):
    print("Generating markdown reports...")
    
    # Identify top heads
    top_heads = {}
    for task in aggregated.keys():
        entropy = np.array(aggregated[task]["mean_entropy"])
        # Low entropy = sharp focus
        flat_indices_sharp = np.argsort(entropy, axis=None)[:3]
        sharp_coords = [(int(i // num_heads), int(i % num_heads), float(entropy.flatten()[i])) for i in flat_indices_sharp]
        
        # High entropy = broad focus
        flat_indices_broad = np.argsort(entropy, axis=None)[-3:][::-1]
        broad_coords = [(int(i // num_heads), int(i % num_heads), float(entropy.flatten()[i])) for i in flat_indices_broad]
        
        top_heads[task] = {"sharpest": sharp_coords, "broadest": broad_coords}
        
    # Task selectivity
    all_ents = np.array([aggregated[t]["mean_entropy"] for t in aggregated.keys()])
    task_variance = np.var(all_ents, axis=0)
    flat_most_selective = np.argsort(task_variance, axis=None)[-5:][::-1]
    selective_coords = [(int(i // num_heads), int(i % num_heads), float(task_variance.flatten()[i])) for i in flat_most_selective]

    report = f"# FOG Cartographer: Descriptive Motif Signatures Report\n\n"
    report += "## Overview\n"
    report += "This report maps the internal representations of a pre-trained LLM across 4 distinct task classes. "
    report += "The goal is to identify descriptive signatures consistent with the FOG (Finite Operator Grammar) hypothesis "
    report += "without making causal claims.\n\n"
    
    report += "## 1. Top Task-Selective Heads\n"
    report += "These heads show the highest variance in attention entropy across tasks, suggesting they shift behavior depending on context.\n"
    for l, h, var in selective_coords:
        report += f"- **Layer {l}, Head {h}** (Variance: {var:.4f})\n"
        
    report += "\n## 2. Task-Specific Profiles\n"
    for task in aggregated.keys():
        report += f"### {task.replace('_', ' ').title()}\n"
        report += "- **Sharpest Heads (Potential Local/Compare):** "
        report += ", ".join([f"L{l}H{h} ({v:.2f})" for l,h,v in top_heads[task]["sharpest"]]) + "\n"
        report += "- **Broadest Heads (Potential Memory/Context):** "
        report += ", ".join([f"L{l}H{h} ({v:.2f})" for l,h,v in top_heads[task]["broadest"]]) + "\n"
        
    report += "\n## 3. FFN and CLAD Observations\n"
    report += "- Review `figures/layerwise_stats.png` to observe macroscopic trends.\n"
    
    report += "\n## 4. Recommendations for Next Stage (Causal Cartography / Scalpel)\n"
    report += "1. **Target Selective Heads:** The highly task-selective heads (Section 1) are prime candidates for causal intervention (knockout/amplification).\n"
    report += "2. **Context Revision Interventions:** For the `context_revision` task, consider attenuating 'broad' heads and amplifying 'sharp' heads during causal validation to see if error rates drop.\n"
    
    with open(f"{OUTPUT_DIR}/summary_report.md", "w") as f:
        f.write(report)
        
    limitations = """# Limitations of the Cartography Report

1. **Descriptive, Not Causal:** This map shows correlations (signatures) but does not prove that a specific head *causes* a behavior. A head with high entropy might just be a "dead" head doing uniform attention, rather than a functional "memory" motif.
2. **Proxy Metrics:** We use normalized entropy and FFN variance as proxies for FOG theoretical constructs. These are heuristic alignments.
3. **Synthetic Prompting:** The tasks use simple, rigid templates. Real-world natural language is messier.
4. **No OOD Verification:** Patterns found here might not generalize beyond the exact syntax used in the prompts.
"""
    with open(f"{OUTPUT_DIR}/notes/limitations.md", "w") as f:
        f.write(limitations)

if __name__ == "__main__":
    print("--- FOG CARTOGRAPHER ---")
    run_cartographer(smoke_test=True)
    aggregated, num_layers, num_heads = run_cartographer(smoke_test=False)
    plot_heatmaps(aggregated, num_layers, num_heads)
    generate_reports(aggregated, num_layers, num_heads)
    print(f"Cartography complete. Check {OUTPUT_DIR}/ for results.")

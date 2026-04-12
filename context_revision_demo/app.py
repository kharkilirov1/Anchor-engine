import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
import torch

from model_controller import ModelController
from metrics import calculate_metrics
from scenarios import PRESETS, get_preset_names, get_preset_by_name

# Initialize Controller
controller = ModelController()

# --- HELPER FUNCTIONS ---
def generate_interpretation(mode, scale, baseline_margin, run_margin):
    if mode == "Baseline" or scale == 1.0:
        return "Baseline run with no intervention."
        
    delta = run_margin - baseline_margin
    direction = "increased" if delta > 0 else "decreased"
    
    if mode == "Selective":
        text = f"Selective intervention {direction} the new-vs-old margin by {delta:+.2f} relative to baseline."
        if scale < 1.0 and delta > 0:
            text += "\nThis is consistent with revision-sensitive heads contributing to old-context persistence: attenuating them helps the model accept the new fact."
        elif scale > 1.0 and delta < 0:
            text += "\nAmplifying these heads forces the model to cling to the old fact."
        return text
        
    if mode == "Control":
        text = f"Control intervention changed the margin by {delta:+.2f} relative to baseline."
        if abs(delta) < 0.5:
            text += "\nControl intervention did not produce a strong directional shift, confirming the specificity of the selective heads."
        return text
        
    return ""

def load_preset(preset_name):
    p = get_preset_by_name(preset_name)
    return p["prompt"], p["old_fact"], p["new_fact"]

def run_single(prompt, old_str, new_str, mode, scale, top_k):
    # First run baseline to get baseline margin for comparison
    _, b_probs, old_id, new_id, _, _, _ = controller.predict_next_token(prompt, old_str, new_str, "Baseline", 1.0)
    b_margin = b_probs[new_id].item() - b_probs[old_id].item() # we can just use prob diff or logit diff
    b_logit_margin = torch.log(b_probs[new_id] + 1e-9) - torch.log(b_probs[old_id] + 1e-9)
    
    # Run targeted
    logits, probs, o_id, n_id, _, _, full_text = controller.predict_next_token(prompt, old_str, new_str, mode, scale)
    
    # Actually calculate correctly from logits
    b_metrics = calculate_metrics(torch.log(b_probs + 1e-9), b_probs, old_id, new_id, controller.tokenizer, top_k)
    r_metrics = calculate_metrics(logits, probs, o_id, n_id, controller.tokenizer, top_k)
    
    # Generate Answer Token
    ans_id = logits.argmax().item()
    ans_text = controller.tokenizer.decode(ans_id)
    
    # Metrics DataFrame
    metrics_df = pd.DataFrame([
        {"Metric": "Margin (Logit New - Old)", "Value": f"{r_metrics['margin']:+.3f}"},
        {"Metric": f"Rank of New Token ({repr(new_str)})", "Value": str(r_metrics['rank_new'])},
        {"Metric": f"Rank of Old Token ({repr(old_str)})", "Value": str(r_metrics['rank_old'])},
        {"Metric": "New in Top-K?", "Value": "Yes" if r_metrics['in_top_k_new'] else "No"},
        {"Metric": "Old in Top-K?", "Value": "Yes" if r_metrics['in_top_k_old'] else "No"}
    ])
    
    # Top 10 Table
    top_10_df = pd.DataFrame(r_metrics['top_10_table'])
    
    # Interpretation
    interpretation = generate_interpretation(mode, scale, b_metrics['margin'], r_metrics['margin'])
    
    return ans_text, full_text, metrics_df, top_10_df, interpretation

def run_dose_sweep(prompt, old_str, new_str):
    scales = [0.0, 0.5, 1.0, 1.5, 2.0]
    
    sel_margins = []
    ctrl_margins = []
    
    for s in scales:
        s_log, s_prob, o_id, n_id, _, _, _ = controller.predict_next_token(prompt, old_str, new_str, "Selective", s)
        c_log, c_prob, _, _, _, _, _ = controller.predict_next_token(prompt, old_str, new_str, "Control", s)
        
        s_m = s_log[n_id].item() - s_log[o_id].item()
        c_m = c_log[n_id].item() - c_log[o_id].item()
        
        sel_margins.append(s_m)
        ctrl_margins.append(c_m)
        
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(scales, sel_margins, marker='o', label="Selective Heads", color="blue")
    ax.plot(scales, ctrl_margins, marker='s', label="Control Heads", color="gray", linestyle='--')
    ax.axvline(1.0, color="red", linestyle=":", alpha=0.5, label="Baseline")
    ax.set_title("Dose-Response: Margin Shift vs Intervention Scale")
    ax.set_xlabel("Head Output Scale Factor (<1 Attenuate, >1 Amplify)")
    ax.set_ylabel("Margin (Logit New - Logit Old)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plot_path = "outputs/dose_sweep.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    
    return plot_path

def run_batch_eval():
    summary_data = []
    
    for p in PRESETS:
        # Baseline
        b_log, b_prob, o_id, n_id, _, _, _ = controller.predict_next_token(p["prompt"], p["old_fact"], p["new_fact"], "Baseline", 1.0)
        b_m = b_log[n_id].item() - b_log[o_id].item()
        
        # Selective Attenuation 0.0
        s_log, s_prob, _, _, _, _, _ = controller.predict_next_token(p["prompt"], p["old_fact"], p["new_fact"], "Selective", 0.0)
        s_m = s_log[n_id].item() - s_log[o_id].item()
        
        # Control Attenuation 0.0
        c_log, c_prob, _, _, _, _, _ = controller.predict_next_token(p["prompt"], p["old_fact"], p["new_fact"], "Control", 0.0)
        c_m = c_log[n_id].item() - c_log[o_id].item()
        
        summary_data.append({
            "Scenario": p["name"],
            "Baseline Margin": f"{b_m:+.2f}",
            "Selective (Scale=0) Margin": f"{s_m:+.2f}",
            "Control (Scale=0) Margin": f"{c_m:+.2f}",
            "Selective Delta": f"{(s_m - b_m):+.2f}",
            "Control Delta": f"{(c_m - b_m):+.2f}"
        })
        
    df = pd.DataFrame(summary_data)
    
    # Save JSON
    with open("outputs/batch_eval_summary.json", "w") as f:
        json.dump(summary_data, f, indent=2)
        
    return df

# --- GRADIO UI ---
with gr.Blocks(title="Context Revision Engine (FOG PoC)") as demo:
    gr.Markdown("# Context Revision Engine: Inference-Time Control Mechanism")
    gr.Markdown("This proof-of-concept demonstrates causal intervention on specific attention heads (discovered via FOG descriptive cartography). Modifying these 'revision-sensitive' heads can shift the model's preference toward new contextual facts.")
    
    with gr.Tabs():
        with gr.Tab("Single Run"):
            with gr.Row():
                with gr.Column(scale=1):
                    preset_drop = gr.Dropdown(choices=get_preset_names(), label="Presets", value=get_preset_names()[0])
                    prompt_txt = gr.Textbox(label="Prompt (Ends before answer)", lines=3)
                    old_txt = gr.Textbox(label="Old Fact (String)")
                    new_txt = gr.Textbox(label="New Fact (String)")
                    
                    mode_radio = gr.Radio(["Baseline", "Selective", "Control"], label="Intervention Mode", value="Baseline")
                    scale_slider = gr.Slider(0.0, 2.0, value=1.0, step=0.1, label="Scale Factor (<1 Attenuate, >1 Amplify)")
                    topk_slider = gr.Slider(5, 50, value=10, step=5, label="Top-K for Table")
                    
                    run_btn = gr.Button("Run Inference", variant="primary")
                    
                    preset_drop.change(fn=load_preset, inputs=[preset_drop], outputs=[prompt_txt, old_txt, new_txt])
                    
                with gr.Column(scale=1):
                    ans_txt = gr.Textbox(label="Generated Next Token", lines=1)
                    full_gen_txt = gr.Textbox(label="Full Generated Response", lines=3)
                    interpretation_txt = gr.Textbox(label="Interpretation", lines=3)
                    metrics_table = gr.Dataframe(label="Context Revision Metrics")
                    top10_table = gr.Dataframe(label="Top-10 Next Tokens")
                    
            run_btn.click(
                fn=run_single,
                inputs=[prompt_txt, old_txt, new_txt, mode_radio, scale_slider, topk_slider],
                outputs=[ans_txt, full_gen_txt, metrics_table, top10_table, interpretation_txt]
            )
            
        with gr.Tab("Dose Sweep"):
            gr.Markdown("Run a dose-response sweep across Scale Factors [0.0, 0.5, 1.0, 1.5, 2.0] for both Selective and Control heads.")
            sweep_btn = gr.Button("Run Dose Sweep", variant="primary")
            sweep_plot = gr.Image(label="Dose-Response Curve")
            
            sweep_btn.click(
                fn=run_dose_sweep,
                inputs=[prompt_txt, old_txt, new_txt],
                outputs=[sweep_plot]
            )
            
        with gr.Tab("Batch Eval"):
            gr.Markdown("Run zero-scaling (full attenuation) on all presets to compare Selective vs Control deltas.")
            batch_btn = gr.Button("Run Batch Eval", variant="primary")
            batch_table = gr.Dataframe(label="Batch Summary")
            
            batch_btn.click(
                fn=run_batch_eval,
                inputs=[],
                outputs=[batch_table]
            )

    gr.Markdown("---")
    gr.Markdown("""
    **Scientific Disclaimer:**
    - This is a proof-of-concept for inference-time context revision control.
    - It does not prove full cognitive motifs.
    - Results depend heavily on model architecture, prompt structure, and tokenizer behavior.
    - Selective heads were identified via prior multi-seed descriptive and causal analysis.
    """)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
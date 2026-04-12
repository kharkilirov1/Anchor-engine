# Context Revision Engine: Inference-Time Control Demo

This repository contains a proof-of-concept Gradio application demonstrating inference-time intervention on a pre-trained Large Language Model (`Qwen/Qwen2.5-0.5B`). 

By scaling the output of specific attention heads (identified via FOG descriptive and causal cartography as "revision-sensitive"), we can shift the model's preference toward newly introduced facts and away from persistent old context, effectively improving Context Revision capabilities without fine-tuning.

## Features
- **Single Run:** Interactively test custom or preset prompts under Baseline, Selective, and Control intervention modes. Compare the generated token, the margin between new and old token logits, and their ranks.
- **Dose Sweep:** Generate a dose-response curve visualizing how varying the intervention scale (from 0.0 to 2.0) monotonically affects the context revision margin.
- **Batch Eval:** Run a zero-scale attenuation across all preset scenarios to quantitatively compare the delta in margins for Selective vs. Control heads.

## Setup and Requirements
The demo is designed to run entirely on a local CPU with ~32 GB RAM. 

Install the required dependencies:
```bash
pip install -r requirements.txt
```

## How to Run
Start the Gradio application:
```bash
python app.py
```
Then open your browser to `http://127.0.0.1:7860`.

## Architecture Details
- **Selective Heads:** Layers: `L0H4, L1H4, L19H12, L23H4, L5H11`. These heads were previously mapped as exhibiting high task-selectivity during contextual contradiction evaluation.
- **Control Heads:** Layers: `L2H1, L3H6, L8H3, L11H13`. These heads exhibited low task-dependent variance.
- **Intervention Logic:** The engine registers forward pre-hooks on the `o_proj` linear layers inside the target attention blocks. During inference, it intercepts the concatenated head outputs and scales the specific target heads by the user-defined `scale_factor`.

## Scientific Limitations
- **Proof-of-Concept:** This is a demonstration of inference-time control, not a generalized reasoning engine.
- **Tokenization Constraints:** The margin and rank metrics are strictly calculated on the first generated token of the target strings. Complex multi-token concepts may exhibit different behaviors.
- **Heuristics:** The mapping of specific heads to "compare" or "memory" motifs remains theory-guided interpretation; this demo confirms their selective causal involvement in context revision but does not prove their exact internal mechanism.

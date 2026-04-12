# FOG Hypothesis Validation Plan (Local CPU)

## Overview
This document outlines the experimental plan to validate the **FOG (Finite Operator Grammar) / Function-Morphology Correspondence** hypothesis using micro-models on a local CPU. The goal is to prove that heterogenous cognitive motifs (memory, comparison, routing) require specialized subspace geometries, and that enforcing these geometries yields better performance and parameter efficiency than uniform baseline architectures.

## Experiment 1: Algorithmic Task Convergence (Parameter-Matched)
**Objective:** Prove that FOG (Motif) architecture learns algorithmic tasks faster and more reliably than a uniform baseline of the exact same parameter count.
* **Tasks:** `Copy`, `Reverse`, `Retrieval` (vocabulary $V=32$, sequence length $L=32$).
* **Models:** 
  1. `Baseline` ($d_{model}=128$)
  2. `Uniform Small` (Parameter-matched to FOG)
  3. `FOG Motif` (Heterogeneous geometry: narrow comparison, wide memory)
* **Metrics:** Cross-entropy loss and token accuracy over 50-100 epochs. Average over 5 random seeds.

## Experiment 2: Cross-Layer Angular Drift (CLAD) Measurement
**Objective:** Quantify the separation of information flow in FOG. Memory motifs should exhibit low angular drift (stable representations), while control/comparison motifs should exhibit high drift.
* **Method:** Pass sequences through trained models from Exp 1. Measure cosine similarity between input and output representations of the FFN/MFFN blocks.
* **Expected Result:** FOG models show a bifurcation in CLAD scores corresponding to different geometric pathways, whereas Baseline models show a homogenous, entangled drift.

## Experiment 3: Subspace Geometry and eRank Collapse
**Objective:** Demonstrate that uniform architectures suffer from subspace crowding (capacity starvation) when forced to perform complex routing/memory tasks, while FOG preserves subspace dimensionality.
* **Method:** Calculate the effective rank (eRank) via SVD of the down-projection matrices ($W_2$) periodically during training.
* **Expected Result:** Baseline/Uniform Small models will show a collapse in eRank as they struggle to multiplex tasks in a monolithic space. FOG models will maintain higher eRank in their specialized wide memory projections.

## Experiment 4: Capacity Starvation Boundary
**Objective:** Identify the exact parameter scale where uniform architectures fail due to polysemanticity overload, and show that FOG pushes this boundary lower.
* **Method:** Sweep $d_{model}$ downwards (e.g., 128 $\to$ 64 $\to$ 32 $\to$ 16).
* **Expected Result:** FOG sustains $>95\%$ accuracy on `Retrieval` at a significantly lower $d_{model}$ than the Baseline, proving superior parameter efficiency via morphological specialization.

## Experiment 5: Motif Signatures Extraction (Activation & Attention)
**Objective:** Verify that specific cognitive motifs ($\Phi^{(\mathrm{compare})}$, $\Phi^{(\mathrm{memory})}$, $\Phi^{(\mathrm{select})}$) naturally emerge in trained FOG models and can be identified by their mathematical signatures.
* **Method:** Train a FOG model on algorithmic tasks (`Copy`, `Reverse`, `Retrieval`). Hook into the forward pass to extract Attention weights (measure entropy) and MFFN gate values (measure polarization/sparsity) at each layer.
* **Expected Result:** Different layers will exhibit distinct, specialized behaviors. For example, early layers might show low attention entropy (sharp focus) acting as a "Compare/Read" motif, while deeper layers exhibit high entropy and high gate polarization, acting as "Memory/Select" motifs.

## Experiment 6: Complex Motif Composition Stress Test
**Objective:** Prove that heterogenous subspace allocation allows FOG to chain cognitive motifs (e.g., retrieve $\to$ filter $\to$ compose) without destructive interference, while uniform baselines fail on complex compositional tasks under tight parameter budgets (~400K).
* **Tasks:**
  1. `DistractorRetrieval` (precise comparison)
  2. `NoisyRetrieval` (noise filtering / selection)
  3. `ChainedRetrieval` (two-hop memory lookup)
  4. `SetIntersection` (parallel scan and array output)
* **Method:** Train parameter-matched Uniform and FOG models on these complex tasks.
* **Expected Result:** FOG will successfully converge on compositional tasks (like `ChainedRetrieval`) due to isolated motif subspaces. The Uniform baseline will experience catastrophic interference and stall at near-random performance.

## Experiment 7: Multi-Task Interference (Continual Learning Analog)
**Objective:** Prove that FOG reduces destructive interference when a model is forced to learn fundamentally different cognitive tasks simultaneously within a constrained parameter budget.
* **Tasks:** A single mixed dataset containing `Copy`, `Reverse`, and `Retrieval` tasks multiplexed.
* **Method:** Train Uniform Baseline (~450K) and FOG Motif (~250K) on the mixed dataset. Measure final accuracy on each individual sub-task.
* **Expected Result:** Uniform models will suffer from representation collapse on specific sub-tasks (usually `Retrieval`, which requires broad attention) because the sharp attention required for `Copy` overwrites shared subspaces. FOG will maintain high accuracy across all tasks because its specialized geometric pathways (`d_compare`, `d_memory`) naturally decouple the operations without interference.

## Experiment 8: Scale-Up Validation (10M+ Parameters)
**Objective:** Demonstrate that the morphological advantages (parameter efficiency, computational separation) of FOG are not artifacts of micro-scale models (<1M parameters), but scale robustly to larger parameter regimes and substantially more complex data distributions.
* **Tasks:** Complex Retrieval and Chained Retrieval with a vastly expanded vocabulary ($V=1024$) and longer contexts ($L=128$, up to 32 key-value pairs).
* **Method:** Train a Uniform Baseline and a FOG Motif model (both parameter-matched at ~10M parameters) on CPU. 
* **Expected Result:** At the 10M scale with long context and large vocabulary, the monolithic Uniform architecture will struggle with "attention dilution" and memory bottlenecks. FOG, by expanding its $\Phi^{(\mathrm{memory})}$ subspace and keeping $\Phi^{(\mathrm{compare})}$ sharply constrained, will achieve higher accuracy and faster convergence, proving the hypothesis holds at scale.

## Experiment 9: Causal Motif Intervention (Knockout Analysis)
**Objective:** Move beyond observational signatures (entropy) and prove *causality*. Demonstrate that specific geometric pathways (e.g., specific attention heads or gating layers) are causally responsible for specific cognitive motifs.
* **Tasks:** A multi-task setup combining `Copy` (requires sharp compare) and `Retrieval` (requires broad memory).
* **Method:** Train a FOG model to high accuracy on both tasks. During inference, systematically apply "knockouts" (zeroing out specific attention heads or FFN gates). Measure the differential impact ($\Delta$Accuracy) on the two tasks.
* **Expected Result:** We will observe a double dissociation. Knocking out "sharp" early-layer heads will catastrophically degrade `Copy` while leaving `Retrieval` relatively intact. Knocking out "broad" deep-layer memory heads or polarizing gates will catastrophically degrade `Retrieval` while leaving `Copy` intact. This causally proves morphological specialization.

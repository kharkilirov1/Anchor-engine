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

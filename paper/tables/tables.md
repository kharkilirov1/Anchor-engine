### Table 1: Exp1 Learning Dynamics & eRank Collapse

| Model | Params | Best Acc | Final Acc | Final Mean eRank |
|---|---|---|---|---|
| baseline_large | 801.5K | 0.937 | 0.936 | 115.2 |
| uniform_small | 453.7K | 0.939 | 0.938 | 88.3 |
| fog_motif | 432.1K | 0.940 | 0.936 | 112.2 |

### Table 2: Capacity Starvation Boundary (Exp4)

| $d_{model}$ | Uniform Acc | Uniform Params | FOG Acc | FOG Params | Parameter Ratio (Uniform/FOG) |
|---|---|---|---|---|---|
| 64 | 0.922 | 154.2K | 0.920 | 84.7K | 1.82x |
| 32 | 0.923 | 40.2K | 0.920 | 22.8K | 1.77x |
| 16 | 0.920 | 10.9K | 0.918 | 7.6K | 1.43x |

### Table 3: Enhanced Causal Knockout - Top Specialized Heads (Exp9)

| Model | Specialized For | Head ID | $\Delta$ Copy Acc | $\Delta$ Reverse Acc | $\Delta$ Retrieval Acc | Dissociation Ratio |
|---|---|---|---|---|---|---|
| Uniform | Copy | L1H1 | +0.109 | -0.049 | +0.058 | 1.9x |
| Uniform | Retrieval | L1H2 | +0.026 | +0.005 | +0.097 | 3.7x |
| FOG | Copy | L2H1 | +0.151 | +0.095 | +0.003 | 47.1x |
| FOG | Retrieval | L1H3 | +0.125 | +0.093 | +0.094 | 0.8x |

*Note: Dissociation Ratio calculates the magnitude of targeted accuracy drop versus off-target accuracy drop. Higher indicates cleaner specialization.*

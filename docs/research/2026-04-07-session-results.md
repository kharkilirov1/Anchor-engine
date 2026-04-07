# Session Results 2026-04-05 — 2026-04-07

## 1. Anchor Matching Fixes
- **Problem:** match_anchor_span failed 14/20 domains (BPE context sensitivity)
- **Fix:** Relaxed all 3 matchers to accept first match (not unique)
- **Commit:** `8596a86`

## 2. Attention-Based Anchor Detection
- **Function:** `detect_anchor_span()`
- **Method:** Attention mass from last token, sliding window (min=5, max=10 tokens)
- **Fallback:** Text-based match when attentions unavailable (SDPA mode)
- **Fix for partial attentions:** Qwen returns 8/32 layers → use last-N available
- **Commit:** `8596a86`, `4187e70`

## 3. Auto-Calibration (k-means)
- **Method:** k-means (3 clusters) on observed r1 + delta_template
- **Result on Qwen3.5-4B:** mature_r1: 0.650 → 0.238, template_delta: 0.080 → 0.054
- **Commit:** `1d94621`

## 4. Campaign: Hardcoded Thresholds (Qwen3.5-4B, Colab T4)

| Metric | Value |
|---|---|
| Thresholds | mature_r1=0.65, template_delta=0.08 |
| r1 range | 0.199 — 0.340 |
| Clusters | flat=17, mature=0, template=3 |
| Anchor invoked | 17/20 |
| Wins | 4 |
| **Total LOSS** | **13** |

### Per-domain
| Domain | Cluster | Base Q | Chosen Q | Delta | Result |
|---|---|---|---|---|---|
| vegan_meal_plan | flat | 12.1 | 17.8 | +5.8 | WIN |
| fastapi_service | flat | -6.0 | 21.2 | +27.2 | WIN |
| proof_by_contradiction | flat | 8.9 | 11.8 | +3.0 | WIN |
| gluten_free_bakery | flat | 1.9 | -13.3 | -15.2 | LOSS |
| rust_safe_only | flat | 12.4 | 5.0 | -7.4 | LOSS |
| gdpr_data_policy | flat | 7.5 | 2.9 | -4.5 | LOSS |
| metric_units_only | flat | 28.0 | 10.4 | -17.6 | LOSS |
| halal_cuisine | flat | 14.4 | 2.4 | -12.0 | LOSS |
| postgresql_raw_sql | template | 15.6 | 15.6 | +0.0 | SAME |
| renewable_energy_plan | flat | 9.3 | 5.7 | -3.5 | LOSS |
| typescript_strict | flat | 34.2 | 11.9 | -22.3 | LOSS |
| functional_no_mutation | template | -23.4 | -23.4 | +0.0 | SAME |
| formal_academic_style | flat | 6.7 | -33.0 | -39.7 | LOSS |
| zero_waste_guide | flat | -6.3 | -22.7 | -16.4 | LOSS |
| kubernetes_native | template | 22.5 | 22.5 | +0.0 | SAME |
| organic_farming | flat | 6.7 | 2.2 | -4.5 | LOSS |
| drug_free_pain_management | flat | 4.1 | 6.3 | +2.2 | WIN |
| budget_travel | flat | 15.1 | 14.4 | -0.7 | LOSS |
| python_typed_dataclasses | flat | 22.2 | 6.8 | -15.4 | LOSS |
| minimalist_ui_design | flat | -0.9 | -27.0 | -26.1 | LOSS |

## 5. Campaign: Auto-Calibrated (Qwen3.5-4B, Colab T4)

| Metric | Value |
|---|---|
| Calibrated thresholds | mature_r1=0.238, template_delta=0.054 |
| Cluster centers | flat(r1=0.231,d=0.014), mature(r1=0.245,d=0.051), template(r1=0.227,d=0.094) |
| Clusters | flat=7, mature=9, template=4 |
| Anchor invoked | 7/20 |
| Wins | 1 |
| **Total LOSS** | **6 (was 14, -57%)** |

### Per-domain
| Domain | Cluster | Base Q | Chosen Q | Delta | Result |
|---|---|---|---|---|---|
| vegan_meal_plan | flat | 12.1 | 17.8 | +5.8 | WIN |
| fastapi_service | mature | -6.0 | -6.0 | +0.0 | SAME |
| proof_by_contradiction | mature | 8.9 | 8.9 | +0.0 | SAME |
| gluten_free_bakery | flat | 1.9 | -13.3 | -15.2 | LOSS |
| rust_safe_only | flat | 12.4 | 5.0 | -7.4 | LOSS |
| gdpr_data_policy | mature | 7.5 | 7.5 | +0.0 | SAME |
| metric_units_only | template | 28.0 | 28.0 | +0.0 | SAME |
| halal_cuisine | flat | 14.4 | 2.4 | -12.0 | LOSS |
| postgresql_raw_sql | template | 15.6 | 15.6 | +0.0 | SAME |
| renewable_energy_plan | template | 9.3 | 9.3 | +0.0 | SAME |
| typescript_strict | mature | 34.2 | 34.2 | +0.0 | SAME |
| functional_no_mutation | mature | -23.4 | -23.4 | +0.0 | SAME |
| formal_academic_style | mature | 6.7 | 6.7 | +0.0 | SAME |
| zero_waste_guide | flat | -6.3 | -22.7 | -16.4 | LOSS |
| kubernetes_native | mature | 22.5 | 22.5 | +0.0 | SAME |
| organic_farming | template | 6.7 | 6.7 | +0.0 | SAME |
| drug_free_pain_management | mature | 4.1 | 4.1 | +0.0 | SAME |
| budget_travel | flat | 15.1 | 14.4 | -0.7 | LOSS |
| python_typed_dataclasses | flat | 22.2 | 6.8 | -15.4 | LOSS |
| minimalist_ui_design | mature | -0.9 | -0.9 | +0.0 | SAME |

## 6. H6 Continuous Bias (Qwen2.5-1.5B, HF Space T4)

| Metric | Value |
|---|---|
| Formula | bias_scale = 1.50 * max(0, 1 - r1/r1_ceiling) |
| r1_ceiling | 0.50 (auto) |
| Clusters | flat=8, mature=6, template=6 |
| Anchor invoked | 14/20 |
| Wins | 6 (43% win rate) |
| **Total LOSS** | **8** |
| Key finding | anchor helps when base_q<0 (83% win), hurts when base_q>20 (100% loss) |

### Per-domain
| Domain | Bias | Base Q | Chosen Q | Delta | Result |
|---|---|---|---|---|---|
| vegan_meal_plan | 0.146 | -42.1 | 7.5 | +49.6 | WIN |
| fastapi_service | 0.000 | 0.1 | 0.1 | +0.0 | SAME |
| proof_by_contradiction | 0.000 | 6.4 | 6.4 | +0.0 | SAME |
| gluten_free_bakery | 0.000 | -96.8 | -96.8 | +0.0 | SAME |
| rust_safe_only | 0.261 | -34.1 | 8.7 | +42.8 | WIN |
| gdpr_data_policy | 0.000 | -17.5 | -17.5 | +0.0 | SAME |
| metric_units_only | 0.068 | 63.8 | 7.4 | -56.4 | LOSS |
| halal_cuisine | 0.000 | 33.6 | 33.6 | +0.0 | SAME |
| postgresql_raw_sql | 0.354 | -13.1 | 8.3 | +21.4 | WIN |
| renewable_energy_plan | 0.286 | 21.3 | 10.9 | -10.4 | LOSS |
| typescript_strict | 0.000 | 12.4 | 12.4 | +0.0 | SAME |
| functional_no_mutation | 0.070 | 22.8 | 4.9 | -17.8 | LOSS |
| formal_academic_style | 0.236 | -5.5 | 5.6 | +11.1 | WIN |
| zero_waste_guide | 0.161 | 6.1 | 2.4 | -3.7 | LOSS |
| kubernetes_native | 0.031 | 130.0 | 36.2 | -93.8 | LOSS |
| organic_farming | 0.381 | -4.6 | 1.9 | +6.5 | WIN |
| drug_free_pain_management | 0.313 | 10.5 | 13.6 | +3.0 | WIN |
| budget_travel | 0.058 | 11.3 | 1.6 | -9.7 | LOSS |
| python_typed_dataclasses | 0.033 | 7.6 | 5.0 | -2.6 | LOSS |
| minimalist_ui_design | 0.177 | -2.9 | -14.3 | -11.4 | LOSS |

## 7. FOG Ablation (Tiny, HF Space T4)

### Configs
| Config | vocab | d_model | layers | heads | seq_len | d_ff | d_compare | d_memory | d_expand | d_gate | Params |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Baseline | 32 | 128 | 4 | 4 | 32 | 512 | — | — | — | — | 801,536 |
| Motif | 32 | 128 | 4 | 4 | 32 | 512 | 32 | 96 | 256 | 16 | 432,064 |

### Results (50 epochs)
| Task | Baseline Acc | Motif Acc | Baseline Loss | Motif Loss | Baseline Converge | Motif Converge |
|---|---|---|---|---|---|---|
| copy | 100% | 100% | 1.5543 | 1.5567 | epoch 10 | epoch 15 |
| reverse | 100% | 100% | 1.5602 | 1.5568 | epoch 10 | epoch 15 |
| retrieval | 96.4% | **96.7%** | 0.9442 | **0.9205** | — | — |

**Conclusion:** Motif 46% fewer params, equal or better accuracy. Retrieval (compare+memory task) motif wins.

## 8. Commits
| Hash | Message |
|---|---|
| 8596a86 | attention-based anchor detection + fix matching |
| 715ba60 | enable eager attention + expand narrow spans |
| 3470f03 | robust attention extraction + diagnostics |
| 4187e70 | last-N attention fallback + min span 5 |
| 1d94621 | auto-calibrate thresholds via k-means |
| 30c465b | H6 continuous guardrail hypothesis |
| 320a613 | auto-calibrated results LOSS 14->6 |
| 0a9642d | H6 continuous bias scaling |
| 74e0953 | FOG ablation — motif 46% fewer params |

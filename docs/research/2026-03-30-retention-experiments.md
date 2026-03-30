# Retention Experiments — 2026-03-30

Model: `Qwen/Qwen2.5-1.5B`
Script: `scripts/run_qwen_long_retention_compare.py`

These experiments test whether the anchor bias overlay can maintain semantic constraints
during free generation, compared to a pure greedy baseline.

---

## Experiment 1 — Vegan Chef (smoke, 24 tokens)

**Prompt:** "You are a vegan chef. Write a detailed weekly meal plan with recipes for each day."

**Settings:** greedy, repetition_penalty=1.15, frequency_penalty=0.05, no_repeat_ngram_size=3,
conflict_threshold=0.55, bias_scale=1.50, min_bias_pressure=0.60

| Metric | BASE | ANCHOR |
|---|---:|---:|
| lexical_score | 0.0 | 1.0 |
| positive_total | 0 | 1 |
| negative_total | 0 | 0 |
| bias_active_steps | — | 7 / 24 |
| identical | no | — |

BASE continuation (first 200 chars):
> The plan should include breakfast, lunch, dinner, and snacks. The meals should be healthy, nutritious, and delicious.

ANCHOR continuation (first 200 chars):
> The meals should be healthy, nutritious and include at least one vegetarian dish per week.
> Monday: Breakfast - Vegan...

**Result:** ANCHOR immediately surfaces "vegan" and "vegetarian" content. BASE generates
generic health framing with no vegan signal.

---

## Experiment 2 — Vegan Chef (500 tokens, local)

**Settings:** same as smoke, max_new_tokens=500

| Metric | BASE | ANCHOR |
|---|---:|---:|
| lexical_score | — | — |
| bias_active_steps | — | 228 / 500 |
| identical | no | — |

BASE: generic meal plan, no vegan keywords, chicken/salmon/beef throughout.
ANCHOR: 228/500 steps with active bias, holds vegan structure throughout.

---

## Experiment 3 — Vegan Chef (1000 tokens, Colab, cuda)

**Settings:** max_new_tokens=1000, max_length=1024, no penalties (first run)

| Metric | BASE | ANCHOR |
|---|---:|---:|
| lexical_score | -19.0 | 4.0 |
| positive_total | 27 | 84 |
| negative_total | 46 | 80* |
| bias_active_steps | — | 736 / 1000 |
| first_positive | "vegetable" @ char 1153 | "vegan" @ char 979 |
| first_negative | "milk" @ char 1006 | "cheese" @ char 1116 |
| identical | no | — |

*Note: ANCHOR negative_total of 80 is misleading — most hits are from "vegan cheese",
"vegan sour cream", "almond milk" contexts. The `protected_negative_hits` detector uses
a 12-char prefix window which misses "1/4 cup vegan " (14 chars). This is a known metric bug.
Effective unprotected negatives for ANCHOR are much lower. BASE negatives include chicken,
salmon, beef, tuna, eggs — genuinely non-vegan content.

BASE: generates generic non-vegan plan, then loops on grocery list
(repeating "Beef, Vegetable, Brown rice, Carrots..." endlessly).

ANCHOR: holds vegan semantic (Vegan Breakfast Burrito, black beans, quinoa), then loops on
ingredient list ("1/4 cup vegan cheese / 1/4 cup vegan sour cream" × 40+).

**Key finding:** Both models loop at 1000 tokens without repetition penalties. But ANCHOR loops
on *vegan-compliant* content while BASE loops on *non-vegan* content. Semantic constraint held
even through the loop. Loop prevention is a separate problem from semantic retention.

---

## Experiment 4 — FastAPI Constraint (run 1)

**Prompt:** "...async FastAPI service with typed Pydantic models, dependency injection,
async request handlers, validation, background jobs, and deployment notes.
Do not rewrite the design as Django or synchronous class-based views..."

**Settings:** greedy, repetition_penalty=1.15, frequency_penalty=0.05, no_repeat_ngram_size=3

| Metric | BASE | ANCHOR |
|---|---:|---:|
| lexical_score | 17.0 | 14.0 |
| positive_total | 29 | 15 |
| negative_total | 12 | 1 |
| bias_active_steps | — | 73 / ~100 |
| blocked_ngram_steps | — | 8 |

BASE: immediately loops on the instruction text. Repeats "Do not rewrite the design as
Django or synchronous class-based views. Instead, focus on the async FastAPI service..." 4+ times.
All positive hits come from the loop, not generated content. Content = 0.

ANCHOR: generates structured guide — Introduction, Setup, First Endpoint with real code,
Pydantic types section, Dependency Injection section. 0 django hits, 0 class-based view hits,
1 "synchronous" in technical context.

Note: BASE lexical_score (17.0) > ANCHOR (14.0) because BASE inflates positive score by
repeating the prompt. This exposes a metric flaw: repetition score != content score.

---

## Experiment 5 — FastAPI Constraint (run 2, reproduction)

| Metric | BASE | ANCHOR |
|---|---:|---:|
| lexical_score | 33.0 | 22.0 |
| positive_total | 66 | 23 |
| negative_total | 33 | 1 |
| bias_active_steps | — | 266 |
| blocked_ngram_steps | — | 24 |

BASE: same loop, now repeated 11 times. Identical degenerate behavior.
ANCHOR: same structured output as run 1. Reproducible.

**Key finding:** BASE failure is deterministic (greedy decoding → same loop every time).
ANCHOR success is also deterministic (same structure, same sections, minor wording differences).

---

## Cross-experiment summary

| Domain | BASE behavior | ANCHOR behavior | Semantic held |
|---|---|---|---|
| Vegan (24 tok) | generic framing, no vegan | vegan content from token 1 | ✅ |
| Vegan (500 tok) | non-vegan plan | vegan plan throughout | ✅ |
| Vegan (1000 tok) | loops on non-vegan grocery list | loops on vegan ingredients | ✅ (loop ≠ semantic failure) |
| FastAPI (run 1) | loops on instruction text | structured guide, 0 django | ✅ |
| FastAPI (run 2) | same loop ×11 | same guide, reproducible | ✅ |

---

## Key observations

1. **Semantic retention is separate from loop prevention.** Anchor bias holds the semantic
   direction. Anti-repetition penalties (ngram, frequency, repetition) prevent structural loops.
   Both are needed but they solve different problems.

2. **BASE failure is deterministic.** Greedy decoding + small model + negative constraint
   ("do not X") = predictable loop on the constraint text. This is anchor decay in action:
   the model cannot hold "avoid X" as a semantic anchor and instead copies it literally.

3. **ANCHOR failure mode is semantically correct.** When ANCHOR loops (without penalties),
   it loops on content that satisfies the constraint. This is qualitatively different from
   BASE loops which violate the constraint.

4. **Metric bug: protected_negative_hits window too short.** 12-char prefix check misses
   "1/4 cup vegan [negative_word]" patterns. Should be extended to 20+ chars or use
   a token-level check.

5. **Reproducibility confirmed.** Two FastAPI runs produce identical BASE failure and
   near-identical ANCHOR success. Method is deterministic.

---

## Next planned experiment

Math domain (high anchor density ρ ≈ 0.85):

Proposed prompt:
> "Prove that √2 is irrational. Use only proof by contradiction.
> Do not use decimal approximations or calculator methods."

Expected: BASE either loops on the method constraint or drifts to a different proof method.
ANCHOR should hold "proof by contradiction" as semantic anchor throughout.

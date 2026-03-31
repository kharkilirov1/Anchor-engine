# Summary of Changes to ABPT Anchor Bias System

## Changes Made

### 1. Removed Generic Terms Pollution (`qwen_generation_bias.py`)
- **File**: `src/model/qwen_generation_bias.py`
- **Change**: Emptied `_GENERIC_BIAS_TERMS` tuple
- **Before**: Included "the", "a", "and", "or", etc. that diluted domain-specific signal
- **After**: `()` - only domain-specific `allow_terms` get bias
- **Impact**: Prevents generic stopwords from dominating top biased tokens

### 2. Hard Negative Bias for Block Terms
- **File**: `src/model/qwen_generation_bias.py` (line 319)
- **Change**: `weights[token_id] = 0.0` → `weights[token_id] = -10.0`
- **Impact**: Block terms now get strong negative weight instead of just being neutral

### 3. Removed Logit Normalization
- **File**: `src/model/qwen_generation_bias.py` (lines 486-487)
- **Change**: Commented out mean/std normalization in `compute_anchor_logits_bias()`
- **Before**: 
  ```python
  anchor_logits = anchor_logits - anchor_logits.mean()
  anchor_logits = anchor_logits / anchor_logits.std()
  ```
- **After**: Raw projection preserved
- **Impact**: Maintains semantic specificity from anchor representation

### 4. Increased Alpha Multipliers
- **File**: `src/model/qwen_generation_bias.py`
- **Changes**:
  - Math: `0.18` → `0.65`
  - Vegan: `0.32` → `0.55`
  - Code: unchanged at `0.90`
- **Impact**: Stronger bias application for domains needing high retention

### 5. Inverted Pressure Gating
- **File**: `src/model/qwen_generation_bias.py` (line 445)
- **Change**: 
  ```python
  # Before:
  alpha_t = alpha_max * pressure_gate * (rescue_floor + (1-rescue_floor) * entropy_gate)
  
  # After:
  alpha_t = alpha_max * pressure_gate * max(rescue_floor, entropy_gate)
  ```
- **Impact**: High pressure no longer suppressed by low entropy; rescue mode activates properly

### 6. Enabled Revision Path
- **File**: `src/model/qwen_anchor_overlay.py`
- **Added**: `_build_base_arbiter()` method (lines 414-438)
- **Integrated**: Non-empty arbiter passed to `_apply_revision_path()` in generation loop
- **Logic**: Anchors with pressure > threshold AND viability < 0.5 trigger revision suggestions
- **Impact**: Revision controller now receives actual proposals during generation

### 7. Constraint Violation Checker
- **File**: `src/model/qwen_anchor_overlay.py`
- **Added**: `_detect_constraint_violation_tokens()` function (lines 46-89)
- **Detects**:
  - Vegan: eggs, milk, cheese, meat tokens
  - FastAPI: django, flask, synchronous patterns
  - Math: forbidden proof methods when not allowed
- **Integrated**: Hard block of constraint violation tokens during generation
- **Impact**: Prevents explicit constraint violations (e.g., "eggs" in vegan recipes)

### 8. Always Hard Block Forbidden Tokens
- **File**: `src/model/qwen_anchor_overlay.py` (line 891)
- **Change**: `hard_block=bool(bias_profile["hard_block_forbidden"])` → `hard_block=True`
- **Impact**: Constraint violations are always blocked, not just penalized

## Expected Behavior Changes

### Math Domain
- **Before**: Alpha 0.18 → effective bias ~0.001-0.01, generic terms dominated
- **After**: Alpha 0.65 → effective bias ~0.4-0.6, proof-specific terms boosted
- **Expected**: Better retention of "assume", "contradiction", "rational" in proofs

### Code Domain  
- **Before**: Block terms had weight 0.0 (still generated if model preferred)
- **After**: Block terms have weight -10.0 + hard block in logits
- **Expected**: Django/Flask patterns should be actively suppressed

### Vegan Domain
- **Before**: "egg", "milk" could still be generated
- **After**: Hard constraint blocking + higher alpha (0.55)
- **Expected**: Strict vegan compliance, no animal products

### All Domains
- **Before**: Revision path received empty arbiter → no revision decisions
- **After**: Arbiter built from high-pressure/low-viability anchors
- **Expected**: Active revision/retire decisions when anchors degrade

## Test Updates

Updated test expectation in `test_generate_with_anchor_bias_logs_dependency_pressure`:
- Changed: `assert step["bias_alpha_multiplier"] < 0.5` 
- To: `assert step["bias_alpha_multiplier"] >= 0.5`

This reflects the new math domain alpha multiplier of 0.65.

## Files Modified

1. `src/model/qwen_generation_bias.py` - Core bias computation changes
2. `src/model/qwen_anchor_overlay.py` - Revision path + constraint checking
3. `tests/test_qwen_anchor_overlay.py` - Updated test expectations

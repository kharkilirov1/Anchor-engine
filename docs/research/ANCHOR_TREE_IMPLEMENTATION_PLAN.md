# Anchor Tree Implementation Plan

Date: 2026-03-30
Status: active implementation plan

## 1. Goal

Move the Qwen anchor pipeline from local anchor control toward trajectory-consistency control.

The new layer should model not only which anchors are active, but which consequence trees those anchors are growing and whether those trees remain globally compatible with the expected task trajectory.

## 2. Core claim

Local anchor scores are not enough.

We need to track:
- expected descendants of a dominant task anchor;
- observed descendants produced by future-influence hints, auxiliary proposals, and later generated spans;
- compatibility between whole trees, not only between individual anchors.

In short:
- anchors say what matters locally;
- tree matching says whether the reasoning trajectory is still globally valid.

## 3. Scope

This plan targets the Qwen overlay research path first.

It does not replace the older ABPT modules immediately. It creates a new structural layer that can later feed:
- proposal ranking;
- revise/keep decisions;
- branch-aware inference.

## 4. Definitions

### 4.1 Anchor tree node

Each node represents a semantic unit in a local reasoning tree.

Fields:
- `node_id`
- `label`
- `text`
- `depth`
- `role`
- `source`
- optional span indices
- optional representation tensor
- score / required / drift metadata

### 4.2 Anchor tree

A tree is a root anchor plus derived descendants and parent-child edges.

Formal view:
- `T(a) = (V_a, E_a)`

### 4.3 Tree graph

The global structural layer is a graph over trees and their relations.

Relations include:
- `supports`
- `requires`
- `compatible`
- `contradicts`
- `expected_next`
- `alternative_to`

## 5. Matching targets

### 5.1 Observed tree -> expected tree

Main structural check.

Use this to score:
- required-step coverage;
- missing expected descendants;
- spurious drift branches;
- local order violations.

### 5.2 Proposal tree -> expected tree

Main repair check.

A proposal is useful if it restores expected descendants, reduces drift, or removes a conflicting subtree.

### 5.3 Tree -> tree between active branches

This captures cross-branch conflict.

A locally plausible branch can still be globally invalid if its subtree conflicts with another active subtree.

## 6. MVP domains

### 6.1 math_ibp

Root:
- `integration_by_parts_only`

Expected descendants:
1. `select_u_and_dv`
2. `derive_du_and_v`
3. `substitute_uv_minus_int_vdu`
4. `reduce_integral_complexity`
5. `repeat_if_needed`
6. `simplify_result`
7. `integration_constant`

Typical drift labels:
- `shortcut_lookup`
- `table_reference`
- `meta_abort`
- `substitution_switch`
- `wrong_symbolic_step`

### 6.2 code_fastapi

Root:
- `async_fastapi_service`

Expected descendants:
1. `typed_request_models`
2. `dependency_injection`
3. `async_handlers`
4. `validation_path`
5. `background_jobs`
6. `deployment_notes`

Typical drift labels:
- `django_view_reframe`
- `synchronous_handler_reframe`
- `template_rendering_branch`

## 7. Scoring

### 7.1 Node match score

For node pair `(n_obs, n_exp)`:

- label compatibility
- role compatibility
- depth compatibility
- text overlap
- optional representation similarity

### 7.2 Tree alignment

Tree alignment should reward:
- matched required nodes;
- structural agreement.

It should penalize:
- missing required nodes;
- spurious observed nodes;
- order violations.

### 7.3 Coverage

Coverage is the fraction of required expected nodes that were matched.

### 7.4 Drift

Drift increases with:
- missing required mass;
- off-path drift nodes;
- ordering failures.

### 7.5 Repair score

Proposal utility should be measured by delta against the current tree:
- coverage gain;
- alignment gain;
- spurious/conflict reduction.

## 8. File plan

### Core types
- `C:\Users\Kharki\Desktop\ABPT\src\model\anchor_tree_types.py`
- `C:\Users\Kharki\Desktop\ABPT\src\model\anchor_tree.py`

### Domain layer
- `C:\Users\Kharki\Desktop\ABPT\src\model\anchor_tree_templates.py`
- `C:\Users\Kharki\Desktop\ABPT\src\model\anchor_tree_domain.py`

### Matching / consistency layer
- `C:\Users\Kharki\Desktop\ABPT\src\model\anchor_tree_match.py`
- later: `anchor_tree_consistency.py`
- later: `anchor_tree_proposals.py`
- later: `anchor_tree_builder.py`

### Tests
- `C:\Users\Kharki\Desktop\ABPT\tests\test_anchor_tree_types.py`
- `C:\Users\Kharki\Desktop\ABPT\tests\test_anchor_tree_templates.py`
- `C:\Users\Kharki\Desktop\ABPT\tests\test_anchor_tree_domain.py`
- `C:\Users\Kharki\Desktop\ABPT\tests\test_anchor_tree_match.py`

## 9. Implementation milestones

### Milestone 1

Data model + expected templates + domain detection + matching core.

Deliverables:
- typed tree structures;
- math/code expected templates;
- domain detector;
- greedy observed->expected matching;
- unit tests.

### Milestone 2

Observed tree builder from:
- active anchors;
- future hint spans;
- auxiliary proposals.

### Milestone 3

Consistency diagnostics:
- drift score;
- cross-tree conflict;
- graph-level consistency.

### Milestone 4

Proposal repair ranking:
- inject proposal candidate;
- recompute tree score;
- rank by repair gain.

### Milestone 5

Integrate tree diagnostics into `QwenAnchorOverlay` reports.

### Milestone 6

Prototype tree-guided branch selection during generation.

## 10. Sprint 1 acceptance criteria

Sprint 1 is successful if:
- expected trees serialize and validate;
- domain detector routes math/code examples correctly;
- healthy trees score above drifted trees;
- spurious drift branches lower alignment;
- tests pass on CPU.

## 11. Out of scope for Sprint 1

Do not build yet:
- learned graph neural controllers;
- full subtree-to-subtree conflict engine;
- decoding-time branch selection;
- end-to-end training on tree labels.

Sprint 1 should stay rule-based, typed, and testable.

## 12. Immediate next action

After fixing this plan in-repo, implement Sprint 1 before touching generation again.


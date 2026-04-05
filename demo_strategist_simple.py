"""
Demo: Local Strategist (Kimi orchestrator) - what theories does it propose?
"""
import os
os.environ['USE_LOCAL_STRATEGIST'] = '1'

from scripts.local_strategist import strategist_local_select
import json


def show_proposal(name, state):
    """Show what strategist proposes for this state."""
    print(f"\n{'='*60}")
    print(f"SCENARIO: {name}")
    print(f"{'='*60}")
    print(f"Phase: {state['current_phase']}, Budget: {state['budget_remaining']}")
    print(f"Completed experiments: {len(state['phases']['1']['experiments'])}")
    
    result = strategist_local_select(state, "")
    
    if result:
        print(f"\n>> STRATEGIST PROPOSES:")
        print(f"   Experiment ID: {result['id']}")
        print(f"   Description:   {result['description']}")
        print(f"   Script:        {result['script']}")
        print(f"   Args:          {result.get('args', {})}")
        print(f"   Metric:        {result.get('result_key', 'N/A')}")
        print(f"   Threshold:     {result.get('success_threshold', 'N/A')}")
        print(f"   Reasoning:     {result.get('reasoning', 'N/A')[:80]}...")
    return result


print("="*60)
print("LOCAL STRATEGIST (KIMI ORCHESTRATOR) - THEORY PROPOSALS")
print("="*60)

# Scenario 1: Clean slate
state1 = {
    'budget_remaining': 10,
    'current_phase': 1,
    'phases': {'1': {'experiments': []}},
    'known_facts': {'model': 'Qwen/Qwen2.5-1.5B'},
    'open_hypotheses': []
}
show_proposal("Start of research (clean slate)", state1)

# Scenario 2: H1 confirmed
state2 = {
    'budget_remaining': 9,
    'current_phase': 1,
    'phases': {'1': {'experiments': [{'hypothesis_id': 'H1', 'status': 'success', 'metric_value': 0.642}]}},
    'known_facts': {'model': 'Qwen/Qwen2.5-1.5B', 'tail_retention_ratio_rho_medium': 0.642},
    'open_hypotheses': [{'id': 'H1_short', 'status': 'untested'}, {'id': 'H1_long', 'status': 'untested'}]
}
show_proposal("H1 confirmed (rho=0.642), need short/long validation", state2)

# Scenario 3: H1 + H1_short done
state3 = {
    'budget_remaining': 8,
    'current_phase': 1,
    'phases': {'1': {'experiments': [
        {'hypothesis_id': 'H1', 'status': 'success', 'metric_value': 0.642},
        {'hypothesis_id': 'H1_short', 'status': 'success', 'metric_value': 0.58}
    ]}},
    'known_facts': {'model': 'Qwen/Qwen2.5-1.5B'},
    'open_hypotheses': [{'id': 'H1_long', 'status': 'untested'}, {'id': 'H2', 'status': 'untested'}]
}
show_proposal("H1 + H1_short done, remaining H1_long", state3)

# Scenario 4: Phase 1 complete
state4 = {
    'budget_remaining': 6,
    'current_phase': 2,
    'phases': {
        '1': {'experiments': [
            {'hypothesis_id': 'H1', 'status': 'success'},
            {'hypothesis_id': 'H1_short', 'status': 'success'},
            {'hypothesis_id': 'H1_long', 'status': 'success'}
        ]},
        '2': {'experiments': []}
    },
    'known_facts': {'model': 'Qwen/Qwen2.5-1.5B', 'phase_1_complete': True},
    'open_hypotheses': [{'id': 'H2', 'status': 'untested'}, {'id': 'H3', 'status': 'untested'}]
}
show_proposal("Phase 1 complete, transition to phase 2", state4)

print("\n" + "="*60)
print("SUMMARY: THEORY PROPOSAL SEQUENCE")
print("="*60)
print("""
The local strategist proposes experiments in this order:

1. H1_short  -> Validate tail_retention_ratio on SHORT profile
2. H1_long   -> Validate tail_retention_ratio on LONG profile  
3. H2        -> Measure group-specific carryover vs universal threshold

LOGIC:
- First validates confirmed finding (tail_retention_ratio) on different profiles
- Then moves to phase 2 experiments (group routing)
- Uses heuristic based on completed experiments list

FOR REAL AI (KIMI) MODE:
- Prompt is displayed with full context
- You can review and modify the proposal
- Save your decision to /tmp/ABPT_strategist_response.json
- Or use LOCAL_STRATEGIST_MODE=auto for automatic Ollama
""")

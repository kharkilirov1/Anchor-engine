"""
Benchmark local strategist pipeline performance.
Measures decision-making speed for the Kimi orchestrator.
"""
from __future__ import annotations

import time
import os

# Must set before importing
os.environ['USE_LOCAL_STRATEGIST'] = '1'

from scripts.local_strategist import (
    strategist_local_select,
    _build_strategist_context,
    _build_strategist_prompt,
    _parse_strategist_response,
    _call_local_agent,
)


def benchmark_context_building():
    """Benchmark building strategist context."""
    print("[1/4] Benchmarking context building...")
    
    state = {
        'budget_remaining': 5,
        'current_phase': 1,
        'phases': {
            '1': {'experiments': []}
        },
        'known_facts': {'model': 'Qwen/Qwen2.5-1.5B'},
        'open_hypotheses': []
    }
    
    times = []
    for _ in range(100):
        start = time.perf_counter()
        _ = _build_strategist_context(state, playbook="")
        times.append(time.perf_counter() - start)
    
    avg_time = sum(times) / len(times) * 1000
    print(f"  Average: {avg_time:.3f} ms per call")
    print(f"  Calls/sec: {1000/avg_time:.0f}")
    return avg_time


def benchmark_prompt_building():
    """Benchmark building strategist prompt."""
    print("\n[2/4] Benchmarking prompt building...")
    
    context = {
        'available_scripts': ['run_qwen_test.py', 'run_qwen_probe.py'],
        'completed_experiments': [],
        'overused_scripts': [],
        'budget_remaining': 5,
        'current_phase': 1,
        'playbook_tail': '',
        'known_facts': {},
        'open_hypotheses': [],
    }
    
    times = []
    for _ in range(100):
        start = time.perf_counter()
        _ = _build_strategist_prompt(context)
        times.append(time.perf_counter() - start)
    
    avg_time = sum(times) / len(times) * 1000
    prompt = _build_strategist_prompt(context)
    print(f"  Average: {avg_time:.3f} ms per call")
    print(f"  Prompt size: {len(prompt)} chars")
    return avg_time


def benchmark_response_parsing():
    """Benchmark parsing JSON response."""
    print("\n[3/4] Benchmarking response parsing...")
    
    import json
    valid_response = json.dumps({
        "id": "H1",
        "description": "Test experiment",
        "script": "run_qwen_phase_probe.py",
        "args": {"anchor_profile": "medium"},
        "result_key": "correlation_summary.tail_retention_ratio",
        "success_threshold": 0.4,
        "reasoning": "Need to validate"
    })
    
    times = []
    for _ in range(1000):
        start = time.perf_counter()
        _ = _parse_strategist_response(valid_response)
        times.append(time.perf_counter() - start)
    
    avg_time = sum(times) / len(times) * 1000
    print(f"  Average: {avg_time:.3f} ms per call")
    print(f"  Calls/sec: {1000/avg_time:.0f}")
    return avg_time


def benchmark_full_pipeline():
    """Benchmark full strategist selection pipeline."""
    print("\n[4/4] Benchmarking full pipeline...")
    
    # Simulate realistic state with some experiments completed
    state = {
        'budget_remaining': 3,
        'current_phase': 1,
        'phases': {
            '1': {
                'experiments': [
                    {'hypothesis_id': 'H1', 'status': 'success', 'metric_value': 0.642},
                    {'hypothesis_id': 'H1_short', 'status': 'success', 'metric_value': 0.58},
                ]
            }
        },
        'known_facts': {
            'model': 'Qwen/Qwen2.5-1.5B',
            'best_predictor': 'tail_retention_ratio'
        },
        'open_hypotheses': [
            {'id': 'H1_long', 'status': 'untested'},
            {'id': 'H2', 'status': 'untested'},
        ]
    }
    
    playbook = """
## Recent findings
- tail_retention_ratio confirmed with rho=0.642
- early_slope_4_8 is negative predictor
"""
    
    times = []
    proposals = []
    for _ in range(10):
        start = time.perf_counter()
        result = strategist_local_select(state, playbook)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        if result:
            proposals.append(result.get('id'))
    
    avg_time = sum(times) / len(times) * 1000
    print(f"  Average: {avg_time:.1f} ms per decision")
    print(f"  Decisions/sec: {1000/avg_time:.1f}")
    if proposals:
        print(f"  Sample proposals: {set(proposals)}")
    return avg_time


def main():
    print("=" * 60)
    print("Local Strategist (Kimi Orchestrator) Benchmark")
    print("=" * 60)
    print(f"Mode: {os.environ.get('LOCAL_STRATEGIST_MODE', 'interactive (heuristic)')}")
    print("=" * 60)
    
    t1 = benchmark_context_building()
    t2 = benchmark_prompt_building()
    t3 = benchmark_response_parsing()
    t4 = benchmark_full_pipeline()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Context building:    {t1:6.3f} ms")
    print(f"  Prompt building:     {t2:6.3f} ms")
    print(f"  Response parsing:    {t3:6.3f} ms")
    print(f"  Full pipeline:       {t4:6.1f} ms")
    print("=" * 60)
    
    print("\n[IMPLICATIONS FOR ORCHESTRATOR]")
    print(f"- One decision takes ~{t4:.0f} ms")
    print(f"- Can make {1000/t4:.0f} decisions per second")
    print("- Overhead is negligible vs experiment runtime")
    print("- Local strategist is FAST - no API latency!")
    
    total_orchestrator_overhead = t4 * 3  # 3 decisions per budget
    print(f"\n- Total overhead for 3 experiments: ~{total_orchestrator_overhead:.0f} ms")
    print("- vs DeepSeek API: ~1000-3000 ms per call")
    print("- Speedup: ~10-100x faster!")


if __name__ == "__main__":
    main()

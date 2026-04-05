"""
Lightweight test for local strategist (Kimi orchestrator) pipeline.
No heavy computation, no API calls.
"""
from __future__ import annotations

import json
import sys
import time


def test_local_strategist_import():
    """Test 1: Import local_strategist module."""
    print("[1/5] Importing local_strategist...", end=" ")
    try:
        from scripts.local_strategist import (
            strategist_local_select,
            _build_strategist_context,
            _build_strategist_prompt,
            _parse_strategist_response,
        )
        print("OK")
        return True
    except Exception as e:
        print(f"FAIL: {e}")
        return False


def test_strategist_context():
    """Test 2: Build strategist context from state."""
    print("[2/5] Building strategist context...", end=" ")
    try:
        from scripts.local_strategist import _build_strategist_context
        
        # Minimal state
        state = {
            'budget_remaining': 3,
            'current_phase': 1,
            'phases': {
                '1': {'experiments': []}
            },
            'known_facts': {'model': 'Qwen/Qwen2.5-1.5B'},
            'open_hypotheses': [
                {'id': 'H1', 'status': 'untested'}
            ]
        }
        
        context = _build_strategist_context(state, playbook="")
        
        assert 'available_scripts' in context
        assert 'completed_experiments' in context
        assert context['budget_remaining'] == 3
        
        print("OK")
        return True
    except Exception as e:
        print(f"FAIL: {e}")
        return False


def test_prompt_building():
    """Test 3: Build strategist prompt."""
    print("[3/5] Building strategist prompt...", end=" ")
    try:
        from scripts.local_strategist import _build_strategist_prompt
        
        context = {
            'available_scripts': ['run_qwen_test.py'],
            'completed_experiments': [],
            'overused_scripts': [],
            'budget_remaining': 3,
            'current_phase': 1,
            'playbook_tail': '',
            'known_facts': {},
            'open_hypotheses': [],
        }
        
        prompt = _build_strategist_prompt(context)
        
        # Check prompt contains expected sections
        assert "ABPT" in prompt or "Strategist" in prompt
        assert "OPEN QUESTIONS" in prompt
        assert "BUDGET" in prompt
        assert "OUTPUT FORMAT" in prompt
        assert "JSON" in prompt
        
        # Check prompt length reasonable
        assert len(prompt) > 500
        
        print(f"OK ({len(prompt)} chars)")
        return True
    except Exception as e:
        print(f"FAIL: {e}")
        return False


def test_response_parsing():
    """Test 4: Parse strategist JSON response."""
    print("[4/5] Parsing strategist response...", end=" ")
    try:
        from scripts.local_strategist import _parse_strategist_response
        
        # Valid JSON response
        valid_response = json.dumps({
            "id": "H1",
            "description": "Test tail_retention_ratio",
            "script": "run_qwen_phase_probe.py",
            "args": {"anchor_profile": "short"},
            "result_key": "correlation_summary.tail_retention_ratio",
            "success_threshold": 0.4,
            "reasoning": "Need to validate on short profile"
        })
        
        result = _parse_strategist_response(valid_response)
        
        assert result is not None
        assert result['id'] == 'H1'
        assert result['script'] == 'run_qwen_phase_probe.py'
        
        # Test with markdown code block
        markdown_response = "```json\n" + valid_response + "\n```"
        result2 = _parse_strategist_response(markdown_response)
        assert result2 is not None
        assert result2['id'] == 'H1'
        
        print("OK")
        return True
    except Exception as e:
        print(f"FAIL: {e}")
        return False


def test_strategist_selection():
    """Test 5: Full strategist selection pipeline."""
    print("[5/5] Full strategist pipeline...", end=" ")
    try:
        import os
        os.environ['USE_LOCAL_STRATEGIST'] = '1'
        
        from scripts.local_strategist import strategist_local_select
        
        # Minimal state
        state = {
            'budget_remaining': 3,
            'current_phase': 1,
            'phases': {
                '1': {'experiments': []}
            },
            'known_facts': {'model': 'Qwen/Qwen2.5-1.5B'},
            'open_hypotheses': []
        }
        
        # This should use heuristic to return a proposal
        result = strategist_local_select(state, playbook="")
        
        # Should return a valid proposal dict
        assert result is not None, "strategist_local_select returned None"
        assert 'id' in result, "Proposal missing 'id' field"
        assert 'description' in result, "Proposal missing 'description' field"
        assert 'script' in result, "Proposal missing 'script' field"
        
        print(f"OK (proposal: {result['id']})")
        return True
    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_orchestrator_integration():
    """Bonus: Test orchestrator can import local strategist."""
    print("[BONUS] Orchestrator integration...", end=" ")
    try:
        import os
        os.environ['USE_LOCAL_STRATEGIST'] = '1'
        
        # Check orchestrator can get local strategist
        sys.path.insert(0, '.')
        
        # Simulate what orchestrator does
        def _get_local_strategist():
            if os.environ.get("USE_LOCAL_STRATEGIST"):
                try:
                    from scripts.local_strategist import strategist_local_select
                    return strategist_local_select
                except ImportError:
                    return None
            return None
        
        local_strat = _get_local_strategist()
        assert local_strat is not None, "Could not load local strategist"
        
        print("OK")
        return True
    except Exception as e:
        print(f"FAIL: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("ABPT Local Strategist (Kimi Orchestrator) Test")
    print("=" * 60)
    
    start_time = time.time()
    
    results = []
    results.append(("Import", test_local_strategist_import()))
    results.append(("Context", test_strategist_context()))
    results.append(("Prompt", test_prompt_building()))
    results.append(("Parsing", test_response_parsing()))
    results.append(("Pipeline", test_strategist_selection()))
    results.append(("Orchestrator", test_orchestrator_integration()))
    
    elapsed = time.time() - start_time
    
    print("=" * 60)
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    print(f"Result: {passed}/{total} tests passed")
    print(f"Time: {elapsed:.2f} sec")
    
    if passed == total:
        print("[OK] All tests passed!")
        print("\nThe local strategist (Kimi orchestrator) is ready to use.")
        print("Run: python scripts/orchestrate.py --local-strategist")
        return 0
    else:
        print("[FAIL] Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

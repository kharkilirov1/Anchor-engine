"""
Demo: Показываем какие теории предлагает локальный стратег (Kimi orchestrator)
"""
import os
os.environ['USE_LOCAL_STRATEGIST'] = '1'

from scripts.local_strategist import strategist_local_select, _build_strategist_prompt, _build_strategist_context
import json


def demo_scenario(name, state, playbook=""):
    """Демонстрируем один сценарий."""
    print(f"\n{'='*60}")
    print(f"СЦЕНАРИЙ: {name}")
    print(f"{'='*60}")
    print(f"Состояние: фаза {state['current_phase']}, бюджет {state['budget_remaining']}")
    print(f"Выполнено экспериментов: {len(state['phases']['1']['experiments'])}")
    
    # Получаем предложение от стратега
    result = strategist_local_select(state, playbook)
    
    if result:
        print(f"\n[PREDICTION] STRATEGIST PROPOSAL:")
        print(f"   ID: {result['id']}")
        print(f"   Описание: {result['description']}")
        print(f"   Скрипт: {result['script']}")
        print(f"   Args: {json.dumps(result.get('args', {}), indent=2)}")
        print(f"   Метрика: {result.get('result_key', 'N/A')}")
        print(f"   Порог успеха: {result.get('success_threshold', 'N/A')}")
        print(f"   Обоснование: {result.get('reasoning', 'N/A')[:100]}...")
    else:
        print("\n⚠️ Стратег не вернул предложение (fallback на rule-based)")
    
    return result


def main():
    print("="*60)
    print("DEMO: Local Strategist (Kimi orchestrator)")
    print("="*60)
    print("\nLooking at theories/experiments proposed by strategist")
    print("depending on research state...")
    
    # Сценарий 1: Начало исследования (пустое состояние)
    state1 = {
        'budget_remaining': 10,
        'current_phase': 1,
        'phases': {'1': {'experiments': []}},
        'known_facts': {'model': 'Qwen/Qwen2.5-1.5B'},
        'open_hypotheses': []
    }
    demo_scenario("Start of research (clean slate)", state1)
    
    # Сценарий 2: H1 уже выполнен успешно
    state2 = {
        'budget_remaining': 9,
        'current_phase': 1,
        'phases': {
            '1': {
                'experiments': [
                    {'hypothesis_id': 'H1', 'status': 'success', 'metric_value': 0.642}
                ]
            }
        },
        'known_facts': {
            'model': 'Qwen/Qwen2.5-1.5B',
            'tail_retention_ratio_rho_medium': 0.642
        },
        'open_hypotheses': [
            {'id': 'H1_short', 'status': 'untested'},
            {'id': 'H1_long', 'status': 'untested'},
        ]
    }
    demo_scenario("H1 confirmed (rho=0.642), need short/long", state2)
    
    # Сценарий 3: H1_short тоже выполнен
    state3 = {
        'budget_remaining': 8,
        'current_phase': 1,
        'phases': {
            '1': {
                'experiments': [
                    {'hypothesis_id': 'H1', 'status': 'success', 'metric_value': 0.642},
                    {'hypothesis_id': 'H1_short', 'status': 'success', 'metric_value': 0.58}
                ]
            }
        },
        'known_facts': {
            'model': 'Qwen/Qwen2.5-1.5B',
            'tail_retention_ratio_rho_medium': 0.642,
            'tail_retention_ratio_rho_short': 0.58
        },
        'open_hypotheses': [
            {'id': 'H1_long', 'status': 'untested'},
            {'id': 'H2', 'status': 'untested'},
        ]
    }
    demo_scenario("H1 + H1_short done, remaining H1_long", state3)
    
    # Сценарий 4: Переход к фазе 2
    state4 = {
        'budget_remaining': 6,
        'current_phase': 2,
        'phases': {
            '1': {
                'experiments': [
                    {'hypothesis_id': 'H1', 'status': 'success'},
                    {'hypothesis_id': 'H1_short', 'status': 'success'},
                    {'hypothesis_id': 'H1_long', 'status': 'success'}
                ]
            },
            '2': {'experiments': []}
        },
        'known_facts': {
            'model': 'Qwen/Qwen2.5-1.5B',
            'phase_1_complete': True
        },
        'open_hypotheses': [
            {'id': 'H2', 'status': 'untested'},
            {'id': 'H3', 'status': 'untested'},
        ]
    }
    demo_scenario("Phase 1 complete, transition to phase 2", state4)
    
    # Сценарий 5: С заполненным playbook
    playbook5 = """
## Recent Findings
- tail_retention_ratio confirmed with rho=+0.642 on medium profile
- early_slope_4_8 is NEGATIVE predictor (rho=-0.14)
- Crystallization zone: L4-L8

## Blocked Scripts
- run_qwen_phase_probe.py (used 3+ times)
"""
    state5 = {
        'budget_remaining': 5,
        'current_phase': 1,
        'phases': {
            '1': {
                'experiments': [
                    {'hypothesis_id': 'H1', 'status': 'success', 'metric_value': 0.642}
                ]
            }
        },
        'known_facts': {},
        'open_hypotheses': []
    }
    demo_scenario("With filled playbook (more context)", state5, playbook5)
    
    print("\n" + "="*60)
    print("ВЫВОДЫ:")
    print("="*60)
    print("""
1. Локальный стратег использует ЭВРИСТИКУ для выбора экспериментов
2. Он предлагает последовательно: H1 → H1_short → H1_long → H2
3. Выбор основан на том, какие эксперименты уже выполнены
4. Стратег старается валидировать подтвержденные находки на разных профилях
5. Для реального AI (Kimi) можно использовать интерактивный режим:
   - Выводится prompt с контекстом
   - Ты принимаешь решение или корректируешь
   - Сохраняешь ответ в /tmp/ABPT_strategist_response.json
""")


if __name__ == "__main__":
    main()

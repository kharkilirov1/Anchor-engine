"""
Lightweight smoke test for ABPT.
Runs in seconds even on low-end PC.
"""
from __future__ import annotations

import sys
import time
from dataclasses import replace


def test_imports():
    """Test 1: Check imports of all key modules."""
    print("[1/5] Checking imports...", end=" ")
    try:
        import torch
        from src.model.config import TOY_CONFIG, ModelConfig
        from src.model.backbone import Backbone
        from src.model.abpt import ABPTModel
        from src.model.attention import MultiHeadAttention, AttentionResidual
        from src.model.branches import BranchRouter
        from src.model.verifier import Verifier
        from src.model.plastic import PlasticLayer
        print("OK")
        return True
    except Exception as e:
        print(f"FAIL: {e}")
        return False


def test_config_creation():
    """Test 2: Create configuration."""
    print("[2/5] Creating config...", end=" ")
    try:
        from src.model.config import TOY_CONFIG
        cfg = replace(TOY_CONFIG)
        assert cfg.vocab_size == 512
        assert cfg.d_model == 64
        assert cfg.n_layers == 3
        print(f"OK (vocab={cfg.vocab_size}, d_model={cfg.d_model})")
        return True
    except Exception as e:
        print(f"FAIL: {e}")
        return False


def test_model_creation():
    """Test 3: Create model with parameter count."""
    print("[3/5] Creating model...", end=" ")
    try:
        import torch
        from src.model.config import TOY_CONFIG
        from src.model.abpt import ABPTModel
        from dataclasses import replace
        
        cfg = replace(TOY_CONFIG)
        model = ABPTModel(cfg)
        
        # Подсчет параметров
        n_params = sum(p.numel() for p in model.parameters())
        print(f"OK ({n_params:,} params, ~{n_params * 4 / 1024 / 1024:.2f} MB)")
        return True
    except Exception as e:
        print(f"FAIL: {e}")
        return False


def test_forward_pass():
    """Test 4: Forward pass on random data."""
    print("[4/5] Forward pass...", end=" ")
    try:
        import torch
        from src.model.config import TOY_CONFIG
        from src.model.abpt import ABPTModel
        from dataclasses import replace
        
        cfg = replace(TOY_CONFIG)
        model = ABPTModel(cfg)
        model.eval()
        
        # Очень маленький батч
        batch_size = 2
        seq_len = 16
        
        x = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))
        targets = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))
        
        with torch.no_grad():
            out = model(x, targets)
        
        # Проверки
        assert "logits" in out
        assert "loss" in out
        assert out["logits"].shape == (batch_size, seq_len, cfg.vocab_size)
        assert out["loss"].item() > 0
        
        print(f"OK (loss={out['loss'].item():.4f})")
        return True
    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_anchor_components():
    """Test 5: Basic anchor components."""
    print("[5/5] Anchor components...", end=" ")
    try:
        import torch
        from src.model.config import TOY_CONFIG
        from src.model.anchor_detector import AnchorDetector
        from src.model.anchor_memory import AnchorMemory
        from src.model.anchor_types import AnchorState
        from dataclasses import replace
        
        cfg = replace(TOY_CONFIG, anchor_threshold=0.5)
        
        # Тест детектора
        detector = AnchorDetector(cfg)
        hidden = torch.randn(2, 8, cfg.d_model)
        history = torch.randn(2, 8, cfg.d_model)
        
        out = detector(hidden, history)
        assert "scores" in out
        assert out["scores"].shape == (2, 8)
        
        # Тест памяти
        memory = AnchorMemory(cfg)
        assert memory is not None
        
        print("OK")
        return True
    except Exception as e:
        print(f"FAIL: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("ABPT Lightweight Smoke Test")
    print("=" * 60)
    print(f"Python: {sys.version.split()[0]}")
    
    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    except:
        pass
    
    print("=" * 60)
    
    start_time = time.time()
    
    results = []
    results.append(("Imports", test_imports()))
    results.append(("Config", test_config_creation()))
    results.append(("Model Creation", test_model_creation()))
    results.append(("Forward Pass", test_forward_pass()))
    results.append(("Anchor Components", test_anchor_components()))
    
    elapsed = time.time() - start_time
    
    print("=" * 60)
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    print(f"Result: {passed}/{total} tests passed")
    print(f"Time: {elapsed:.2f} sec")
    
    if passed == total:
        print("[OK] All tests passed!")
        return 0
    else:
        print("[FAIL] Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

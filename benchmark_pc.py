"""
Benchmark runs tailored for your PC specs:
- CPU: Intel 10 cores / 20 threads
- RAM: 32 GB
- No GPU (CPU-only PyTorch)
"""
from __future__ import annotations

import sys
import time
import torch
import torch.nn as nn
from dataclasses import replace

from src.model.config import TOY_CONFIG, ModelConfig
from src.model.abpt import ABPTModel
from src.model.abpt_anchor_v1 import ABPTAnchorV1


def benchmark_forward_backward(model: nn.Module, batch_size: int, seq_len: int, vocab_size: int, device: str = "cpu"):
    """Benchmark forward + backward pass."""
    model = model.to(device)
    model.train()
    
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    
    # Warmup
    for _ in range(2):
        out = model(x, targets)
        out["loss"].backward()
        model.zero_grad()
    
    # Benchmark
    times = []
    for _ in range(5):
        start = time.perf_counter()
        out = model(x, targets)
        loss = out["loss"]
        loss.backward()
        model.zero_grad()
        times.append(time.perf_counter() - start)
    
    return sum(times) / len(times)


def benchmark_inference(model: nn.Module, batch_size: int, seq_len: int, vocab_size: int, device: str = "cpu"):
    """Benchmark inference only (no grad)."""
    model = model.to(device)
    model.eval()
    
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model(x)
    
    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(10):
            start = time.perf_counter()
            _ = model(x)
            times.append(time.perf_counter() - start)
    
    return sum(times) / len(times)


def run_benchmarks():
    """Run all benchmarks optimized for your PC."""
    print("=" * 60)
    print("ABPT Benchmark for Your PC")
    print("=" * 60)
    print(f"Device: CPU (Intel 10-core)")
    print(f"PyTorch: {torch.__version__}")
    print(f"Threads: {torch.get_num_threads()}")
    print("=" * 60)
    
    device = "cpu"
    results = []
    
    # Test 1: Tiny model (233K params) - very light
    print("\n[1/6] Tiny Model (233K params) - Forward+Backward")
    cfg_tiny = replace(TOY_CONFIG)
    model_tiny = ABPTModel(cfg_tiny)
    t = benchmark_forward_backward(model_tiny, batch_size=4, seq_len=32, vocab_size=cfg_tiny.vocab_size, device=device)
    print(f"  Batch=4, Seq=32: {t*1000:.1f} ms/iter")
    results.append(("Tiny FWD+BWD (B4,S32)", t*1000))
    
    # Test 2: Tiny model inference
    print("\n[2/6] Tiny Model - Inference only")
    t = benchmark_inference(model_tiny, batch_size=8, seq_len=64, vocab_size=cfg_tiny.vocab_size, device=device)
    print(f"  Batch=8, Seq=64: {t*1000:.1f} ms/iter")
    results.append(("Tiny Inference (B8,S64)", t*1000))
    
    # Test 3: Small model (1M params)
    print("\n[3/6] Small Model (~1M params)")
    cfg_small = ModelConfig(
        vocab_size=1000,
        d_model=128,
        n_heads=4,
        n_layers=4,
        d_ff=256,
        max_seq_len=128,
        use_attn_res=True,
        use_branches=True,
        use_verifier=True,
        use_plastic=True,
    )
    model_small = ABPTModel(cfg_small)
    n_params = sum(p.numel() for p in model_small.parameters())
    print(f"  Params: {n_params:,} ({n_params/1e6:.2f}M)")
    t = benchmark_forward_backward(model_small, batch_size=2, seq_len=64, vocab_size=cfg_small.vocab_size, device=device)
    print(f"  Batch=2, Seq=64: {t*1000:.1f} ms/iter")
    results.append(("Small FWD+BWD (B2,S64)", t*1000))
    
    # Test 4: Small model inference
    print("\n[4/6] Small Model - Inference")
    t = benchmark_inference(model_small, batch_size=4, seq_len=128, vocab_size=cfg_small.vocab_size, device=device)
    print(f"  Batch=4, Seq=128: {t*1000:.1f} ms/iter")
    throughput = (4 * 128) / t  # tokens/sec
    print(f"  Throughput: {throughput:.0f} tokens/sec")
    results.append(("Small Inference (B4,S128)", t*1000))
    
    # Test 5: Anchor V1 model
    print("\n[5/6] Anchor V1 Model (~1M params)")
    cfg_anchor = ModelConfig(
        vocab_size=1000,
        d_model=128,
        n_heads=4,
        n_layers=4,
        d_ff=256,
        max_seq_len=128,
        anchor_threshold=0.5,
        anchor_ttl_init=4.0,
    )
    model_anchor = ABPTAnchorV1(cfg_anchor)
    t = benchmark_forward_backward(model_anchor, batch_size=2, seq_len=32, vocab_size=cfg_anchor.vocab_size, device=device)
    print(f"  Batch=2, Seq=32: {t*1000:.1f} ms/iter")
    results.append(("Anchor V1 FWD+BWD", t*1000))
    
    # Test 6: Memory usage estimation
    print("\n[6/6] Memory Estimation")
    sizes = [
        (TOY_CONFIG, "Tiny (233K)"),
        (cfg_small, "Small (1M)"),
        (cfg_anchor, "Anchor (1M+)"),
    ]
    for cfg, name in sizes:
        model = ABPTModel(cfg)
        n_params = sum(p.numel() for p in model.parameters())
        param_memory = n_params * 4 / 1024 / 1024  # float32 = 4 bytes
        print(f"  {name}: {param_memory:.1f} MB params")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, time_ms in results:
        print(f"  {name:30s}: {time_ms:6.1f} ms")
    print("=" * 60)
    
    # Recommendations
    print("\n[RECOMMENDATIONS FOR YOUR PC]")
    print("- CPU-only PyTorch detected")
    print("- Use batch_size <= 4 for training")
    print("- Use batch_size <= 8 for inference")
    print("- Tiny models (233K) are fastest")
    print("- Small models (1M) are usable but slower")
    print("- For actual training: use --preset toy --device cpu")
    
    return results


if __name__ == "__main__":
    # Set optimal thread count for your 10-core CPU
    torch.set_num_threads(10)
    print(f"Set PyTorch threads: {torch.get_num_threads()}")
    
    results = run_benchmarks()
    sys.exit(0)

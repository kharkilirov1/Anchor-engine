# 📊 Benchmark Results for Your PC

## 🖥️ System Specs
```
OS:        Windows 11
CPU:       Intel 10 cores (20 threads)
RAM:       32 GB (17 GB available)
PyTorch:   2.10.0 (CPU-only)
GPU:       None
Optimization: 10 PyTorch threads
```

---

## 🧪 Test 1: Model Performance (benchmark_pc.py)

| Test | Config | Time | Notes |
|------|--------|------|-------|
| **Tiny FWD+BWD** | 233K params, B=4, S=32 | **36.5 ms** | Training speed |
| **Tiny Inference** | 233K params, B=8, S=64 | **14.2 ms** | Fastest inference |
| **Small FWD+BWD** | 1.08M params, B=2, S=64 | **42.5 ms** | Usable training |
| **Small Inference** | 1.08M params, B=4, S=128 | **20.0 ms** | **25,622 tokens/sec** |
| **Anchor V1 FWD+BWD** | ~1M params, B=2, S=32 | **114.6 ms** | Anchor overhead ~3x |

### Memory Footprint
| Model | Parameters | Memory |
|-------|-----------|--------|
| Tiny | 233K | 0.9 MB |
| Small | 1.08M | 4.1 MB |
| Anchor V1 | ~1M | 4.1 MB |

### ✅ Your PC can handle:
- **Training**: Tiny models (batch ≤ 4), Small models (batch ≤ 2)
- **Inference**: Up to 25K tokens/sec on Small model
- **Anchor experiments**: Full pipeline works, 3x slower than base

---

## 🎯 Test 2: Local Strategist Performance (benchmark_local_strategist.py)

| Component | Time | Throughput |
|-----------|------|------------|
| **Context building** | 0.702 ms | 1,425 calls/sec |
| **Prompt building** | 0.013 ms | 76,923 calls/sec |
| **Response parsing** | 0.005 ms | 186,546 calls/sec |
| **Full pipeline** | **3.6 ms** | **282 decisions/sec** |

### 🚀 Speedup vs API
| Metric | Local (Your PC) | DeepSeek API | Speedup |
|--------|----------------|--------------|---------|
| One decision | **3.6 ms** | 1,000-3,000 ms | **278-833x** |
| 3 experiments | **11 ms** | 3,000-9,000 ms | **278-833x** |
| Network latency | **0 ms** | 100-500 ms | ∞ |
| Cost | **$0** | $0.001-0.01/call | Free |

---

## 📋 Test 3: Pipeline Integration (test_local_strategist.py)

```
✅ All 6 tests passed (0.03 sec)
├── Import local_strategist
├── Build strategist context
├── Build prompt (1707 chars)
├── Parse JSON response
├── Full pipeline → proposal: H1_long
└── Orchestrator integration
```

---

## 💡 Recommendations for Your PC

### For Training:
```bash
# Fastest - use this for experiments
python train.py --preset toy --device cpu --steps 1000

# Slower but better quality
python train.py --preset toy --device cpu --steps 5000 --batch-size 2
```

### For Orchestrator:
```bash
# Use local strategist (FAST!)
python scripts/orchestrate.py --budget 3 --local-strategist

# Or with environment variable
set USE_LOCAL_STRATEGIST=1
python scripts/orchestrate.py --budget 3 --llm-strategist
```

### Model Size Limits:
| Model | Max Batch (Train) | Max Batch (Inference) |
|-------|------------------|----------------------|
| Tiny (233K) | 4-8 | 16-32 |
| Small (1M) | 2-4 | 8-16 |
| Medium (10M+) | ❌ Too slow | 2-4 |

---

## ⚡ Performance Summary

```
Your PC is GOOD for:
✅ Tiny model training/experiments
✅ Small model inference
✅ Local orchestrator (282 decisions/sec!)
✅ Anchor prototype development

Your PC is LIMITED for:
❌ Large model training (> 10M params)
❌ Batch sizes > 8
❌ Long sequences (> 512) with large models
```

---

## 🎬 Quick Start Commands

```bash
# 1. Run lightweight test
python test_lightweight.py

# 2. Run model benchmark
python benchmark_pc.py

# 3. Test local strategist
python test_local_strategist.py

# 4. Run orchestrator with local AI
python scripts/orchestrate.py --budget 3 --local-strategist
```

---

*Generated for: Intel 10-core, 32GB RAM, CPU-only PyTorch*

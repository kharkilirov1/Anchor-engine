import json
from dataclasses import replace

from evaluate import evaluate
import train as train_module
from src.model.abpt import ABPTModel
from src.model.abpt_b import ABPTModelB
from src.model.abpt_anchor_v1 import ABPTAnchorV1
from src.model.config import TOY_CONFIG


def _reset_train_data() -> None:
    train_module._train_data = None
    train_module._val_data = None


def test_build_model_supports_anchor_stage():
    cfg = replace(TOY_CONFIG, max_seq_len=16, vocab_size=64)

    model_a = train_module.build_model(cfg, stage="a", device="cpu")
    model_b = train_module.build_model(cfg, stage="b", device="cpu")
    model_anchor = train_module.build_model(cfg, stage="anchor", device="cpu")

    assert isinstance(model_a, ABPTModel)
    assert isinstance(model_b, ABPTModelB)
    assert isinstance(model_anchor, ABPTAnchorV1)


def test_stage_b_short_runs_cap_eq_warmup():
    cfg = replace(TOY_CONFIG, max_steps=8, eq_warmup_steps=50)
    effective = train_module._prepare_effective_cfg(cfg, stage="b")
    assert effective.eq_warmup_steps == 2
    assert cfg.eq_warmup_steps == 50


def test_train_anchor_stage_runs_on_random_data(tmp_path):
    _reset_train_data()
    cfg = replace(
        TOY_CONFIG,
        vocab_size=64,
        max_seq_len=16,
        batch_size=2,
        eval_interval=1,
        max_steps=1,
    )

    model = train_module.train(cfg, device="cpu", stage="anchor", data_dir=str(tmp_path))

    assert isinstance(model, ABPTAnchorV1)
    assert hasattr(model, "training_history")
    assert len(model.training_history) == 1
    assert "anchors_active" in model.training_history[0]


def test_train_anchor_stage_can_save_history(tmp_path):
    _reset_train_data()
    cfg = replace(
        TOY_CONFIG,
        vocab_size=64,
        max_seq_len=16,
        batch_size=2,
        eval_interval=1,
        max_steps=1,
    )
    history_path = tmp_path / "anchor_history.json"

    model = train_module.train(
        cfg,
        device="cpu",
        stage="anchor",
        data_dir=str(tmp_path),
        history_path=str(history_path),
    )

    assert isinstance(model, ABPTAnchorV1)
    assert history_path.exists()
    payload = json.loads(history_path.read_text(encoding="utf-8"))
    assert len(payload) == 1
    assert "proposal_influence" in payload[0]
    assert "detector_alignment_loss" in payload[0]


def test_train_anchor_stage_runs_on_synthetic_anchor_data(tmp_path):
    _reset_train_data()
    cfg = replace(
        TOY_CONFIG,
        vocab_size=64,
        max_seq_len=24,
        batch_size=4,
        eval_interval=1,
        max_steps=1,
    )

    model = train_module.train(
        cfg,
        device="cpu",
        stage="anchor",
        data_dir=str(tmp_path),
        dataset="anchor-synthetic",
    )

    assert isinstance(model, ABPTAnchorV1)
    assert len(model.training_history) == 1
    assert model.training_history[0]["anchors_active"] >= 0.0


def test_evaluate_anchor_stage_returns_anchor_metrics(tmp_path):
    _reset_train_data()
    cfg = replace(
        TOY_CONFIG,
        vocab_size=64,
        max_seq_len=16,
        batch_size=2,
    )

    metrics = evaluate(cfg, device="cpu", stage="anchor", data_dir=str(tmp_path), num_batches=2)

    assert "loss" in metrics
    assert "bpb" in metrics
    assert "anchors_active" in metrics
    assert "proposal_influence" in metrics
    assert "component_detector_alignment_loss" in metrics
    assert "component_context_stability_loss" in metrics


def test_evaluate_anchor_stage_on_synthetic_anchor_data(tmp_path):
    _reset_train_data()
    cfg = replace(
        TOY_CONFIG,
        vocab_size=64,
        max_seq_len=24,
        batch_size=4,
    )

    metrics = evaluate(
        cfg,
        device="cpu",
        stage="anchor",
        data_dir=str(tmp_path),
        dataset="anchor-synthetic",
        num_batches=2,
    )

    assert "anchors_active" in metrics
    assert "anchor_contradiction" in metrics


def test_train_baseline_stage_runs_on_cached_the_stack_data(tmp_path):
    _reset_train_data()
    cache_path = tmp_path / "bigcode__the_stack_smol_xs_python_1024.bin"
    cache_path.write_bytes((b"def foo():\n    return 1\n" * 128))
    cfg = replace(
        TOY_CONFIG,
        vocab_size=64,
        max_seq_len=16,
        batch_size=2,
        eval_interval=1,
        max_steps=1,
    )

    model = train_module.train(
        cfg,
        device="cpu",
        stage="a",
        data_dir=str(tmp_path),
        dataset="the-stack",
        the_stack_repo="bigcode/the-stack-smol-xs",
        the_stack_lang="python",
        the_stack_bytes=1024,
    )

    assert isinstance(model, ABPTModel)
    assert len(model.training_history) == 1
    assert "loss" in model.training_history[0]


def test_evaluate_anchor_stage_on_cached_the_stack_data(tmp_path):
    _reset_train_data()
    cache_path = tmp_path / "bigcode__the_stack_smol_xs_python_1024.bin"
    cache_path.write_bytes((b"class A:\n    pass\n" * 128))
    cfg = replace(
        TOY_CONFIG,
        vocab_size=64,
        max_seq_len=16,
        batch_size=2,
    )

    metrics = evaluate(
        cfg,
        device="cpu",
        stage="anchor",
        data_dir=str(tmp_path),
        dataset="the-stack",
        num_batches=2,
        the_stack_repo="bigcode/the-stack-smol-xs",
        the_stack_lang="python",
        the_stack_bytes=1024,
    )

    assert "loss" in metrics
    assert "anchors_active" in metrics


def test_train_anchor_stage_runs_on_cached_the_stack_bpe_data(tmp_path):
    _reset_train_data()
    prefix = "bigcode__the_stack_smol_xs_python_1024_bpe128"
    cache_path = tmp_path / f"{prefix}_ids.pt"
    torch_ids = __import__("torch").tensor([i % 128 for i in range(512)], dtype=__import__("torch").long)
    __import__("torch").save(torch_ids, cache_path)
    (tmp_path / f"{prefix}_meta.json").write_text('{"vocab_size": 128, "token_count": 512}', encoding="utf-8")
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.add_special_tokens(["[PAD]", "[UNK]", "[BOS]", "[EOS]"])
    tokenizer.add_tokens(["def", "foo", "return", "1"])
    tokenizer.save(str(tmp_path / f"{prefix}_tokenizer.json"))

    cfg = replace(
        TOY_CONFIG,
        vocab_size=64,
        max_seq_len=16,
        batch_size=2,
        eval_interval=1,
        max_steps=1,
    )

    model = train_module.train(
        cfg,
        device="cpu",
        stage="anchor",
        data_dir=str(tmp_path),
        dataset="the-stack-bpe",
        the_stack_repo="bigcode/the-stack-smol-xs",
        the_stack_lang="python",
        the_stack_bytes=1024,
        the_stack_vocab_size=128,
    )

    assert isinstance(model, ABPTAnchorV1)
    assert len(model.training_history) == 1
    assert "anchors_active" in model.training_history[0]


def test_train_anchor_stage_runs_on_cached_tinystories_bpe_data(tmp_path):
    _reset_train_data()
    prefix = "roneneldan__TinyStories_tinystories_1024_bpe128"
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.add_special_tokens(["[PAD]", "[UNK]", "[BOS]", "[EOS]"])
    tokenizer.add_tokens(["Once", "upon", "a", "time"])
    tokenizer.save(str(tmp_path / f"{prefix}_tokenizer.json"))
    torch_mod = __import__("torch")
    torch_mod.save(torch_mod.tensor([i % 128 for i in range(512)], dtype=torch_mod.long), tmp_path / f"{prefix}_train_ids.pt")
    torch_mod.save(torch_mod.tensor([i % 128 for i in range(128)], dtype=torch_mod.long), tmp_path / f"{prefix}_val_ids.pt")
    (tmp_path / f"{prefix}_meta.json").write_text('{"vocab_size": 128, "train_token_count": 512, "val_token_count": 128}', encoding="utf-8")

    cfg = replace(
        TOY_CONFIG,
        vocab_size=64,
        max_seq_len=16,
        batch_size=2,
        eval_interval=1,
        max_steps=1,
    )

    model = train_module.train(
        cfg,
        device="cpu",
        stage="anchor",
        data_dir=str(tmp_path),
        dataset="tinystories-bpe",
        tinystories_repo="roneneldan/TinyStories",
        tinystories_bytes=1024,
        tinystories_vocab_size=128,
    )

    assert isinstance(model, ABPTAnchorV1)
    assert len(model.training_history) == 1
    assert "anchors_active" in model.training_history[0]


def test_train_anchor_stage_runs_on_cached_openwebmath_bpe_data(tmp_path):
    _reset_train_data()
    prefix = "open_web_math__open_web_math_1024_bpe128"
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.add_special_tokens(["[PAD]", "[UNK]", "[BOS]", "[EOS]"])
    tokenizer.add_tokens(["Theorem", "proof", "integral", "limit"])
    tokenizer.save(str(tmp_path / f"{prefix}_tokenizer.json"))
    torch_mod = __import__("torch")
    torch_mod.save(torch_mod.tensor([i % 128 for i in range(512)], dtype=torch_mod.long), tmp_path / f"{prefix}_ids.pt")
    (tmp_path / f"{prefix}_meta.json").write_text('{"vocab_size": 128, "token_count": 512}', encoding="utf-8")

    cfg = replace(
        TOY_CONFIG,
        vocab_size=64,
        max_seq_len=16,
        batch_size=2,
        eval_interval=1,
        max_steps=1,
    )

    model = train_module.train(
        cfg,
        device="cpu",
        stage="anchor",
        data_dir=str(tmp_path),
        dataset="openwebmath-bpe",
        openwebmath_repo="open-web-math/open-web-math",
        openwebmath_bytes=1024,
        openwebmath_vocab_size=128,
    )

    assert isinstance(model, ABPTAnchorV1)
    assert len(model.training_history) == 1
    assert "anchors_active" in model.training_history[0]

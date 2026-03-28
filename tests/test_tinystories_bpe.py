import json
from pathlib import Path

import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers

from src.data.tinystories_bpe import load_tinystories_bpe


def test_load_tinystories_bpe_uses_cached_token_ids(tmp_path):
    prefix = "roneneldan__TinyStories_tinystories_1024_bpe128"
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.add_special_tokens(["[PAD]", "[UNK]", "[BOS]", "[EOS]"])
    tokenizer.add_tokens(["Once", "upon", "a", "time"])
    tokenizer.save(str(Path(tmp_path) / f"{prefix}_tokenizer.json"))

    train_ids = torch.tensor([i % 128 for i in range(512)], dtype=torch.long)
    val_ids = torch.tensor([i % 128 for i in range(128)], dtype=torch.long)
    torch.save(train_ids, Path(tmp_path) / f"{prefix}_train_ids.pt")
    torch.save(val_ids, Path(tmp_path) / f"{prefix}_val_ids.pt")
    Path(tmp_path, f"{prefix}_meta.json").write_text(
        json.dumps({"vocab_size": 128, "train_token_count": 512, "val_token_count": 128}),
        encoding="utf-8",
    )

    train, val = load_tinystories_bpe(
        seq_len=16,
        device="cpu",
        data_dir=str(tmp_path),
        repo_id="roneneldan/TinyStories",
        target_bytes=1024,
        vocab_size=128,
    )

    x, y = train.get_batch(2)

    assert x.shape == (2, 16)
    assert y.shape == (2, 16)
    assert train.vocab_size == 128
    assert len(train) == 512
    assert len(val) == 128

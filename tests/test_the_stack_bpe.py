import json
from pathlib import Path

import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers

from src.data.the_stack_bpe import BPETokenDataset, load_the_stack_bpe


def test_bpe_token_dataset_shapes():
    token_ids = torch.arange(512, dtype=torch.long)
    ds = BPETokenDataset(token_ids=token_ids, vocab_size=128, split="train", seq_len=16, device="cpu")

    x, y = ds.get_batch(4)

    assert x.shape == (4, 16)
    assert y.shape == (4, 16)
    assert ds.vocab_size == 128


def test_load_the_stack_bpe_uses_cached_token_ids(tmp_path):
    prefix = "bigcode__the_stack_smol_xs_python_1024_bpe128"
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.add_special_tokens(["[PAD]", "[UNK]", "[BOS]", "[EOS]"])
    tokenizer.add_tokens(["def", "foo", "return", "1"])
    tokenizer.save(str(Path(tmp_path) / f"{prefix}_tokenizer.json"))

    token_ids = torch.tensor(list(range(256)), dtype=torch.long)
    torch.save(token_ids, Path(tmp_path) / f"{prefix}_ids.pt")
    Path(tmp_path, f"{prefix}_meta.json").write_text(
        json.dumps({"vocab_size": 128, "token_count": 256}),
        encoding="utf-8",
    )

    train, val = load_the_stack_bpe(
        seq_len=16,
        device="cpu",
        data_dir=str(tmp_path),
        repo_id="bigcode/the-stack-smol-xs",
        lang="python",
        target_bytes=1024,
        vocab_size=128,
    )

    x, y = train.get_batch(2)

    assert x.shape == (2, 16)
    assert y.shape == (2, 16)
    assert train.vocab_size == 128
    assert len(train) > len(val)

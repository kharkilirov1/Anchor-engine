from pathlib import Path

from src.data.the_stack import ByteCorpusDataset, load_the_stack


def test_byte_corpus_dataset_shapes():
    data = bytes(range(256)) * 8
    ds = ByteCorpusDataset(data=data, split="train", seq_len=16, device="cpu")

    x, y = ds.get_batch(4)

    assert x.shape == (4, 16)
    assert y.shape == (4, 16)
    assert ds.vocab_size == 256


def test_load_the_stack_uses_cached_bytes(tmp_path):
    cache_path = Path(tmp_path) / "bigcode__the_stack_smol_xs_python_1024.bin"
    cache_path.write_bytes((b"print('hello')\n" * 128))

    train, val = load_the_stack(
        seq_len=16,
        device="cpu",
        data_dir=str(tmp_path),
        repo_id="bigcode/the-stack-smol-xs",
        lang="python",
        target_bytes=1024,
    )

    x, y = train.get_batch(2)

    assert x.shape == (2, 16)
    assert y.shape == (2, 16)
    assert len(train) > len(val)

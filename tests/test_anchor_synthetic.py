from src.data.anchor_synthetic import AnchorSyntheticDataset, load_anchor_synthetic


def test_anchor_synthetic_dataset_shapes():
    train, val = load_anchor_synthetic(device="cpu")

    x, y = train.get_batch(4)

    assert x.shape == (4, 24)
    assert y.shape == (4, 24)
    assert train.vocab_size >= 55
    assert len(train) > len(val)


def test_anchor_synthetic_dataset_len_is_repeat_scaled():
    train = AnchorSyntheticDataset(split="train", train_repeats=2, val_repeats=1)
    val = AnchorSyntheticDataset(split="val", train_repeats=2, val_repeats=1)

    assert len(train) == 18
    assert len(val) == 9

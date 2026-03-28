import torch
from src.model.attention import MultiHeadAttention, AttentionResidual


def test_mha_output_shape():
    mha = MultiHeadAttention(d_model=64, n_heads=4, dropout=0.0)
    x = torch.randn(2, 16, 64)
    out = mha(x, x, x)
    assert out.shape == (2, 16, 64)


def test_mha_causal_mask():
    mha = MultiHeadAttention(d_model=64, n_heads=4, dropout=0.0)
    x = torch.randn(1, 8, 64)
    out = mha(x, x, x, causal=True)
    assert out.shape == (1, 8, 64)


def test_attnres_output_shape():
    attnres = AttentionResidual(d_model=64, layer_idx=3)
    layer_outputs = [torch.randn(2, 16, 64) for _ in range(4)]
    current = torch.randn(2, 16, 64)
    out = attnres(current, layer_outputs)
    assert out.shape == (2, 16, 64)


def test_attnres_single_layer():
    attnres = AttentionResidual(d_model=64, layer_idx=0)
    layer_outputs = [torch.randn(2, 16, 64)]
    current = torch.randn(2, 16, 64)
    out = attnres(current, layer_outputs)
    assert out.shape == (2, 16, 64)


def test_attnres_empty_layers():
    attnres = AttentionResidual(d_model=64, layer_idx=0)
    current = torch.randn(2, 16, 64)
    out = attnres(current, [])
    assert out.shape == (2, 16, 64)

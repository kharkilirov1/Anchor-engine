import torch
from src.model.config import TOY_CONFIG
from src.model.backbone import Backbone
from dataclasses import replace


def test_backbone_output_shape():
    cfg = replace(TOY_CONFIG)
    backbone = Backbone(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 32))
    out = backbone(x)
    assert out["hidden"].shape == (2, 32, cfg.d_model)
    assert "layer_outputs" in out


def test_backbone_no_attnres():
    cfg = replace(TOY_CONFIG, use_attn_res=False)
    backbone = Backbone(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 32))
    out = backbone(x)
    assert out["hidden"].shape == (2, 32, cfg.d_model)


def test_backbone_layer_outputs_count():
    cfg = replace(TOY_CONFIG)
    backbone = Backbone(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    out = backbone(x)
    # input embedding + n_layers
    assert len(out["layer_outputs"]) == cfg.n_layers + 1


def test_backbone_causal():
    cfg = replace(TOY_CONFIG, dropout=0.0)
    backbone = Backbone(cfg)
    backbone.eval()
    x = torch.randint(0, cfg.vocab_size, (1, 16))
    out1 = backbone(x)["hidden"][0, 5, :]
    x2 = x.clone()
    x2[0, 6:] = torch.randint(0, cfg.vocab_size, (10,))
    out2 = backbone(x2)["hidden"][0, 5, :]
    assert torch.allclose(out1, out2, atol=1e-5)

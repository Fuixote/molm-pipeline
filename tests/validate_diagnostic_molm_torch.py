#!/usr/bin/env python
"""PyTorch-only smoke validation for migrated DiagnosticMOLM."""

import os
import random
import sys

os.environ.setdefault("MOLM_OUTPUT_DIR", "/tmp/molm_pipeline_results")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase0_config import (
    DiagnosticMOLM,
    focal_bce_with_logits,
    gap_hinge_loss,
    ranking_loss,
)


SEED = 42


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DiagnosticMOLM(
        input_dim=11,
        latent_dim=5,
        shared_dims=[13, 7],
        tower_dims=[6, 4],
        dropout_rate=0.2,
    ).to(device)

    x = torch.randn(8, 11, device=device)
    y_aff = torch.tensor([0, 1, 1, 0, 1, 0, 1, 0], dtype=torch.float32, device=device)
    y_spec = torch.tensor([1, 0, 1, 0, 0, 1, 0, 1], dtype=torch.float32, device=device)

    model.eval()
    with torch.no_grad():
        out1 = model(x, training=False)
        out2 = model(x, training=False)

    expected_shapes = {
        "logit_aff": (8,),
        "logit_spec": (8,),
        "z_aff": (8, 5),
        "z_spec": (8, 5),
        "aff_prob": (8,),
        "spec_prob": (8,),
    }
    for key, shape in expected_shapes.items():
        assert key in out1, f"missing output key: {key}"
        assert tuple(out1[key].shape) == shape, f"{key} shape {tuple(out1[key].shape)} != {shape}"
        assert torch.isfinite(out1[key]).all(), f"{key} contains non-finite values"

    assert torch.allclose(out1["logit_aff"], out2["logit_aff"]), "eval affinity logits are not deterministic"
    assert torch.allclose(out1["logit_spec"], out2["logit_spec"]), "eval specificity logits are not deterministic"

    model.train()
    out = model(x, training=True)
    loss = (
        focal_bce_with_logits(y_aff, out["logit_aff"])
        + focal_bce_with_logits(y_spec, out["logit_spec"])
        + 0.3 * ranking_loss(out["logit_aff"], y_aff)
        + 0.6 * ranking_loss(out["logit_spec"], y_spec)
        + 0.2 * gap_hinge_loss(out["logit_aff"], y_aff)
        + 0.6 * gap_hinge_loss(out["logit_spec"], y_spec)
    )
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert grads, "no gradients were produced"
    max_grad = max(float(g.abs().max().detach().cpu()) for g in grads)
    assert np.isfinite(max_grad), "gradient contains non-finite values"

    print("DiagnosticMOLM PyTorch validation passed")
    print(f"device={device}")
    print(f"loss={float(loss.detach().cpu()):.6f}")
    print(f"max_grad={max_grad:.6f}")


if __name__ == "__main__":
    main()

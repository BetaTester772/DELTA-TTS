"""RE-USE's Mamba layer: the fused Metal kernels (MPS) against the plain PyTorch path (CPU)."""

import pytest
import torch

from delta_tts.reuse import Mamba

requires_mps = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs Apple MPS")


@requires_mps
@pytest.mark.parametrize(("batch", "seq_len", "d_conv"), [(3, 50, 4), (2, 7, 3)])
def test_metal_matches_pytorch(batch: int, seq_len: int, d_conv: int) -> None:
    torch.manual_seed(0)
    layer = Mamba(d_model=32, d_state=16, d_conv=d_conv, expand=4).eval()
    with torch.no_grad():  # weights in the trained range: decays of various speeds, nonzero skips
        layer.A_log.copy_(torch.log(torch.rand_like(layer.A_log) * 8 + 0.5))
        layer.D.normal_()
        layer.dt_proj.bias.uniform_(-6, 3)
    x = torch.randn(batch, seq_len, 32)
    with torch.no_grad():
        expected = layer(x)
        actual = layer.to("mps")(x.to("mps")).cpu()
    assert torch.allclose(actual, expected, rtol=1e-4, atol=1e-4), (actual - expected).abs().max()

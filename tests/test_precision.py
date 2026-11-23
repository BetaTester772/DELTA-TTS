"""The fast mode's overflow fallback, on a stand-in model (no checkpoints needed)."""

import logging

import pytest
import torch
from torch import nn

from delta_tts.precision import ReducedPrecisionForward


class ScaledLogits(nn.Module):
    """Stand-in for DeltaTTS: logits = speech_token * scale, through a parameter so that the
    reduced-precision copy computes them in its own dtype."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1000.0))

    def logits(self, text_token: torch.Tensor, speech_token: torch.Tensor) -> torch.Tensor:
        return (speech_token.to(self.scale.dtype) * self.scale).unsqueeze(-1)


def test_no_overflow_uses_reduced_precision() -> None:
    forward = ReducedPrecisionForward(ScaledLogits(), torch.float16)
    out = forward.logits(torch.zeros(1, 1), torch.tensor([[1.0, 2.0]]))
    assert out.dtype == torch.float32
    assert torch.equal(out, torch.tensor([[[1000.0], [2000.0]]]))
    assert not forward.in_fallback and forward.num_fallbacks == 0


def test_overflow_falls_back_to_float32(caplog: pytest.LogCaptureFixture) -> None:
    forward = ReducedPrecisionForward(ScaledLogits(), torch.float16)
    big = torch.tensor([[100.0, 1.0]])  # 100 * 1000 > 65504: inf in float16
    with caplog.at_level(logging.WARNING):
        out = forward.logits(torch.zeros(1, 1), big)
    assert torch.equal(out, forward.exact.logits(torch.zeros(1, 1), big))  # the float32 result
    assert forward.in_fallback and forward.num_fallbacks == 1
    assert "overflow" in caplog.text

    # the rest of the utterance stays in float32, even for inputs that would fit
    out = forward.logits(torch.zeros(1, 1), torch.tensor([[1.0]]))
    assert out.dtype == torch.float32 and forward.num_fallbacks == 1

    # the next utterance tries float16 again
    forward.reset()
    assert not forward.in_fallback
    forward.logits(torch.zeros(1, 1), torch.tensor([[1.0]]))
    assert not forward.in_fallback


def test_bfloat16_does_not_overflow_there() -> None:
    forward = ReducedPrecisionForward(ScaledLogits(), torch.bfloat16)
    out = forward.logits(torch.zeros(1, 1), torch.tensor([[100.0]]))
    assert torch.isfinite(out).all() and not forward.in_fallback


def test_reduced_copy_is_independent() -> None:
    model = ScaledLogits()
    forward = ReducedPrecisionForward(model, torch.float16)
    assert forward.reduced is not model and forward.reduced.scale.dtype == torch.float16
    assert model.scale.dtype == torch.float32


def test_status() -> None:
    assert ReducedPrecisionForward(ScaledLogits(), torch.float16).status() == {"backend": "float16", "state": "ready"}

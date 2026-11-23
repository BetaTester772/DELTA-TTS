"""DeltaTTSPipeline.set_compute and backend_status, with stand-ins for the models (no checkpoints)."""

import pytest
import torch
from torch import nn

from delta_tts import pipeline
from delta_tts.pipeline import DeltaTTSPipeline
from delta_tts.precision import ReducedPrecisionForward


class StandInDeltaTTS(nn.Module):
    def __init__(self, backbone_dir: str) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))

    def load_checkpoint(self, path: str) -> None:
        pass

    def merge_lora(self) -> None:
        pass


class StandInCosyVoice3:
    sample_rate = 24000

    def __init__(self, backbone_dir: str) -> None:
        pass


@pytest.fixture
def tts(monkeypatch: pytest.MonkeyPatch) -> DeltaTTSPipeline:
    monkeypatch.setattr(pipeline, "DeltaTTS", StandInDeltaTTS)
    monkeypatch.setattr(pipeline, "CosyVoice3", StandInCosyVoice3)
    return DeltaTTSPipeline("backbone", "checkpoint.pt", device="cpu")


def test_switching_reuses_the_fast_backend(tts: DeltaTTSPipeline) -> None:
    assert tts.compute == "exact" and tts.forward is tts.model
    assert tts.backend_status() == {"backend": "float32", "state": "ready"}

    tts.set_compute("fast")  # on the CPU: float16 in PyTorch
    fast = tts.forward
    assert tts.compute == "fast" and isinstance(fast, ReducedPrecisionForward)
    assert tts.backend_status() == {"backend": "float16", "state": "ready"}

    tts.set_compute("exact")
    assert tts.forward is tts.model and tts.backend_status()["backend"] == "float32"
    tts.set_compute("fast")
    assert tts.forward is fast  # built once


def test_init_goes_through_set_compute(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline, "DeltaTTS", StandInDeltaTTS)
    monkeypatch.setattr(pipeline, "CosyVoice3", StandInCosyVoice3)
    tts = DeltaTTSPipeline("backbone", "checkpoint.pt", device="cpu", compute="fast")
    assert tts.compute == "fast" and isinstance(tts.forward, ReducedPrecisionForward)
    with pytest.raises(ValueError, match="compute"):
        tts.set_compute("quick")
    assert tts.compute == "fast"


def test_prewarm_builds_the_fast_backend_without_switching(tts: DeltaTTSPipeline) -> None:
    assert tts.backend_status("fast") == {"backend": None, "state": "not_started"}
    tts.prewarm_fast()
    assert tts.compute == "exact" and tts.forward is tts.model
    assert tts.backend_status() == {"backend": "float32", "state": "ready"}
    assert tts.backend_status("fast") == {"backend": "float16", "state": "ready"}
    prewarmed = tts._fast
    tts.set_compute("fast")
    assert tts.forward is prewarmed  # the switch reuses the prewarmed backend
    with pytest.raises(ValueError, match="compute"):
        tts.backend_status("quick")

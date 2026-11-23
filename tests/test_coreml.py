"""The Neural Engine backend without Core ML or checkpoints: buckets, packing, exact padding of
the export forward (on a tiny random Qwen2), and the fallback paths of CoreMLForward (with
stand-in Core ML functions)."""

import logging
import os

import numpy as np
import pytest
import torch
from torch import nn
from transformers import Qwen2Config, Qwen2ForCausalLM

from delta_tts import coreml
from delta_tts.coreml import CoreMLForward, pack, select_bucket
from delta_tts.model import HIDDEN_DIM, BidirectionalEncoder, ConvModule, _AfterBlock

# -----------------------------
# Buckets and packing
# -----------------------------


def test_select_bucket() -> None:
    buckets = [192, 256, 320]
    assert select_bucket(1, buckets) == 192
    assert select_bucket(192, buckets) == 192
    assert select_bucket(193, buckets) == 256
    assert select_bucket(320, buckets) == 320
    assert select_bucket(321, buckets) is None
    assert select_bucket(10, []) is None


def test_pack() -> None:
    embeddings = np.arange(3 * HIDDEN_DIM, dtype=np.float32).reshape(3, HIDDEN_DIM) / 1000
    x = pack(embeddings, 5)
    assert x.shape == (1, 5, HIDDEN_DIM + 1) and x.dtype == np.float16
    np.testing.assert_array_equal(x[0, :3, :HIDDEN_DIM], embeddings.astype(np.float16))
    np.testing.assert_array_equal(x[0, :, HIDDEN_DIM], [1, 1, 1, 0, 0])
    assert not x[0, 3:].any()


def test_prewarm_order_starts_near_typical_length() -> None:
    order = coreml.prewarm_order(coreml.BUCKETS)
    assert sorted(order) == sorted(coreml.BUCKETS)
    assert order[0] == min(coreml.BUCKETS, key=lambda b: abs(b - coreml.TYPICAL_SEQ_LEN))
    assert order[-1] == max(coreml.BUCKETS)


def test_read_buckets(tmp_path: str) -> None:
    model = os.path.join(tmp_path, "m.mlmodelc")
    os.makedirs(model)
    with open(os.path.join(model, "model.mil"), "w", encoding="utf-8") as f:
        f.write("program(1.3)\n{\n    func L320<ios18>(tensor<fp16, [1, 320, 897]> x) {\n    }\n"
                "    func L192<ios18>(tensor<fp16, [1, 192, 897]> x) {\n    }\n}\n")
    assert coreml.read_buckets(model) == [192, 320]
    assert coreml.default_path("/a/b/delta_tts.pt") == "/a/b/" + coreml.MODEL_NAME


# -----------------------------
# Exact padding of the export forward
# -----------------------------


class _PeftLike(nn.Module):
    """Stands in for the PEFT wrapper: `get_base_model()` returns the Qwen2 model."""

    def __init__(self, causal_lm: nn.Module) -> None:
        super().__init__()
        self.causal_lm = causal_lm

    def get_base_model(self) -> nn.Module:
        return self.causal_lm


class TinyDeltaTTS(nn.Module):
    """DeltaTTS's structure (bidirectional Qwen2 with a ConvModule after every block) at a small
    size, randomly initialized."""

    def __init__(self, hidden_dim: int = 64, vocab_size: int = 50) -> None:
        super().__init__()
        torch.manual_seed(0)
        config = Qwen2Config(hidden_size=hidden_dim, intermediate_size=2 * hidden_dim, num_hidden_layers=2,
                             num_attention_heads=4, num_key_value_heads=2, vocab_size=vocab_size)
        qwen2 = Qwen2ForCausalLM(config)
        self.bidirectional_encoder = BidirectionalEncoder(_PeftLike(qwen2))
        self.llm_decoder = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.conv_modules = nn.ModuleList([ConvModule(hidden_dim) for _ in qwen2.model.layers])
        for layer, conv in zip(qwen2.model.layers, self.conv_modules, strict=True):
            layer.register_forward_hook(_AfterBlock(conv))
        self.eval()

    def reference(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Unpadded logits, as DeltaTTS.logits computes them. (1, seq_len, vocab_size)"""
        return self.llm_decoder(self.bidirectional_encoder(embeddings))


@torch.no_grad()
def test_padding_does_not_change_the_logits() -> None:
    model = TinyDeltaTTS()
    seq_len, bucket = 37, 64
    embeddings = torch.randn(1, seq_len, 64)  # (1, seq_len, hidden_dim)
    reference = model.reference(embeddings)  # (1, seq_len, vocab_size)

    forward = coreml._ExportForward(model)  # replaces the depth-wise convolutions of `model`
    x = torch.zeros(1, bucket, 64 + 1)  # (1, bucket, hidden_dim + 1)
    x[0, :seq_len, :-1] = embeddings[0]
    x[0, :seq_len, -1] = 1
    padded = forward(x)[:, :seq_len]  # (1, seq_len, vocab_size)
    torch.testing.assert_close(padded, reference, rtol=0, atol=1e-5)

    # without the masks, the padding would change the logits
    x[0, :, -1] = 1
    assert (forward(x)[:, :seq_len] - reference).abs().max() > 1e-3


def test_masked_depthwise_conv_matches_conv1d() -> None:
    conv = nn.Conv1d(8, 8, 31, padding=15, groups=8)
    masked = coreml._MaskedDepthwiseConv(conv)
    x = torch.randn(1, 8, 20)  # (batch, channels, seq_len)
    padded = torch.cat([x, torch.randn(1, 8, 12)], dim=2)  # garbage in the padding
    masked.valid = torch.cat([torch.ones(1, 1, 20), torch.zeros(1, 1, 12)], dim=2)
    with torch.no_grad():
        torch.testing.assert_close(masked(padded)[:, :, :20], conv(x), rtol=0, atol=1e-6)


# -----------------------------
# CoreMLForward fallbacks, with stand-in Core ML functions
# -----------------------------


class StandInModel(nn.Module):
    """Stands in for the float32 DeltaTTS: embeddings of zeros, logits of ones."""

    def embed(self, text_token: torch.Tensor, speech_token: torch.Tensor) -> torch.Tensor:
        return torch.zeros(1, 2 + text_token.shape[1] + speech_token.shape[1], HIDDEN_DIM)

    def logits(self, text_token: torch.Tensor, speech_token: torch.Tensor) -> torch.Tensor:
        return torch.ones(1, 2 + text_token.shape[1] + speech_token.shape[1], 4)


class StandInFunction:
    """Stands in for a loaded Core ML function: returns `value` everywhere."""

    def __init__(self, value: float) -> None:
        self.value = value
        self.inputs: list[np.ndarray] = []

    def predict(self, inputs: dict) -> dict:
        self.inputs.append(inputs["x"])
        return {"logits": np.full((1, inputs["x"].shape[1], 4), self.value, dtype=np.float16)}


@pytest.fixture
def forward(tmp_path: str, monkeypatch: pytest.MonkeyPatch) -> CoreMLForward:
    """CoreMLForward on a fake model with buckets 8 and 16, whose loader loads nothing by itself."""
    monkeypatch.setattr(coreml._BucketLoader, "_run", lambda self: None)
    model = os.path.join(tmp_path, "m.mlmodelc")
    os.makedirs(model)
    with open(os.path.join(model, "model.mil"), "w", encoding="utf-8") as f:
        f.write("    func L8<ios18>(x) {\n    func L16<ios18>(x) {\n")
    return CoreMLForward(StandInModel(), model, prewarm=False)  # type: ignore[arg-type]


def tokens(text_len: int, speech_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Text and speech tokens for a sequence of 2 + text_len + speech_len positions."""
    return torch.zeros(1, text_len, dtype=torch.long), torch.zeros(1, speech_len, dtype=torch.long)


def test_ready_bucket_runs_on_the_neural_engine(forward: CoreMLForward) -> None:
    function = forward.loader.ready[16] = StandInFunction(2.0)
    out = forward.logits(*tokens(3, 6))  # 11 positions -> bucket 16
    assert out.dtype == torch.float32 and out.shape == (1, 11, 4)
    assert torch.equal(out, torch.full((1, 11, 4), 2.0))
    x = function.inputs[0]
    assert x.shape == (1, 16, HIDDEN_DIM + 1) and x[0, :, -1].sum() == 11
    assert forward.num_steps == {"fast": 1, "float32": 0}


def test_unready_bucket_runs_the_utterance_in_float32(forward: CoreMLForward) -> None:
    forward.loader._queue = [8]
    out = forward.logits(*tokens(3, 6))  # bucket 16, not loaded
    assert torch.equal(out, torch.ones(1, 11, 4))
    assert forward.loader._queue == [16, 8]  # requested first

    # still float32 once the bucket is ready, until the next utterance
    forward.loader.ready[16] = StandInFunction(2.0)
    assert torch.equal(forward.logits(*tokens(3, 6)), torch.ones(1, 11, 4))
    forward.reset()
    assert torch.equal(forward.logits(*tokens(3, 6)), torch.full((1, 11, 4), 2.0))
    assert forward.num_steps == {"fast": 1, "float32": 2} and forward.num_fallbacks == 0


def test_too_long_runs_in_float32(forward: CoreMLForward) -> None:
    forward.loader.ready[16] = StandInFunction(2.0)
    assert torch.equal(forward.logits(*tokens(10, 10)), torch.ones(1, 22, 4))
    assert forward.loader._queue == [] and forward.num_fallbacks == 0


def test_overflow_falls_back_to_float32(forward: CoreMLForward, caplog: pytest.LogCaptureFixture) -> None:
    forward.loader.ready[8] = StandInFunction(float("inf"))
    with caplog.at_level(logging.WARNING):
        out = forward.logits(*tokens(2, 3))
    assert torch.equal(out, torch.ones(1, 7, 4))
    assert forward.in_fallback and forward.num_fallbacks == 1 and "overflow" in caplog.text

    forward.loader.ready[8] = StandInFunction(2.0)  # sticky for the rest of the utterance
    assert torch.equal(forward.logits(*tokens(2, 3)), torch.ones(1, 7, 4))
    forward.reset()
    assert torch.equal(forward.logits(*tokens(2, 3)), torch.full((1, 7, 4), 2.0))
    assert forward.num_fallbacks == 1


def test_wait_times_out_when_nothing_loads(forward: CoreMLForward) -> None:
    assert not forward.wait([8], timeout=0.05)
    forward.loader.ready[8] = StandInFunction(2.0)
    assert forward.wait([8], timeout=0.05)


# -----------------------------
# Background loading, with stand-in loads
# -----------------------------


def stand_in_loads(monkeypatch: pytest.MonkeyPatch, fail: dict[int, int]) -> list[int]:
    """Replace the helper and in-process loads: bucket b fails its first fail[b] helper loads.
    Returns the helper loads, in order."""
    loads: list[int] = []

    def load_in_helper(self: coreml._BucketLoader, bucket: int) -> float:
        loads.append(bucket)
        if loads.count(bucket) <= fail.get(bucket, 0):
            raise RuntimeError("compiled program deleted")
        return 0.0

    monkeypatch.setattr(coreml._BucketLoader, "_load_in_helper", load_in_helper)
    monkeypatch.setattr(coreml._BucketLoader, "_load", lambda self, bucket: f"function {bucket}")
    monkeypatch.setattr(coreml._BucketLoader, "_stop_helper", lambda self: None)
    return loads


def test_failed_bucket_is_tried_again_after_the_others(monkeypatch: pytest.MonkeyPatch) -> None:
    loads = stand_in_loads(monkeypatch, fail={16: 1})
    loader = coreml._BucketLoader("unused", [16, 8])
    assert loader.wait([16, 8], timeout=5)
    assert loads == [16, 8, 16] and loader.ready == {16: "function 16", 8: "function 8"}


def test_bucket_failing_twice_runs_in_float32(monkeypatch: pytest.MonkeyPatch) -> None:
    loads = stand_in_loads(monkeypatch, fail={8: 2})
    loader = coreml._BucketLoader("unused", [8, 16])
    assert not loader.wait([8, 16], timeout=5)
    assert loads == [8, 16, 8] and loader.failed == {8} and loader.get(8) is None
    assert loader._queue == []  # a failed bucket is not queued again

"""The exported Neural Engine model against the float32 PyTorch model. Needs the checkpoints, an
exported Core ML model (DELTA_TTS_COREML_MODEL, or the default path next to the checkpoint),
macOS and coremltools; skipped otherwise."""

import os

import pytest
import torch

from delta_tts import coreml
from delta_tts.model import MASK_TOKEN, DeltaTTS

BACKBONE_DIR = "pretrained_models/Fun-CosyVoice3-0.5B-2512"
CHECKPOINT = "pretrained_models/DELTA-TTS/delta_tts.pt"
COREML_MODEL = os.environ.get("DELTA_TTS_COREML_MODEL", coreml.default_path(CHECKPOINT))

pytestmark = pytest.mark.skipif(
    not (coreml.is_available() and os.path.isfile(CHECKPOINT) and os.path.isdir(BACKBONE_DIR) and os.path.isdir(COREML_MODEL)),
    reason="needs macOS, coremltools, the checkpoints and an exported Core ML model",
)


@torch.no_grad()
def test_neural_engine_logits_match_float32() -> None:
    model = DeltaTTS(BACKBONE_DIR).float().eval()
    model.load_checkpoint(CHECKPOINT)
    model.merge_lora()
    forward = coreml.CoreMLForward(model, COREML_MODEL, prewarm=False)

    generator = torch.Generator().manual_seed(0)
    text = torch.randint(0, 5000, (1, 40), generator=generator)  # (1, text_len)
    speech = torch.randint(0, 6561, (1, 200), generator=generator)  # (1, speech_seq_len)
    speech[:, 120:199] = MASK_TOKEN
    seq_len = 1 + 40 + 1 + 200
    bucket = coreml.select_bucket(seq_len, forward.buckets)
    assert bucket is not None and forward.wait([bucket], timeout=600)

    reference = model.logits(text, speech)  # (1, seq_len, vocab_size)
    forward.reset()
    logits = forward.logits(text, speech)  # (1, seq_len, vocab_size)
    assert forward.num_steps == {"fast": 1, "float32": 0}
    assert logits.shape == reference.shape and torch.isfinite(logits).all()
    agreement = (logits.argmax(-1) == reference.argmax(-1)).float().mean().item()
    assert agreement > 0.9, f"argmax agreement {agreement:.3f}"

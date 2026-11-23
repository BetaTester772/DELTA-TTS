"""Progress reporting of `generate`, on a stand-in model (no checkpoints needed)."""

import torch

from delta_tts.generate import generate, unmask_schedule
from delta_tts.model import SPEECH_TOKEN_SIZE


class PositionalLogits:
    """Stand-in for DeltaTTS: deterministic logits that favor token (position * 7) % SPEECH_TOKEN_SIZE."""

    def logits(self, text_token: torch.Tensor, speech_token: torch.Tensor) -> torch.Tensor:
        seq_len = 1 + text_token.shape[1] + 1 + speech_token.shape[1]
        logits = torch.zeros(1, seq_len, SPEECH_TOKEN_SIZE + 200)
        positions = torch.arange(seq_len)
        logits[0, positions, (positions * 7) % SPEECH_TOKEN_SIZE] = 10.0 + positions.float() / seq_len
        return logits


def _run(progress=None, speech_len: int = 40, num_steps: int = 16) -> torch.Tensor:
    text = torch.zeros(1, 5, dtype=torch.long)
    prompt = torch.arange(10).unsqueeze(0)
    return generate(PositionalLogits(), text, prompt, speech_len, num_steps=num_steps, progress=progress)


def test_progress_counts_forward_passes() -> None:
    calls: list[tuple[int, int]] = []
    _run(progress=lambda step, total: calls.append((step, total)))
    total = sum(1 for n in unmask_schedule(40, 16, 0.3) if n > 0)
    assert calls == [(step, total) for step in range(1, total + 1)]


def test_progress_does_not_change_the_output() -> None:
    assert torch.equal(_run(), _run(progress=lambda step, total: None))


def test_short_utterance_has_one_pass_per_token() -> None:
    calls: list[tuple[int, int]] = []
    _run(progress=lambda step, total: calls.append((step, total)), speech_len=5)
    assert calls[-1] == (5, 5) and len(calls) == 5

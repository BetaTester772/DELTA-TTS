"""Approximate DELTA-TTS forwards with a float32 fallback (the `fast` compute mode).

What this file contains
    FallbackForward          the fallback logic shared by the fast forwards
    ReducedPrecisionForward  the forward in float16 (or bfloat16) with PyTorch
The Neural Engine forward (delta_tts/coreml.py) is the other `FallbackForward`.

float16 represents magnitudes only up to 65504. An activation beyond that becomes inf and
turns the logits into inf/NaN. Every forward is therefore checked: a step whose logits are not
finite is recomputed by the float32 model, and the rest of that utterance stays in float32
(similar inputs would overflow again). The next utterance tries the fast forward again.

The fast outputs differ numerically from the float32 (`exact`) ones; whether that matters is
measured, not assumed (see the README).
"""

import copy
import logging

import torch

from delta_tts.model import DeltaTTS

__all__ = ["FallbackForward", "ReducedPrecisionForward"]

logger = logging.getLogger(__name__)


class FallbackForward:
    """`DeltaTTS.logits` by a fast, approximate forward, with the float32 model as the fallback.

    Subclasses implement `_fast_logits`, which may return None for a step it cannot serve; that
    step then runs in float32. Non-finite fast logits are recomputed in float32, and the rest of
    the utterance stays in float32 until `reset`.

    Args:
        model: The float32 DELTA-TTS model, kept as the fallback.
    """

    name = "fast"  # names the fast forward in the overflow warning

    def __init__(self, model: DeltaTTS) -> None:
        self.exact = model
        self.in_fallback = False  # set on overflow, cleared by `reset` for the next utterance
        self.num_fallbacks = 0  # utterances that overflowed
        self.num_steps = {"fast": 0, "float32": 0}  # forwards served by each path

    def reset(self) -> None:
        """Start a new utterance with the fast forward."""
        self.in_fallback = False

    def status(self) -> dict:
        """What runs the forward, for display: {"backend": "float16" (or "bfloat16"), "state": "ready"}."""
        return {"backend": self.name, "state": "ready"}

    def _fast_logits(self, text_token: torch.Tensor, speech_token: torch.Tensor) -> torch.Tensor | None:
        """Same contract as `DeltaTTS.logits`, or None to run this step in float32."""
        raise NotImplementedError

    @torch.no_grad()
    def logits(self, text_token: torch.Tensor, speech_token: torch.Tensor) -> torch.Tensor:
        """Same contract as `DeltaTTS.logits`; float32 logits. Shape: (1, seq_len, vocab_size)"""
        if not self.in_fallback:
            logits = self._fast_logits(text_token, speech_token)
            if logits is not None:
                if bool(torch.isfinite(logits).all()):
                    self.num_steps["fast"] += 1
                    return logits
                self.in_fallback = True
                self.num_fallbacks += 1
                logger.warning("%s overflow in the DELTA-TTS forward; finishing this utterance in float32", self.name)
        self.num_steps["float32"] += 1
        return self.exact.logits(text_token, speech_token)


class ReducedPrecisionForward(FallbackForward):
    """`DeltaTTS.logits` in float16 or bfloat16, with a float32 fallback on overflow.

    Args:
        model: The float32 DELTA-TTS model, kept as the fallback.
        dtype: torch.float16 or torch.bfloat16.
    """

    def __init__(self, model: DeltaTTS, dtype: torch.dtype = torch.float16) -> None:
        super().__init__(model)
        self.dtype = dtype
        self.name = str(dtype).removeprefix("torch.")
        self.reduced = copy.deepcopy(model).to(dtype)

    def _fast_logits(self, text_token: torch.Tensor, speech_token: torch.Tensor) -> torch.Tensor:
        return self.reduced.logits(text_token, speech_token).float()  # (1, seq_len, vocab_size)

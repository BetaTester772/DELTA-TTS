"""Reduced-precision DELTA-TTS forward with a float32 fallback (the `fast` compute mode).

float16 represents magnitudes only up to 65504. An activation beyond that becomes inf and
turns the logits into inf/NaN. Every forward is therefore checked: a step whose logits are not
finite is recomputed by the float32 model, and the rest of that utterance stays in float32
(similar inputs would overflow again). The next utterance tries the reduced precision again.

The reduced-precision outputs differ numerically from the float32 (`exact`) ones; whether that
matters is measured, not assumed (see the README).
"""

import copy
import logging

import torch

from delta_tts.model import DeltaTTS

__all__ = ["ReducedPrecisionForward"]

logger = logging.getLogger(__name__)


class ReducedPrecisionForward:
    """`DeltaTTS.logits` in float16 or bfloat16, with a float32 fallback on overflow.

    Args:
        model: The float32 DELTA-TTS model, kept as the fallback.
        dtype: torch.float16 or torch.bfloat16.
    """

    def __init__(self, model: DeltaTTS, dtype: torch.dtype = torch.float16) -> None:
        self.exact = model
        self.dtype = dtype
        self.reduced = copy.deepcopy(model).to(dtype)
        self.in_fallback = False  # set on overflow, cleared by `reset` for the next utterance
        self.num_fallbacks = 0  # utterances that fell back

    def reset(self) -> None:
        """Start a new utterance in reduced precision."""
        self.in_fallback = False

    @torch.no_grad()
    def logits(self, text_token: torch.Tensor, speech_token: torch.Tensor) -> torch.Tensor:
        """Same contract as `DeltaTTS.logits`; float32 logits. Shape: (1, seq_len, vocab_size)"""
        if not self.in_fallback:
            logits = self.reduced.logits(text_token, speech_token).float()
            if bool(torch.isfinite(logits).all()):
                return logits
            self.in_fallback = True
            self.num_fallbacks += 1
            logger.warning("%s overflow in the DELTA-TTS forward; finishing this utterance in float32",
                           str(self.dtype).removeprefix("torch."))
        return self.exact.logits(text_token, speech_token)

"""Zero-shot TTS pipeline: text and prompt audio in, waveform out.

DELTA-TTS replaces only the autoregressive LLM of CosyVoice 3. Everything around it is the
original CosyVoice 3 code, loaded from the submodule in third_party/CosyVoice.

One call of `DeltaTTSPipeline.synthesize`
    1. CosyVoice 3 frontend   text tokens, prompt speech tokens, prompt mel, speaker embedding
                              (prompt features are cached, so a repeated prompt is processed once)
    2. Output length          estimated from the prompt (delta_tts/duration.py)
    3. DELTA-TTS              speech tokens by iterative unmasking (delta_tts/generate.py)
    4. CosyVoice 3 decoder    flow matching + vocoder, conditioned on the prompt
With an `AudioProcessor`, the prompt is enhanced/trimmed before step 1 and the output trimmed
after step 4 (delta_tts/audio.py).
"""

import logging
import os
import sys
import uuid
import warnings
from collections import OrderedDict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torchaudio

from delta_tts.duration import estimate_speech_len
from delta_tts.generate import generate
from delta_tts.model import DeltaTTS
from delta_tts.precision import FallbackForward, ReducedPrecisionForward

COSYVOICE_DIR = Path(__file__).resolve().parent.parent / "third_party" / "CosyVoice"
sys.path.insert(0, str(COSYVOICE_DIR))
sys.path.insert(0, str(COSYVOICE_DIR / "third_party" / "Matcha-TTS"))
from cosyvoice.cli.cosyvoice import CosyVoice3

if TYPE_CHECKING:
    from delta_tts.audio import AudioProcessor

__all__ = ["DeltaTTSPipeline", "default_device"]

logger = logging.getLogger(__name__)

# Token ids of "You are a helpful assistant.<|endofprompt|>", the instruction CosyVoice 3 is
# trained with. It is placed before the prompt transcript, as in CosyVoice 3.
INSTRUCT_TOKENS = [2610, 525, 264, 10950, 17847, 13, 151646]

PROMPT_CACHE_SIZE = 16  # prompts whose features are kept

# progress(stage, step, total), see `DeltaTTSPipeline.synthesize`
Progress = Callable[[str, int, int], None]
ACOUSTIC_STEPS = 10  # Euler steps of the CosyVoice 3 flow decoder (n_timesteps in its flow.inference)


def default_device() -> str:
    """CUDA if available, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class DeltaTTSPipeline:
    """Text and prompt audio in, waveform out.

    Args:
        backbone_dir: CosyVoice 3 model directory (Fun-CosyVoice3-0.5B-2512).
        checkpoint: Path to the DELTA-TTS checkpoint `delta_tts.pt`.
        device: Torch device for DELTA-TTS and the CosyVoice 3 decoder. `default_device()` when None.
        audio_processor: Optional `delta_tts.audio.AudioProcessor` (Silero VAD, RE-USE) applied
            to the prompt before and to the output after synthesis.
        compute: "exact" runs DELTA-TTS in float32. "fast" runs it approximately, with a float32
            fallback on overflow; its outputs differ numerically. On MPS, with an exported Core ML
            model and coremltools, it runs on the Neural Engine (delta_tts/coreml.py); otherwise
            in `fast_dtype` with PyTorch (delta_tts/precision.py).
        fast_dtype: Reduced precision of the PyTorch fast mode.
        coreml_model: Compiled Core ML model for the fast mode on MPS. By default the one that
            `python -m delta_tts.coreml export` writes next to `checkpoint`, if it exists.

    `set_compute` switches the mode later; `backend_status` tells what runs DELTA-TTS.
    """

    def __init__(self, backbone_dir: str, checkpoint: str, device: str | None = None,
                 audio_processor: "AudioProcessor | None" = None, compute: str = "exact",
                 fast_dtype: torch.dtype = torch.float16, coreml_model: str | None = None) -> None:
        _check_compute(compute)
        self.device = torch.device(device or default_device())
        self.model = DeltaTTS(backbone_dir).to(self.device).float().eval()
        self.model.load_checkpoint(checkpoint)
        self.model.merge_lora()
        self._fast_options = (checkpoint, coreml_model, fast_dtype)
        self._fast: FallbackForward | None = None  # built on the first switch to "fast", then kept
        self.forward: DeltaTTS | FallbackForward = self.model
        self._compute = "exact"
        self.set_compute(compute)
        with warnings.catch_warnings():
            # The CosyVoice config imports pyworld, which imports pkg_resources; setuptools 80
            # (used on macOS) warns about that import.
            warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)
            self.cosyvoice = CosyVoice3(backbone_dir)
        self.sample_rate = self.cosyvoice.sample_rate  # 24 kHz
        self.audio_processor = audio_processor
        self._prompt_cache: OrderedDict[tuple, dict] = OrderedDict()
        if self.device.type == "mps":
            _move_decoder_to_mps(self.cosyvoice.model, self.device)

    @torch.no_grad()
    def synthesize(self, text: str, prompt_wav: str, prompt_text: str, speech_len: int | None = None,
                   length_scale: float = 1.0, progress: Progress | None = None, **decoding) -> torch.Tensor:
        """Synthesize `text` in the voice of `prompt_wav`.

        Args:
            text: Text to synthesize.
            prompt_wav: Path to the prompt audio.
            prompt_text: Transcript of the prompt audio.
            speech_len: Output length in speech tokens (25 per second). Estimated from the
                prompt when None.
            length_scale: Factor on the estimated length (above 1: slower speech). Ignored when
                `speech_len` is given.
            progress: Called as progress(stage, step, total) as synthesis advances:
                "prompt" (0, 1) and (1, 1) when the prompt is processed (not when it is cached),
                "tokens" after each DELTA-TTS forward pass, "acoustic" after each flow-matching
                step and "vocoder" (0, 1) and (1, 1). The device is synchronized at each step, so
                the steps follow the actual computation; the output does not change.
            **decoding: Overrides for `delta_tts.generate.generate` (num_steps, top_p, ...).
        Returns:
            Waveform at `self.sample_rate`. Shape: (1, num_samples)
        """
        frontend = self.cosyvoice.frontend

        # -----------------------------
        # 1. CosyVoice 3 frontend
        # -----------------------------
        report = progress or _no_progress
        prompt = self._prompt(prompt_wav, prompt_text, report)
        target_text_token, _ = frontend._extract_text_token(frontend.text_normalize(text, split=False, text_frontend=True))
        # Text sequence: [instruction] [prompt transcript] [target text]
        target_text_token = target_text_token.to(self.device)  # (1, target_text_len)
        prompt_text_token = prompt["text_token"].to(self.device)  # (1, prompt_text_len)
        instruct = torch.tensor([INSTRUCT_TOKENS], dtype=target_text_token.dtype, device=self.device)  # (1, 7)
        text_token = torch.cat([instruct, prompt_text_token, target_text_token], dim=1)  # (1, text_len)
        prompt_speech_token = prompt["speech_token"].to(self.device)  # (1, prompt_len)

        # -----------------------------
        # 2. Output length
        # -----------------------------
        if speech_len is None:
            # The length rule uses the untrimmed prompt token count (see `_extract_prompt`).
            speech_len = estimate_speech_len(prompt_text, text, prompt["untrimmed_speech_len"])
            speech_len = max(1, round(speech_len * length_scale))

        # -----------------------------
        # 3. DELTA-TTS: speech tokens by iterative unmasking
        # -----------------------------
        if isinstance(self.forward, FallbackForward):
            self.forward.reset()
        speech_token = generate(self.forward, text_token, prompt_speech_token, speech_len,
                                progress=(lambda step, total: report("tokens", step, total)) if progress else None,
                                **decoding)  # (1, speech_len)

        # -----------------------------
        # 4. CosyVoice 3 flow decoder and vocoder
        # -----------------------------
        if progress is None:
            wav = self._token2wav(speech_token, prompt)
        else:
            with self._reporting_decoder(report):
                wav = self._token2wav(speech_token, prompt)
        if self.audio_processor is not None:
            wav = self.audio_processor.process_output(wav, self.sample_rate)
        return wav

    def prepare_prompt(self, prompt_wav: str, prompt_text: str) -> tuple[torch.Tensor, int]:
        """Process the prompt now (the `AudioProcessor`'s enhancement and trimming, if any, and
        the CosyVoice 3 frontend), so that a later `synthesize` with the same prompt file and
        transcript skips this work.

        Returns:
            The prompt audio the model uses, i.e. after processing, and its sample rate.
            Shape: (1, num_samples)
        """
        return self._prompt(prompt_wav, prompt_text)["audio"]

    def clear_prompt_cache(self) -> None:
        """Forget all cached prompts (features, speaker embeddings and processed audio)."""
        self._prompt_cache.clear()

    @property
    def compute(self) -> str:
        """The compute mode, "exact" or "fast" (see `set_compute`)."""
        return self._compute

    def set_compute(self, mode: str) -> None:
        """Switch DELTA-TTS between "exact" (float32) and "fast".

        The fast backend is built on the first switch to "fast" and kept for later switches. The
        switch returns at once: the Neural Engine buckets load in the background, and until its
        bucket is ready an utterance runs in float32 (`backend_status` shows the progress). Not
        safe during a `synthesize` call: the caller serializes the two.
        """
        _check_compute(mode)
        if mode == "fast" and self._fast is None:
            self._fast = self._fast_forward(*self._fast_options)
        self.forward = self._fast if mode == "fast" and self._fast is not None else self.model
        self._compute = mode

    def backend_status(self) -> dict:
        """What runs DELTA-TTS now, for display; a snapshot that never waits for a compilation.

        Returns:
            {"backend": "float32", "state": "ready"} in the exact mode; in the fast mode
            `ReducedPrecisionForward.status()` ({"backend": "float16", "state": "ready"}) or
            `CoreMLForward.status()` (backend "neural_engine", with its loading progress).
        """
        if isinstance(self.forward, FallbackForward):
            return self.forward.status()
        return {"backend": "float32", "state": "ready"}

    def _fast_forward(self, checkpoint: str, coreml_model: str | None, fast_dtype: torch.dtype) -> FallbackForward:
        """The fast mode: the Neural Engine on MPS when an exported Core ML model and coremltools
        are available, otherwise reduced precision in PyTorch."""
        if self.device.type == "mps":
            from delta_tts import coreml

            path = coreml_model or coreml.default_path(checkpoint)
            if coreml_model is not None and not os.path.isdir(coreml_model):
                raise FileNotFoundError(f"Core ML model not found: {coreml_model}")
            if os.path.isdir(path) and coreml.is_available():
                logger.info("Fast mode: DELTA-TTS on the Neural Engine (Core ML float16, %s)", path)
                return coreml.CoreMLForward(self.model, path)
            logger.info("Fast mode: no Neural Engine (%s)", "coremltools is not installed" if os.path.isdir(path)
                        else f"no Core ML model at {path}; see `python -m delta_tts.coreml export`")
        logger.info("Fast mode: DELTA-TTS in %s on %s", str(fast_dtype).removeprefix("torch."), self.device)
        return ReducedPrecisionForward(self.model, fast_dtype)

    def _prompt(self, prompt_wav: str, prompt_text: str, report: Progress | None = None) -> dict:
        """Prompt features, from the cache when the same prompt file and transcript were seen.

        With an `AudioProcessor`, the features are those of the processed prompt.
        """
        stat = os.stat(prompt_wav)
        key = (os.path.abspath(prompt_wav), stat.st_mtime_ns, stat.st_size, prompt_text)
        if key in self._prompt_cache:
            self._prompt_cache.move_to_end(key)
            return self._prompt_cache[key]
        report = report or _no_progress
        report("prompt", 0, 1)
        if self.audio_processor is None:
            prompt = self._extract_prompt(prompt_wav, prompt_text)
            prompt["audio"] = _load_mono(prompt_wav)
        else:
            with self.audio_processor.prepare_prompt(prompt_wav) as processed_prompt_wav:
                prompt = self._extract_prompt(processed_prompt_wav, prompt_text)
                prompt["audio"] = _load_mono(processed_prompt_wav)
        report("prompt", 1, 1)
        self._prompt_cache[key] = prompt
        if len(self._prompt_cache) > PROMPT_CACHE_SIZE:
            self._prompt_cache.popitem(last=False)
        return prompt

    def _extract_prompt(self, prompt_wav: str, prompt_text: str) -> dict:
        """The prompt half of CosyVoice 3 `CosyVoiceFrontEnd.frontend_zero_shot`, plus the
        untrimmed speech-token count, so that the speech tokenizer runs once per prompt.

        Returns:
            text_token: Prompt transcript tokens. Shape: (1, prompt_text_len)
            speech_token: Prompt speech tokens, trimmed to half the mel length. Shape: (1, prompt_len)
            speech_feat: Prompt mel, two frames per speech token. Shape: (1, 2 * prompt_len, 80)
            embedding: Speaker embedding. Shape: (1, speaker_dim)
            untrimmed_speech_len: Speech-token count before trimming.
        """
        frontend = self.cosyvoice.frontend
        text_token, _ = frontend._extract_text_token(frontend.text_normalize(prompt_text, split=False, text_frontend=True))
        speech_feat, _ = frontend._extract_speech_feat(prompt_wav)
        speech_token, _ = frontend._extract_speech_token(prompt_wav)
        untrimmed_speech_len = speech_token.shape[1]
        if self.sample_rate == 24000:
            # As in frontend_zero_shot: force two mel frames per speech token.
            token_len = min(speech_feat.shape[1] // 2, speech_token.shape[1])
            speech_feat, speech_token = speech_feat[:, :2 * token_len], speech_token[:, :token_len]
        embedding = frontend._extract_spk_embedding(prompt_wav)
        return {"text_token": text_token, "speech_token": speech_token, "speech_feat": speech_feat,
                "embedding": embedding.reshape(1, -1), "untrimmed_speech_len": untrimmed_speech_len}

    @contextmanager
    def _reporting_decoder(self, report: Progress) -> Iterator[None]:
        """Report the flow-matching steps and the vocoder while `_token2wav` runs.

        Wraps the flow decoder's per-step estimator call and the vocoder for the duration of the
        block. Each step synchronizes the device: otherwise the queued GPU work would let the
        reported steps run ahead of the computation.
        """
        model = self.cosyvoice.model
        decoder = model.flow.decoder
        estimate, vocode = decoder.forward_estimator, model.hift.inference
        steps = 0

        def estimate_and_report(*args, **kwargs):
            nonlocal steps
            out = estimate(*args, **kwargs)
            _synchronize(self.device)
            steps += 1
            report("acoustic", min(steps, ACOUSTIC_STEPS), ACOUSTIC_STEPS)
            return out

        def vocode_and_report(*args, **kwargs):
            report("vocoder", 0, 1)
            out = vocode(*args, **kwargs)
            report("vocoder", 1, 1)
            return out

        decoder.forward_estimator = estimate_and_report
        model.hift.inference = vocode_and_report
        try:
            yield
        finally:
            del decoder.forward_estimator  # back to the class method
            del model.hift.inference

    def _token2wav(self, speech_token: torch.Tensor, prompt: dict) -> torch.Tensor:
        """Decode speech tokens to a waveform with the prompt as acoustic reference.

        Args:
            speech_token: Generated speech tokens. Shape: (1, speech_len)
            prompt: Prompt features from `_extract_prompt`.
        Returns:
            Waveform. Shape: (1, num_samples)
        """
        model = self.cosyvoice.model
        uid = str(uuid.uuid1())
        model.hift_cache_dict[uid] = None
        wav = model.token2wav(
            token=speech_token.to(self.device, dtype=torch.int32),
            prompt_token=prompt["speech_token"].to(self.device, dtype=torch.int32),
            prompt_feat=prompt["speech_feat"].to(self.device),
            embedding=prompt["embedding"].to(self.device),
            token_offset=0,
            uuid=uid,
            finalize=True,
        )  # (1, num_samples)
        model.hift_cache_dict.pop(uid, None)
        return wav.cpu()


class _HiFTFloat32:
    """The CosyVoice 3 vocoder with its F0 predictor in float32, so that all of it runs on MPS.

    `CausalHiFTGenerator.inference` casts the F0 predictor to float64, which MPS does not
    support. float64 keeps the F0 of consecutive chunks consistent in streaming synthesis; this
    pipeline vocodes each utterance in one call (finalize=True), where float32 gives the same F0
    (at most 0.004 Hz apart and no voiced/unvoiced flips on 16 demo utterances). Otherwise the
    same steps as `CausalHiFTGenerator.inference`.
    """

    def __init__(self, hift: torch.nn.Module) -> None:
        self.hift = hift

    @torch.inference_mode()
    def inference(self, speech_feat: torch.Tensor, finalize: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        hift = self.hift
        f0 = hift.f0_predictor(speech_feat, finalize=finalize)  # mel -> F0
        source = hift.f0_upsamp(f0[:, None]).transpose(1, 2)  # F0 -> source
        source, _, _ = hift.m_source(source)
        source = source.transpose(1, 2)
        mel = speech_feat if finalize else speech_feat[:, :, :-hift.f0_predictor.condnet[0].causal_padding]
        return hift.decode(x=mel, s=source, finalize=finalize), source


def _check_compute(mode: str) -> None:
    if mode not in ("exact", "fast"):
        raise ValueError(f"compute must be 'exact' or 'fast', got {mode!r}")


def _no_progress(stage: str, step: int, total: int) -> None:
    pass


def _synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def _load_mono(path: str) -> tuple[torch.Tensor, int]:
    """Audio file as a mono float waveform. Shape: (1, num_samples)"""
    wav, sample_rate = torchaudio.load(path)
    return wav.mean(dim=0, keepdim=True), sample_rate


def _move_decoder_to_mps(model, device: torch.device) -> None:
    """Place the CosyVoice 3 flow decoder and vocoder on MPS; CosyVoice itself only knows CUDA
    and CPU.

    Args:
        model: `CosyVoice3Model`, loaded on the CPU.
        device: MPS device.
    """
    model.device = device
    model.flow.to(device)
    hift = model.hift.to(device)
    # The vocoder's STFT and inverse STFT (n_fft = 16) are tiny and run on the CPU, since MPS
    # complex support is partial; the results are the same.
    stft, istft = type(hift)._stft, type(hift)._istft
    hift._stft = lambda x: tuple(y.to(x.device) for y in stft(hift, x.cpu()))
    hift._istft = lambda magnitude, phase: istft(hift, magnitude.cpu(), phase.cpu()).to(magnitude.device)
    model.hift = _HiFTFloat32(hift)

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
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from delta_tts.duration import estimate_speech_len
from delta_tts.generate import generate
from delta_tts.model import DeltaTTS

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
    """

    def __init__(self, backbone_dir: str, checkpoint: str, device: str | None = None,
                 audio_processor: "AudioProcessor | None" = None) -> None:
        self.device = torch.device(device or default_device())
        self.model = DeltaTTS(backbone_dir).to(self.device).float().eval()
        self.model.load_checkpoint(checkpoint)
        self.model.merge_lora()
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
    def synthesize(self, text: str, prompt_wav: str, prompt_text: str, speech_len: int | None = None, **decoding) -> torch.Tensor:
        """Synthesize `text` in the voice of `prompt_wav`.

        Args:
            text: Text to synthesize.
            prompt_wav: Path to the prompt audio.
            prompt_text: Transcript of the prompt audio.
            speech_len: Output length in speech tokens (25 per second). Estimated from the
                prompt when None.
            **decoding: Overrides for `delta_tts.generate.generate` (num_steps, top_p, ...).
        Returns:
            Waveform at `self.sample_rate`. Shape: (1, num_samples)
        """
        frontend = self.cosyvoice.frontend

        # -----------------------------
        # 1. CosyVoice 3 frontend
        # -----------------------------
        prompt = self._prompt(prompt_wav, prompt_text)
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

        # -----------------------------
        # 3. DELTA-TTS: speech tokens by iterative unmasking
        # -----------------------------
        speech_token = generate(self.model, text_token, prompt_speech_token, speech_len, **decoding)  # (1, speech_len)

        # -----------------------------
        # 4. CosyVoice 3 flow decoder and vocoder
        # -----------------------------
        wav = self._token2wav(speech_token, prompt)
        if self.audio_processor is not None:
            wav = self.audio_processor.process_output(wav, self.sample_rate)
        return wav

    def _prompt(self, prompt_wav: str, prompt_text: str) -> dict:
        """Prompt features, from the cache when the same prompt file and transcript were seen.

        With an `AudioProcessor`, the features are those of the processed prompt.
        """
        stat = os.stat(prompt_wav)
        key = (os.path.abspath(prompt_wav), stat.st_mtime_ns, stat.st_size, prompt_text)
        if key in self._prompt_cache:
            self._prompt_cache.move_to_end(key)
            return self._prompt_cache[key]
        if self.audio_processor is None:
            prompt = self._extract_prompt(prompt_wav, prompt_text)
        else:
            with self.audio_processor.prepare_prompt(prompt_wav) as processed_prompt_wav:
                prompt = self._extract_prompt(processed_prompt_wav, prompt_text)
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

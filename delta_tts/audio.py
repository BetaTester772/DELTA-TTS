"""Audio processing around DELTA-TTS: RE-USE speech enhancement and Silero VAD.

Requires the `audio` extra (`uv sync --extra audio`).

What is applied, each step optional
    Prompt, before synthesis
        1. RE-USE        removes noise, reverberation, clipping, codec artifacts, ... (delta_tts/reuse.py)
        2. Silero VAD    trims non-speech at the start and end
    Output, after synthesis
        3. Silero VAD    trims non-speech at the start and end

Only leading and trailing non-speech is removed, so the prompt transcript stays valid.

Standalone, on one file:
    python -m delta_tts.audio input.wav output.wav --enhance --trim
"""

import argparse
import logging
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager

import torch
import torchaudio

from delta_tts.reuse import REPO_ID, ReUSE

__all__ = ["AudioProcessor"]

logger = logging.getLogger(__name__)

VAD_SAMPLE_RATE = 16000  # Silero VAD runs at 8 or 16 kHz


class AudioProcessor:
    """Prompt and output processing with RE-USE and Silero VAD.

    Args:
        enhance_prompt: Enhance the prompt with RE-USE.
        trim_prompt: Trim leading and trailing non-speech of the prompt with Silero VAD.
        trim_output: Trim leading and trailing non-speech of the output with Silero VAD.
        reuse_dir: RE-USE directory (`config.json`, `model.safetensors`) or Hugging Face repo id.
        device: Torch device for RE-USE. Silero VAD runs on the CPU.
        pad_ms: Audio kept before the first and after the last speech segment, so that soft
            onsets and word endings (e.g. a final fricative weakened by enhancement) are not cut.
    """

    def __init__(self, enhance_prompt: bool = False, trim_prompt: bool = False, trim_output: bool = False,
                 reuse_dir: str = REPO_ID, device: str = "cpu", pad_ms: int = 200) -> None:
        self.trim_prompt = trim_prompt
        self.trim_output = trim_output
        self.pad_ms = pad_ms
        self.enhancer = ReUSE(reuse_dir, device) if enhance_prompt else None
        self.vad = None
        if trim_prompt or trim_output:
            try:
                from silero_vad import get_speech_timestamps, load_silero_vad
            except ImportError as e:
                raise ImportError("Silero VAD is part of the `audio` extra: uv sync --extra audio") from e
            self.vad = load_silero_vad()
            self._get_speech_timestamps = get_speech_timestamps

    @contextmanager
    def prepare_prompt(self, prompt_wav: str) -> Iterator[str]:
        """Yield the path of the processed prompt (a temporary file, removed afterwards).

        The CosyVoice 3 frontend reads the prompt from a path, so the processed audio is written
        to a temporary WAV file at the prompt's sample rate.
        """
        if self.enhancer is None and not self.trim_prompt:
            yield prompt_wav
            return
        wav, sample_rate = torchaudio.load(prompt_wav)
        wav = self.process(wav.mean(dim=0, keepdim=True), sample_rate, enhance=self.enhancer is not None, trim=self.trim_prompt)
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "prompt.wav")
            torchaudio.save(path, wav, sample_rate)
            yield path

    def process_output(self, wav: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Trim the generated waveform if `trim_output`. Shape: (1, num_samples)"""
        return self.trim(wav, sample_rate) if self.trim_output else wav

    def process(self, wav: torch.Tensor, sample_rate: int, enhance: bool = True, trim: bool = True) -> torch.Tensor:
        """Enhance and then trim a mono waveform.

        Args:
            wav: Waveform. Shape: (1, num_samples)
            sample_rate: Sample rate of `wav`.
            enhance: Apply RE-USE (requires `enhance_prompt=True` at construction).
            trim: Apply Silero VAD trimming.
        Returns:
            Processed waveform at `sample_rate`. Shape: (1, num_samples')
        """
        if enhance:
            if self.enhancer is None:
                raise ValueError("RE-USE is not loaded; construct with enhance_prompt=True")
            wav = self.enhancer(wav, sample_rate)
        if trim:
            wav = self.trim(wav, sample_rate)
        return wav

    def trim(self, wav: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Remove non-speech before the first and after the last speech segment.

        Args:
            wav: Waveform. Shape: (1, num_samples)
            sample_rate: Sample rate of `wav`.
        Returns:
            Trimmed waveform, or `wav` unchanged when no speech is detected. Shape: (1, num_samples')
        """
        wav_16k = wav if sample_rate == VAD_SAMPLE_RATE else torchaudio.functional.resample(wav, sample_rate, VAD_SAMPLE_RATE)
        segments = self._get_speech_timestamps(wav_16k[0].float(), self.vad, sampling_rate=VAD_SAMPLE_RATE, speech_pad_ms=self.pad_ms)
        if not segments:
            logger.warning("Silero VAD found no speech; the audio is left untrimmed")
            return wav
        start = segments[0]["start"] * sample_rate // VAD_SAMPLE_RATE
        end = min(wav.shape[1], -(-segments[-1]["end"] * sample_rate // VAD_SAMPLE_RATE))
        return wav[:, start:end]


def main() -> None:
    parser = argparse.ArgumentParser(description="Enhance (RE-USE) and/or trim (Silero VAD) one audio file.")
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--enhance", action="store_true", help="RE-USE speech enhancement")
    parser.add_argument("--trim", action="store_true", help="Silero VAD trimming of leading/trailing non-speech")
    parser.add_argument("--reuse_dir", default=REPO_ID)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from delta_tts.pipeline import default_device

    processor = AudioProcessor(enhance_prompt=args.enhance, trim_prompt=args.trim, reuse_dir=args.reuse_dir,
                               device=args.device or default_device())
    wav, sample_rate = torchaudio.load(args.input)
    wav = processor.process(wav.mean(dim=0, keepdim=True), sample_rate, enhance=args.enhance, trim=args.trim)
    torchaudio.save(args.output, wav, sample_rate)
    logger.info("Saved %s", args.output)


if __name__ == "__main__":
    main()

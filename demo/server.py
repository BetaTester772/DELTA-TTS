"""Local HTTP server for the DELTA-TTS demo app (demo/DeltaTTSDemo).

The model is loaded once, in a background thread, and requests are synthesized one at a time.
Run from the repository root:
    uv run python demo/server.py [--config demo/config.json] [--dry-run]

Endpoints (JSON)
    GET  /health       status ("loading", "ready" or "error"), device, compute, sample_rate, dry_run
    GET  /config       the config, with preset paths resolved
    POST /synthesize   text and prompt in; base64 WAV, timing and warnings out

Every notice in "warnings" is {level: "info" | "warning" | "error", code, message}. A request
that cannot be served gets an HTTP 4xx/5xx status and {error, code, warnings}.

--dry-run imports no model: /synthesize returns a tone as long as the estimated output, after
the same checks, so that the app can be developed without the checkpoints or a GPU.

Security: the server is for local use by the demo app and must not be exposed to a network.
It binds to 127.0.0.1, accepts only Host headers of the loopback addresses (DNS rebinding),
rejects requests with an Origin header and non-JSON POSTs (cross-site requests from a browser),
reads prompt paths only inside the repository, the preset directories and its upload directory,
and bounds request bodies and the uploaded prompts it keeps.
"""

from __future__ import annotations

import argparse
import array
import base64
import binascii
import copy
import hashlib
import importlib.util
import io
import json
import logging
import math
import os
import shutil
import signal
import sys
import tempfile
import threading
import time
import wave
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import FrameType, ModuleType
from typing import Any, Self

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "demo" / "config.json"

SPEECH_TOKEN_RATE = 25  # CosyVoice 3 speech tokens per second
MAX_PROMPT_SECONDS = 30  # the CosyVoice 3 speech tokenizer rejects longer prompts
MIN_PROMPT_SAMPLE_RATE = 16000  # CosyVoice 3 does not upsample prompts from below 16 kHz
COMPUTE_MODES = ("exact", "fast")
PROCESSING_KEYS = ("enhance_prompt", "trim_prompt", "trim_output")
LOCAL_HOSTS = ("127.0.0.1", "localhost", "::1")

# Limits on what a request can make the server read and keep
MAX_BODY_BYTES = 25 * 2**20
MAX_UPLOAD_BYTES = 18 * 2**20  # decoded prompt audio
MAX_UPLOAD_SECONDS = 60
MIN_UPLOAD_SAMPLE_RATE = 8000
MAX_UPLOAD_FILES = 32  # uploaded prompts kept, least recently used dropped first
MAX_UPLOAD_DIR_BYTES = 200 * 2**20
AUDIO_ERRORS = (OSError, RuntimeError, EOFError, ValueError, ArithmeticError, wave.Error)  # soundfile raises RuntimeErrors

DRY_RUN_SAMPLE_RATE = 24000
DRY_RUN_LOAD_SECONDS = 1.0  # simulated model loading, so that clients see the "loading" state
DRY_RUN_SYNTHESIS_SECONDS = 0.5  # simulated synthesis time

logger = logging.getLogger("demo.server")

Notice = dict[str, str]


def notice(level: str, code: str, message: str) -> Notice:
    """One entry of the "warnings" list. level: "info", "warning" or "error"."""
    return {"level": level, "code": code, "message": message}


class RequestError(Exception):
    """A request that cannot be served, reported as an "error" notice with an HTTP status."""

    def __init__(self, status: HTTPStatus, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


def bad_request(code: str, message: str) -> RequestError:
    return RequestError(HTTPStatus.BAD_REQUEST, code, message)


class LogCollector(logging.Handler):
    """Collects the warnings that the `delta_tts` library logs from the current thread."""

    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.thread = threading.get_ident()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        if record.thread == self.thread:
            self.records.append(record)

    def __enter__(self) -> Self:
        logging.getLogger("delta_tts").addHandler(self)
        return self

    def __exit__(self, *exc_info: object) -> None:
        logging.getLogger("delta_tts").removeHandler(self)


def _load_duration_rule() -> ModuleType:
    """delta_tts/duration.py, loaded without the package `__init__`, which imports torch and
    CosyVoice 3."""
    path = REPO_ROOT / "delta_tts" / "duration.py"
    spec = importlib.util.spec_from_file_location("_delta_tts_duration", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


duration_rule = _load_duration_rule()


# -----------------------------
# Config and audio helpers
# -----------------------------
def load_config(path: Path) -> dict[str, Any]:
    """Read the demo config (demo/config.json)."""
    config = json.loads(path.read_text(encoding="utf-8"))
    missing = [key for key in ("server", "presets", "synthesis", "processing", "warnings") if key not in config]
    if missing:
        raise SystemExit(f"{path}: missing sections {', '.join(missing)}")
    return config


def resolve(path: str) -> Path:
    """A config or request path, relative to the repository root unless absolute."""
    resolved = Path(path).expanduser()
    return resolved if resolved.is_absolute() else REPO_ROOT / resolved


def audio_info(path: Path) -> tuple[float, int]:
    """Duration in seconds and sample rate of an audio file, read with soundfile as CosyVoice 3
    reads the prompt (the standard-library `wave` module when soundfile is missing)."""
    try:
        import soundfile
    except ImportError:
        with wave.open(str(path)) as f:
            return f.getnframes() / f.getframerate(), f.getframerate()
    info = soundfile.info(str(path))
    return info.frames / info.samplerate, int(info.samplerate)


class UploadStore:
    """Uploaded prompts in a private temporary directory, removed by `close`.

    Each file is named by its content hash, so that a repeated upload maps to the same path and
    hits the pipeline's prompt cache (keyed by path, mtime and size). The store keeps at most
    `max_files` files and `max_bytes` bytes and drops the least recently used first. Recency is
    tracked in memory rather than by touching mtimes, which would invalidate the prompt cache.
    """

    def __init__(self, max_files: int = MAX_UPLOAD_FILES, max_bytes: int = MAX_UPLOAD_DIR_BYTES) -> None:
        self.dir = Path(tempfile.mkdtemp(prefix="delta_tts_demo_uploads_")).resolve()  # mode 0700
        self.max_files = max_files
        self.max_bytes = max_bytes
        self._files: OrderedDict[Path, int] = OrderedDict()  # path -> size, least recently used first

    def add(self, audio: bytes, suffix: str) -> Path:
        """Store `audio` after checking that it is audio of a sane length and sample rate.

        Raises:
            RequestError: The audio is unreadable, too long or sampled too low.
        """
        target = self.dir / f"{hashlib.sha256(audio).hexdigest()[:32]}{suffix}"
        if target in self._files and target.is_file():
            self._files.move_to_end(target)
            return target
        fd, incoming = tempfile.mkstemp(dir=self.dir, prefix=".incoming_", suffix=suffix)
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(audio)
            self._validate(Path(incoming))
            os.replace(incoming, target)
        finally:
            Path(incoming).unlink(missing_ok=True)
        self._files[target] = len(audio)
        self._files.move_to_end(target)
        self._evict()
        return target

    def close(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    @staticmethod
    def _validate(path: Path) -> None:
        try:
            seconds, sample_rate = audio_info(path)
        except AUDIO_ERRORS as e:
            logger.info("Rejected an uploaded prompt: %s: %s", type(e).__name__, e)
            raise bad_request("prompt_unreadable", "The uploaded prompt is not readable audio; use a WAV or FLAC file.") from None
        if seconds > MAX_UPLOAD_SECONDS:
            raise bad_request("prompt_long", f"The uploaded prompt is {seconds:.1f} s long; at most {MAX_UPLOAD_SECONDS} s are accepted.")
        if sample_rate < MIN_UPLOAD_SAMPLE_RATE:
            raise bad_request("prompt_low_sample_rate",
                              f"The uploaded prompt is sampled at {sample_rate} Hz; at least {MIN_UPLOAD_SAMPLE_RATE} Hz are accepted.")

    def _evict(self) -> None:
        while len(self._files) > 1 and (len(self._files) > self.max_files or sum(self._files.values()) > self.max_bytes):
            path, _ = self._files.popitem(last=False)
            path.unlink(missing_ok=True)


def estimate_output_seconds(text: str, prompt_text: str, prompt_seconds: float | None, length_scale: float) -> float:
    """Output duration by the length rule of delta_tts/duration.py, as the pipeline applies it.

    The pipeline counts the prompt's speech tokens; here they are approximated from its duration.
    """
    if prompt_seconds is None:
        prompt_text, prompt_len = "", 0  # the rule's fallback speaking rate
    else:
        prompt_len = round(prompt_seconds * SPEECH_TOKEN_RATE)
    speech_len = duration_rule.estimate_speech_len(prompt_text, text, prompt_len)
    return max(1, round(speech_len * length_scale)) / SPEECH_TOKEN_RATE


def wav_bytes(pcm16: bytes, sample_rate: int) -> bytes:
    """Mono little-endian 16-bit PCM samples as a WAV file."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(pcm16)
    return buffer.getvalue()


def dry_run_tone(seconds: float, sample_rate: int = DRY_RUN_SAMPLE_RATE) -> bytes:
    """A soft 220 Hz tone with a syllable-like 4 Hz envelope and 20 ms fades, as 16-bit PCM."""
    n = max(1, round(seconds * sample_rate))
    fade = max(1, min(n // 2, sample_rate // 50))
    samples = array.array("h")
    for i in range(n):
        t = i / sample_rate
        gain = min(1.0, i / fade, (n - 1 - i) / fade) * (0.6 + 0.4 * math.sin(2 * math.pi * 4 * t))
        samples.append(round(0.2 * 32767 * gain * math.sin(2 * math.pi * 220 * t)))
    if sys.byteorder == "big":
        samples.byteswap()
    return samples.tobytes()


# -----------------------------
# Requests
# -----------------------------
@dataclass
class SynthesisRequest:
    """A validated /synthesize request; unset fields take the config defaults."""

    text: str
    prompt_text: str
    prompt_wav: Path
    length_scale: float
    decoding: dict[str, Any]  # num_steps, temperature, top_p, t_shift
    compute: str
    processing: tuple[bool, bool, bool]  # enhance_prompt, trim_prompt, trim_output
    thresholds: dict[str, float]


def parse_request(body: dict[str, Any], config: dict[str, Any], uploads: UploadStore,
                  prompt_roots: list[Path]) -> SynthesisRequest:
    """Validate the request body and fill in the defaults of the config."""
    synthesis, processing = config["synthesis"], config["processing"]
    overrides = body.get("warnings") or {}
    if not isinstance(overrides, dict):
        raise bad_request("invalid_parameter", "warnings must be an object of thresholds")

    def get(key: str, defaults: dict[str, Any], source: dict[str, Any] = body) -> Any:
        """`source[key]`, or the default when it is missing or null."""
        value = source.get(key)
        return defaults[key] if value is None else value

    def number(key: str, value: Any, valid: Callable[[float], bool], requirement: str) -> float:
        if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
            raise bad_request("invalid_parameter", f"{key} must be a number, got {value!r}")
        if not valid(value):
            raise bad_request("invalid_parameter", f"{key} must be {requirement}, got {value!r}")
        return float(value)

    def string(key: str) -> str:
        value = get(key, {key: ""})
        if not isinstance(value, str):
            raise bad_request("invalid_parameter", f"{key} must be a string")
        return value

    num_steps = number("num_steps", get("num_steps", synthesis), lambda v: 1 <= v <= 256 and v == int(v),
                       "an integer in [1, 256]")
    compute = get("compute", synthesis)
    if compute not in COMPUTE_MODES:
        raise bad_request("invalid_parameter", f"compute must be 'exact' or 'fast', got {compute!r}")
    flags = []
    for key in PROCESSING_KEYS:
        value = get(key, processing)
        if not isinstance(value, bool):
            raise bad_request("invalid_parameter", f"{key} must be true or false, got {value!r}")
        flags.append(value)
    thresholds = {key: number(key, get(key, config["warnings"], overrides), lambda v: v >= 0, "at least 0")
                  for key in config["warnings"]}

    return SynthesisRequest(
        text=string("text"),
        prompt_text=string("prompt_text"),
        prompt_wav=resolve_prompt(body, uploads, prompt_roots),
        length_scale=number("length_scale", get("length_scale", synthesis), lambda v: 0 < v <= 10, "in (0, 10]"),
        decoding={
            "num_steps": int(num_steps),
            "temperature": number("temperature", get("temperature", synthesis), lambda v: v >= 0, "at least 0"),
            "top_p": number("top_p", get("top_p", synthesis), lambda v: 0 < v <= 1, "in (0, 1]"),
            "t_shift": number("t_shift", get("t_shift", synthesis), lambda v: v > 0, "above 0"),
        },
        compute=compute,
        processing=(flags[0], flags[1], flags[2]),
        thresholds=thresholds,
    )


def resolve_prompt(body: dict[str, Any], uploads: UploadStore, prompt_roots: list[Path]) -> Path:
    """The prompt audio file: an upload (`prompt_wav_base64`) or a path (`prompt_wav_path`) that
    resolves, symlinks included, inside one of `prompt_roots`. Responses never name the
    resolved path; the details are logged."""
    data, path = body.get("prompt_wav_base64"), body.get("prompt_wav_path")
    if data:
        if not isinstance(data, str):
            raise bad_request("invalid_parameter", "prompt_wav_base64 must be a string")
        if len(data) > (MAX_UPLOAD_BYTES + 2) // 3 * 4:
            raise RequestError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "too_large",
                               f"The prompt audio is larger than {MAX_UPLOAD_BYTES // 2**20} MB.")
        try:
            audio = base64.b64decode(data, validate=True)
        except (binascii.Error, ValueError):
            raise bad_request("prompt_unreadable", "prompt_wav_base64 is not valid base64.") from None
        suffix = Path(str(body.get("prompt_wav_name") or "prompt.wav")).suffix.lower()
        if not (2 <= len(suffix) <= 6 and suffix[1:].isalnum()):
            suffix = ".wav"
        return uploads.add(audio, suffix)
    if path:
        if not isinstance(path, str):
            raise bad_request("invalid_parameter", "prompt_wav_path must be a string")
        resolved = resolve(path).resolve()
        if not any(resolved.is_relative_to(root) for root in prompt_roots):
            logger.warning("Rejected prompt_wav_path %r: %s is outside %s", path, resolved, [str(r) for r in prompt_roots])
            raise RequestError(HTTPStatus.FORBIDDEN, "prompt_not_allowed", "Prompt audio not allowed.")
        if not resolved.is_file():
            logger.warning("prompt_wav_path %r: %s is not a file", path, resolved)
            raise bad_request("prompt_not_found", "Prompt audio not found.")
        return resolved
    raise bad_request("prompt_missing", "No prompt audio: give prompt_wav_path or prompt_wav_base64.")


def check_request(request: SynthesisRequest, notices: list[Notice]) -> float:
    """Checks before synthesis. Problems that would fail synthesis raise `RequestError`; the
    others are appended to `notices`.

    Returns:
        The estimated output duration in seconds.
    """
    limits = request.thresholds
    if not request.text.strip():
        raise bad_request("text_empty", "The text to synthesize is empty.")
    if len(request.text) > limits["max_text_chars"]:
        notices.append(notice("warning", "text_long",
                              f"The text has {len(request.text)} characters (more than {limits['max_text_chars']:g}). "
                              "DELTA-TTS generates the whole utterance at once; consider splitting long text into sentences."))
    if not request.prompt_text.strip():
        notices.append(notice("warning", "transcript_missing",
                              "No prompt transcript. DELTA-TTS expects the words spoken in the prompt, and the output "
                              "length falls back to a default speaking rate; expect lower quality."))

    enhance, trim_prompt, trim_output = request.processing
    needed = (["safetensors"] if enhance else []) + (["silero_vad"] if trim_prompt or trim_output else [])
    missing = [module for module in needed if importlib.util.find_spec(module) is None]
    if missing:
        raise bad_request("audio_extra_missing",
                          f"Prompt/output processing needs the `audio` extra ({', '.join(missing)} not installed): "
                          "uv sync --extra audio (on macOS: uv sync --extra mac --extra audio).")

    try:
        prompt_seconds, sample_rate = audio_info(request.prompt_wav)
    except AUDIO_ERRORS as e:
        logger.info("Cannot read the prompt %s: %s: %s", request.prompt_wav, type(e).__name__, e)
        raise bad_request("prompt_unreadable", "Cannot read the prompt audio; use a WAV or FLAC file.") from None
    if sample_rate < MIN_PROMPT_SAMPLE_RATE:
        raise bad_request("prompt_low_sample_rate",
                          f"The prompt is sampled at {sample_rate} Hz; CosyVoice 3 needs at least {MIN_PROMPT_SAMPLE_RATE} Hz.")
    if sample_rate < limits["min_prompt_sample_rate"]:
        notices.append(notice("warning", "prompt_low_sample_rate",
                              f"The prompt is sampled at {sample_rate} Hz (below {limits['min_prompt_sample_rate']:g} Hz); "
                              "the cloned voice may sound muffled."))
    if prompt_seconds > MAX_PROMPT_SECONDS and not trim_prompt:
        raise bad_request("prompt_long",
                          f"The prompt is {prompt_seconds:.1f} s long; the CosyVoice 3 speech tokenizer accepts at most "
                          f"{MAX_PROMPT_SECONDS} s. Use a shorter clip.")
    if prompt_seconds < limits["min_prompt_seconds"]:
        notices.append(notice("warning", "prompt_short",
                              f"The prompt is {prompt_seconds:.1f} s long (under {limits['min_prompt_seconds']:g} s); "
                              "a short prompt gives a less accurate voice and speaking rate."))
    elif prompt_seconds > limits["max_prompt_seconds"]:
        notices.append(notice("warning", "prompt_long",
                              f"The prompt is {prompt_seconds:.1f} s long (over {limits['max_prompt_seconds']:g} s); "
                              "a long prompt slows synthesis, and CosyVoice 3 rejects prompts over "
                              f"{MAX_PROMPT_SECONDS} s after trimming."))

    estimate = estimate_output_seconds(request.text, request.prompt_text, prompt_seconds, request.length_scale)
    if estimate > limits["max_output_seconds"]:
        notices.append(notice("warning", "output_long",
                              f"The estimated output is {estimate:.1f} s long (over {limits['max_output_seconds']:g} s); "
                              "long outputs take longer and may lose quality. Consider splitting the text."))
    return estimate


# -----------------------------
# Model
# -----------------------------
class TTSService:
    """The loaded pipeline (or the dry run) and the state of its runtime settings.

    Args:
        config: The demo config.
        dry_run: Serve a synthetic tone instead of loading DELTA-TTS.
    """

    def __init__(self, config: dict[str, Any], dry_run: bool) -> None:
        self.config = config
        self.dry_run = dry_run
        self.status = "loading"  # "loading", "ready" or "error"
        self.error: str | None = None
        self.device: str | None = None
        self.compute: str = config["server"].get("compute", "exact")
        self.processing: tuple[bool, bool, bool] = (False, False, False)
        self.sample_rate = DRY_RUN_SAMPLE_RATE
        self.tts: Any = None  # DeltaTTSPipeline
        self._fast_forward: Any = None  # ReducedPrecisionForward, created on first use
        self._processors: dict[tuple[bool, bool, bool], Any] = {}  # AudioProcessor per setting
        self._lock = threading.Lock()  # one synthesis at a time
        self.uploads = UploadStore()
        # Directories that prompt_wav_path may point into
        preset_dirs = {resolve(preset["wav"]).resolve().parent for preset in config["presets"]}
        self.prompt_roots = [REPO_ROOT, self.uploads.dir, *sorted(preset_dirs - {REPO_ROOT})]

    def start_loading(self) -> None:
        threading.Thread(target=self._load, name="model-loader", daemon=True).start()

    def close(self) -> None:
        self.uploads.close()

    def _load(self) -> None:
        try:
            if self.compute not in COMPUTE_MODES:
                raise ValueError(f"server.compute must be 'exact' or 'fast', got {self.compute!r}")
            if self.dry_run:
                time.sleep(DRY_RUN_LOAD_SECONDS)
            else:
                self._load_model()
        except Exception as e:  # reported through /health
            logger.exception("Loading the model failed")
            self.error = f"{type(e).__name__}: {e}"
            self.status = "error"
            return
        self.status = "ready"
        logger.info("Ready: %s", "dry run" if self.dry_run else f"{self.device}, {self.compute}")

    def _load_model(self) -> None:
        sys.path.insert(0, str(REPO_ROOT))
        from delta_tts import DeltaTTSPipeline

        server = self.config["server"]
        logger.info("Loading DELTA-TTS (%s)", self.compute)
        self.tts = DeltaTTSPipeline(str(resolve(server["backbone_dir"])), str(resolve(server["checkpoint"])),
                                    device=server.get("device"), compute=self.compute)
        self.device = str(self.tts.device)
        self.sample_rate = self.tts.sample_rate
        if self.compute == "fast":
            self._fast_forward = self.tts.forward

    def health(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "error": self.error,
            "device": self.device,
            "compute": self.compute,
            "processing": dict(zip(PROCESSING_KEYS, self.processing, strict=True)),
            "sample_rate": self.sample_rate,
            "dry_run": self.dry_run,
            "busy": self._lock.locked(),
        }

    def public_config(self) -> dict[str, Any]:
        """The config with each preset's resolved `wav_path` and whether it exists."""
        config = copy.deepcopy(self.config)
        for preset in config["presets"]:
            path = resolve(preset["wav"])
            preset["wav_path"], preset["exists"] = str(path), path.is_file()
        config["repo_root"] = str(REPO_ROOT)
        return config

    def synthesize(self, body: dict[str, Any], notices: list[Notice]) -> dict[str, Any]:
        """Serve one /synthesize request. Notices are appended to `notices` as they arise."""
        if self.status == "loading":
            raise RequestError(HTTPStatus.SERVICE_UNAVAILABLE, "server_loading",
                               "The model is still loading; try again in a moment.")
        if self.status == "error":
            raise RequestError(HTTPStatus.SERVICE_UNAVAILABLE, "server_failed", f"The model failed to load: {self.error}")
        with self._lock:
            request = parse_request(body, self.config, self.uploads, self.prompt_roots)
            estimate = check_request(request, notices)
            if self.dry_run:
                pcm, elapsed, fell_back = self._synthesize_dry_run(request, estimate, notices)
            else:
                pcm, elapsed, fell_back = self._synthesize_model(request, notices)
        duration = len(pcm) / 2 / self.sample_rate
        return {
            "audio_wav_base64": base64.b64encode(wav_bytes(pcm, self.sample_rate)).decode("ascii"),
            "sample_rate": self.sample_rate,
            "duration_s": round(duration, 3),
            "estimated_duration_s": round(estimate, 3),
            "elapsed_s": round(elapsed, 3),
            "rtf": round(elapsed / duration, 4) if duration > 0 else None,
            "compute": request.compute,
            "fell_back": fell_back,
            "dry_run": self.dry_run,
            "warnings": notices,
        }

    def _synthesize_dry_run(self, request: SynthesisRequest, estimate: float,
                            notices: list[Notice]) -> tuple[bytes, float, bool]:
        start = time.perf_counter()
        time.sleep(DRY_RUN_SYNTHESIS_SECONDS)
        pcm = dry_run_tone(estimate, self.sample_rate)
        elapsed = time.perf_counter() - start
        self.compute, self.processing = request.compute, request.processing
        notices.append(notice("info", "dry_run", "Dry run: the output is a tone as long as the estimated speech."))
        if any(request.processing):
            notices.append(notice("info", "dry_run_processing", "Dry run: prompt and output processing are not applied."))
        return pcm, elapsed, False

    def _synthesize_model(self, request: SynthesisRequest, notices: list[Notice]) -> tuple[bytes, float, bool]:
        import torch

        logs = LogCollector()
        try:
            with logs:
                self._set_compute(request.compute)
                self._set_processing(request.processing)
                start = time.perf_counter()
                wav = self.tts.synthesize(request.text, str(request.prompt_wav), request.prompt_text,
                                          length_scale=request.length_scale, **request.decoding)  # (1, num_samples)
                elapsed = time.perf_counter() - start
        finally:  # the library's warnings are reported also when synthesis fails
            fell_back = bool(getattr(self.tts.forward, "in_fallback", False))
            for record in logs.records:
                if fell_back and record.name == "delta_tts.precision":
                    continue  # reported below
                level = "error" if record.levelno >= logging.ERROR else "warning"
                notices.append(notice(level, "library", record.getMessage()))
        if fell_back:
            notices.append(notice("warning", "fast_fallback",
                                  "float16 overflowed in this utterance, so it was finished in float32; "
                                  "the output is valid but synthesis was slower."))
        pcm = wav[0].clamp(-1, 1).mul(32767).round().to(torch.int16).numpy().astype("<i2").tobytes()
        return pcm, elapsed, fell_back

    def _set_compute(self, compute: str) -> None:
        """Switch between float32 (exact) and float16 with fallback (fast)."""
        if compute == self.compute:
            return
        if compute == "fast":
            if self._fast_forward is None:
                import torch

                from delta_tts.precision import ReducedPrecisionForward

                logger.info("Creating the float16 copy of DELTA-TTS for the fast mode")
                self._fast_forward = ReducedPrecisionForward(self.tts.model, torch.float16)
            self.tts.forward = self._fast_forward
        else:
            self.tts.forward = self.tts.model
        self.compute = compute

    def _set_processing(self, processing: tuple[bool, bool, bool]) -> None:
        """Set the pipeline's audio processor (Silero VAD, RE-USE) for these flags."""
        if processing == self.processing:
            return
        if any(processing):
            if processing not in self._processors:
                from delta_tts.audio import AudioProcessor

                enhance, trim_prompt, trim_output = processing
                logger.info("Loading audio processing: %s", dict(zip(PROCESSING_KEYS, processing, strict=True)))
                self._processors[processing] = AudioProcessor(
                    enhance_prompt=enhance, trim_prompt=trim_prompt, trim_output=trim_output,
                    reuse_dir=self.config["server"].get("reuse_dir", "nvidia/NVIDIA-NemotronLabs-RE-USE"),
                    device=self.device or "cpu")
            self.tts.audio_processor = self._processors[processing]
        else:
            self.tts.audio_processor = None
        if processing[:2] != self.processing[:2]:
            # Cached prompt features were computed with the previous prompt processing.
            self.tts._prompt_cache.clear()
        self.processing = processing


# -----------------------------
# HTTP
# -----------------------------
class Handler(BaseHTTPRequestHandler):
    """JSON endpoints over `TTSService`."""

    # set by `main`
    service: TTSService
    allowed_hosts: frozenset[str] = frozenset()  # Host headers: the loopback addresses with the port
    allowed_origins: frozenset[str] = frozenset()  # Origin headers: none by default (the app sends none)
    server_version = "DeltaTTSDemo/1.0"

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        try:
            self._check_client()
            if path == "/health":
                payload = self.service.health()
            elif path == "/config":
                payload = self.service.public_config()
            else:
                raise RequestError(HTTPStatus.NOT_FOUND, "not_found", "No such endpoint")
        except RequestError as e:
            self._send_error(e, [])
        else:
            self._send(HTTPStatus.OK, payload)

    def do_OPTIONS(self) -> None:
        # No CORS: a browser's preflight for a cross-origin JSON POST fails here.
        self._send_error(RequestError(HTTPStatus.METHOD_NOT_ALLOWED, "method_not_allowed", "OPTIONS is not supported"), [])

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0]
        notices: list[Notice] = []
        try:
            self._check_client()
            if path != "/synthesize":
                raise RequestError(HTTPStatus.NOT_FOUND, "not_found", "No such endpoint")
            result = self.service.synthesize(self._read_json(), notices)
        except RequestError as e:
            self._send_error(e, notices)
        except Exception as e:
            logger.exception("Synthesis failed")
            error = RequestError(HTTPStatus.INTERNAL_SERVER_ERROR, "synthesis_failed",
                                 f"Synthesis failed: {type(e).__name__}: {e}")
            self._send_error(error, notices)
        else:
            self._send(HTTPStatus.OK, result)

    def _check_client(self) -> None:
        """Reject requests that a web page could send: another Host (DNS rebinding) or an Origin
        outside the allowlist (cross-site requests)."""
        host = (self.headers.get("Host") or "").lower()
        if host not in self.allowed_hosts:
            logger.warning("Rejected a request with Host %r", host)
            raise RequestError(HTTPStatus.FORBIDDEN, "forbidden", "Host not allowed")
        origin = self.headers.get("Origin")
        if origin is not None and origin not in self.allowed_origins:
            logger.warning("Rejected a request with Origin %r", origin)
            raise RequestError(HTTPStatus.FORBIDDEN, "forbidden", "Cross-origin requests are not allowed")

    def _read_json(self) -> dict[str, Any]:
        content_type = (self.headers.get("Content-Type") or "").split(";", 1)[0].strip().lower()
        if content_type != "application/json":
            raise RequestError(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "unsupported_media_type",
                               "Content-Type must be application/json")
        if self.headers.get("Content-Length") is None:
            raise RequestError(HTTPStatus.LENGTH_REQUIRED, "length_required", "Content-Length is required")
        try:
            length = int(self.headers["Content-Length"])
        except ValueError:
            raise bad_request("bad_request", "Invalid Content-Length") from None
        if not 0 <= length <= MAX_BODY_BYTES:
            raise RequestError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "too_large",
                               f"The request is larger than {MAX_BODY_BYTES // 2**20} MB")
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise bad_request("bad_request", f"The request body is not valid JSON: {e}") from None
        if not isinstance(body, dict):
            raise bad_request("bad_request", "The request body must be a JSON object")
        return body

    def _send_error(self, error: RequestError, notices: list[Notice]) -> None:
        notices.append(notice("error", error.code, error.message))
        self._send(error.status, {"error": error.message, "code": error.code, "warnings": notices})

    def _send(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        data = json.dumps(payload).encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            logger.info("The client closed the connection before the response")

    def log_message(self, format: str, *args: Any) -> None:
        level = logging.DEBUG if getattr(self, "path", "").startswith("/health") else logging.INFO
        logger.log(level, "%s %s", self.address_string(), format % args)


def _ignore_stop_signals() -> None:
    """Ignore further SIGTERM/SIGINT, so that a second signal cannot interrupt the cleanup:
    `uv run` forwards the signals it receives, and the app sends SIGINT and then SIGTERM to the
    process group, so the server often gets more than one."""
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _exit_on_sigterm(signum: int, frame: FrameType | None) -> None:
    _ignore_stop_signals()
    raise SystemExit(0)  # runs the cleanup in `main`


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="demo config (default: demo/config.json)")
    parser.add_argument("--dry-run", action="store_true", help="load no model; synthesize a tone of the estimated length")
    parser.add_argument("--host", help="overrides server.host of the config")
    parser.add_argument("--port", type=int, help="overrides server.port of the config")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    config = load_config(Path(args.config))
    host, port = args.host or config["server"]["host"], args.port or config["server"]["port"]
    if host not in LOCAL_HOSTS:
        logger.warning("Listening on %s: this server is for local use only and must not be exposed to a network", host)
    service = TTSService(config, dry_run=args.dry_run)
    Handler.service = service
    Handler.allowed_hosts = frozenset({f"127.0.0.1:{port}", f"localhost:{port}", f"[::1]:{port}"})
    Handler.allowed_origins = frozenset(config["server"].get("allowed_origins") or [])
    try:
        httpd = ThreadingHTTPServer((host, port), Handler)
    except OSError as e:
        service.close()
        raise SystemExit(f"Cannot listen on {host}:{port} ({e}); is another server running?") from None
    signal.signal(signal.SIGTERM, _exit_on_sigterm)
    service.start_loading()
    logger.info("Serving on http://%s:%d%s", host, port, " (dry run)" if args.dry_run else "")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        _ignore_stop_signals()
        httpd.server_close()
        service.close()


if __name__ == "__main__":
    main()

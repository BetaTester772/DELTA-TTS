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
import shutil
import signal
import sys
import tempfile
import threading
import time
import wave
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
MAX_BODY_BYTES = 64 * 1024 * 1024

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


def parse_request(body: dict[str, Any], config: dict[str, Any], upload_dir: Path) -> SynthesisRequest:
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
        prompt_wav=resolve_prompt(body, upload_dir),
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


def resolve_prompt(body: dict[str, Any], upload_dir: Path) -> Path:
    """The prompt audio file: an uploaded one (`prompt_wav_base64`, stored under its content
    hash, so that the pipeline's prompt cache recognizes a repeated upload) or a path."""
    data, path = body.get("prompt_wav_base64"), body.get("prompt_wav_path")
    if data:
        if not isinstance(data, str):
            raise bad_request("invalid_parameter", "prompt_wav_base64 must be a string")
        try:
            audio = base64.b64decode(data, validate=True)
        except (binascii.Error, ValueError):
            raise bad_request("prompt_unreadable", "prompt_wav_base64 is not valid base64") from None
        suffix = Path(str(body.get("prompt_wav_name") or "prompt.wav")).suffix.lower()
        if not (2 <= len(suffix) <= 6 and suffix[1:].isalnum()):
            suffix = ".wav"
        target = upload_dir / f"{hashlib.sha256(audio).hexdigest()[:32]}{suffix}"
        if not target.exists():
            target.write_bytes(audio)
        return target
    if path:
        if not isinstance(path, str):
            raise bad_request("invalid_parameter", "prompt_wav_path must be a string")
        resolved = resolve(path)
        if not resolved.is_file():
            raise bad_request("prompt_not_found", f"Prompt audio not found: {resolved}")
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
    except (OSError, RuntimeError, EOFError, ValueError, wave.Error) as e:  # soundfile raises RuntimeErrors
        raise bad_request("prompt_unreadable",
                          f"Cannot read the prompt audio ({type(e).__name__}: {e}). Use a WAV or FLAC file.") from None
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
        self.upload_dir = Path(tempfile.mkdtemp(prefix="delta_tts_demo_"))

    def start_loading(self) -> None:
        threading.Thread(target=self._load, name="model-loader", daemon=True).start()

    def close(self) -> None:
        shutil.rmtree(self.upload_dir, ignore_errors=True)

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
            request = parse_request(body, self.config, self.upload_dir)
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

    service: TTSService  # set by `main`
    server_version = "DeltaTTSDemo/1.0"

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/health":
            self._send(HTTPStatus.OK, self.service.health())
        elif path == "/config":
            self._send(HTTPStatus.OK, self.service.public_config())
        else:
            self._send_error(RequestError(HTTPStatus.NOT_FOUND, "not_found", f"No endpoint {path}"), [])

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0]
        notices: list[Notice] = []
        try:
            if path != "/synthesize":
                raise RequestError(HTTPStatus.NOT_FOUND, "not_found", f"No endpoint {path}")
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

    def _read_json(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            raise bad_request("bad_request", "Invalid Content-Length") from None
        if length > MAX_BODY_BYTES:
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


def _exit_on_sigterm(signum: int, frame: FrameType | None) -> None:
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
    service = TTSService(config, dry_run=args.dry_run)
    Handler.service = service
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
        httpd.server_close()
        service.close()


if __name__ == "__main__":
    main()

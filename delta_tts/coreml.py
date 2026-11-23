"""DELTA-TTS forward on the Apple Neural Engine (Core ML): the `fast` compute mode on macOS.

What this file contains
    Buckets         sequence-length buckets, bucket selection and input packing
    Export          DeltaTTS (LoRA merged) -> compiled Core ML model, one function per bucket
    CoreMLForward   `logits()` on the Neural Engine, with the float32 fallback of FallbackForward
    CLI             python -m delta_tts.coreml export | warm

How a forward runs on the Neural Engine
    PyTorch computes the embeddings of [SOS, text, TASK, speech] ([MASK] substitution included).
    They are zero-padded to the next bucket, with one extra channel marking real positions:

        x[0, i] = [embedding_i, 1]   for i < seq_len
        x[0, i] = [0, ..., 0,   0]   for seq_len <= i < bucket            x: (1, bucket, 897)

    The padding does not change the logits of the real positions (0 difference in float32): padded
    keys get -1e4 in attention, position ids count real positions only, and the depth-wise
    convolutions see zeros at padded positions, as at the end of an unpadded sequence. The Core ML
    model is float16; a sequence longer than the largest bucket runs in float32 in PyTorch.

Why it is built this way (measured on an M4 Pro, macOS 27.2; numbers in the README)
    - One fixed-shape function per bucket, in one multifunction model sharing the weights. A model
      with enumerated shapes does not run on the Neural Engine at all.
    - The depth-wise convolution is a (31, 1) Conv2d over (batch, channels, seq_len, 1). As a
      Conv1d or a (1, 31) Conv2d it is not supported, and Core ML splits the model between the
      Neural Engine and the GPU (50% slower).
    - Compute units CPU_AND_NE; ALL mixes in the GPU and is slower.
    - Loading a function compiles it for the Neural Engine (about 10 s). macOS caches the result
      per user, executable name (`python`, `python3`, ...) and model path; an `.mlpackage` is
      compiled to a new temporary path at every load, so the model is saved compiled
      (`.mlmodelc`), and `export` loads every function once to fill the cache.
    - A load holds the Python GIL for its whole duration. The runtime therefore compiles in a
      helper process, which fills the shared cache, and then loads the cached function (about
      0.3 s) in this process. Until a bucket is ready, its utterances run in float32.
"""

import argparse
import contextlib
import copy
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from importlib.util import find_spec
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from delta_tts.model import HIDDEN_DIM, DeltaTTS
from delta_tts.precision import FallbackForward

__all__ = ["BUCKETS", "CoreMLForward", "default_path", "export", "is_available", "pack", "select_bucket", "warm"]

logger = logging.getLogger(__name__)

# -----------------------------
# Buckets
# -----------------------------
# Each bucket is at most a third longer than the previous one, which bounds the padding. Every
# bucket costs a compilation (about 10 s) and about 0.85 GB of macOS cache when used.
BUCKETS = (192, 256, 320, 384, 512, 640, 768, 1024)
TYPICAL_SEQ_LEN = 300  # about 5 s of prompt and 5 s of output; buckets are prepared nearest first
INPUT_DIM = HIDDEN_DIM + 1  # embedding + real-position channel
MODEL_NAME = "delta_tts_ane.mlmodelc"


def default_path(checkpoint: str) -> str:
    """Where `export` writes the Core ML model by default: next to the DELTA-TTS checkpoint."""
    return os.path.join(os.path.dirname(os.path.abspath(checkpoint)), MODEL_NAME)


def is_available() -> bool:
    """Whether Core ML can run here: macOS with coremltools installed (the `mac` extra)."""
    return sys.platform == "darwin" and find_spec("coremltools") is not None


def select_bucket(seq_len: int, buckets: Sequence[int]) -> int | None:
    """The smallest bucket that holds `seq_len` positions, or None if none does."""
    return min((b for b in buckets if b >= seq_len), default=None)


def pack(embeddings: np.ndarray, bucket: int) -> np.ndarray:
    """Zero-pad embeddings to a bucket and append the real-position channel.

    Args:
        embeddings: Embeddings of [SOS, text, TASK, speech]. Shape: (seq_len, hidden_dim)
        bucket: Padded length, at least seq_len.
    Returns:
        Core ML input `x`, float16. Shape: (1, bucket, hidden_dim + 1)
    """
    seq_len, hidden_dim = embeddings.shape
    x = np.zeros((1, bucket, hidden_dim + 1), dtype=np.float16)  # (1, bucket, hidden_dim + 1)
    x[0, :seq_len, :hidden_dim] = embeddings
    x[0, :seq_len, hidden_dim] = 1
    return x


def prewarm_order(buckets: Iterable[int]) -> list[int]:
    """Buckets in the order they are prepared: nearest to a typical sequence length first."""
    return sorted(buckets, key=lambda b: (abs(b - TYPICAL_SEQ_LEN), b))


def read_buckets(path: str) -> list[int]:
    """Buckets of an exported model, from the function names (`L<bucket>`) in its program."""
    with open(os.path.join(path, "model.mil"), encoding="utf-8") as f:
        buckets = sorted({int(m) for m in re.findall(r"^\s*func L(\d+)<", f.read(), flags=re.MULTILINE)})
    if not buckets:
        raise ValueError(f"{path} has no L<bucket> functions; export it with `python -m delta_tts.coreml export`")
    return buckets


# -----------------------------
# Export
# -----------------------------
class _MaskedDepthwiseConv(nn.Module):
    """ConvModule.dw for export: a (kernel, 1) Conv2d over (batch, channels, seq_len, 1), which
    the Neural Engine runs, with the input zeroed at padded positions.

    Args:
        conv: The trained depth-wise Conv1d.
    """

    def __init__(self, conv: nn.Conv1d) -> None:
        super().__init__()
        assert conv.bias is not None
        self.register_buffer("weight", conv.weight.detach().unsqueeze(3))  # (channels, 1, kernel, 1)
        self.register_buffer("bias", conv.bias.detach())  # (channels,)
        self.groups = conv.groups
        self.padding = int(conv.padding[0])  # type: ignore[index]
        self.valid = torch.ones(1, 1, 1)  # (1, 1, seq_len); set by _ExportForward before every forward

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, channels, seq_len) -> (batch, channels, seq_len)"""
        x = (x * self.valid).unsqueeze(3)  # (batch, channels, seq_len, 1)
        x = F.conv2d(x, self.weight, self.bias, padding=(self.padding, 0), groups=self.groups)  # (batch, channels, seq_len, 1)
        return x.squeeze(3)  # (batch, channels, seq_len)


class _ExportForward(nn.Module):
    """The part of `DeltaTTS.logits` that runs on the Neural Engine: packed embeddings to logits.

    Args:
        model: DELTA-TTS with LoRA merged. Its ConvModules are modified, so pass a copy.
    """

    def __init__(self, model: DeltaTTS) -> None:
        super().__init__()
        self.backbone = model.bidirectional_encoder.qwen2_model.get_base_model().model
        self.decoder = model.llm_decoder
        self.dws: list[_MaskedDepthwiseConv] = []
        for conv in model.conv_modules:  # the blocks' forward hooks call these ConvModules
            conv.dw = _MaskedDepthwiseConv(conv.dw)  # type: ignore[arg-type]
            self.dws.append(conv.dw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Packed input (see `pack`) to logits.

        Args:
            x: Embeddings and real-position channel. Shape: (1, bucket, hidden_dim + 1)
        Returns:
            Logits; rows past the real sequence are meaningless. Shape: (1, bucket, vocab_size)
        """
        embeddings, valid = x[:, :, :-1], x[:, :, -1]  # (1, bucket, hidden_dim), (1, bucket)
        for dw in self.dws:
            dw.valid = valid.unsqueeze(1)  # (1, 1, bucket)
        key_mask = ((valid - 1.0) * 1e4).view(1, 1, 1, -1)  # (1, 1, 1, bucket); 0 = attend, -1e4 = padding
        position_ids = (torch.cumsum(valid, dim=1) - 1.0).clamp(min=0).to(torch.int32)  # (1, bucket)
        hidden = self.backbone(inputs_embeds=embeddings, attention_mask=key_mask, position_ids=position_ids,
                               cache_position=position_ids[0], use_cache=False, return_dict=True).last_hidden_state
        return self.decoder(hidden)  # (1, bucket, vocab_size)


def export(model: DeltaTTS, out: str, buckets: Sequence[int] = BUCKETS, log: Callable[[str], None] = logger.info) -> None:
    """Export the DELTA-TTS forward as a compiled float16 Core ML model, one function per bucket.

    Each bucket is converted on its own (fixed shapes run on the Neural Engine) and merged into
    one multifunction model that stores the weights once. Only the compiled model is kept.

    Args:
        model: Float32 DELTA-TTS on the CPU, checkpoint loaded and LoRA merged. Not modified.
        out: Output path, ending in `.mlmodelc`; replaced if it exists.
        buckets: Sequence lengths, one Core ML function each.
        log: Progress messages.
    """
    import coremltools as ct
    from coremltools.models.utils import MultiFunctionDescriptor, save_multifunction

    if not out.endswith(".mlmodelc"):
        raise ValueError(f"out must end in .mlmodelc, got {out}")
    buckets = sorted(set(buckets))
    forward = _ExportForward(copy.deepcopy(model).float().cpu()).eval()
    out_dir = os.path.dirname(os.path.abspath(out))
    os.makedirs(out_dir, exist_ok=True)

    # The intermediate packages (about 0.8 GB each) are merged one bucket at a time, so that at
    # most three exist at once.
    with tempfile.TemporaryDirectory(dir=out_dir, prefix=".delta_tts_coreml_") as tmp:
        merged: str | None = None
        start = time.perf_counter()
        for i, bucket in enumerate(buckets):
            part = _convert(forward, bucket)
            desc = MultiFunctionDescriptor()
            for done in buckets[:i]:  # the buckets merged so far
                desc.add_function(str(merged), src_function_name=f"L{done}", target_function_name=f"L{done}")
            desc.add_function(part.package_path, src_function_name="main", target_function_name=f"L{bucket}")
            desc.default_function_name = f"L{buckets[0]}"
            path = os.path.join(tmp, f"merged_{i}.mlpackage")
            save_multifunction(desc, path)
            del part  # removes its temporary package
            if merged is not None:
                shutil.rmtree(merged)
            merged = path
            log(f"converted bucket {bucket} ({i + 1}/{len(buckets)}, {time.perf_counter() - start:.0f} s)")

        assert merged is not None
        compiled = ct.utils.compile_model(merged, os.path.join(tmp, MODEL_NAME))
        if os.path.exists(out):
            shutil.rmtree(out)
        os.replace(compiled, out)
    log(f"saved {out} ({_size_gb(out):.2f} GB, buckets {buckets})")


def _convert(forward: _ExportForward, bucket: int) -> Any:
    """Trace and convert the forward for one bucket. Returns a `ct.models.MLModel` backed by a
    temporary package."""
    import coremltools as ct

    example = torch.zeros(1, bucket, INPUT_DIM)  # (1, bucket, hidden_dim + 1); padding in the last quarter
    real = bucket * 3 // 4
    example[0, :real, :-1] = 0.02 * torch.randn(real, HIDDEN_DIM, generator=torch.Generator().manual_seed(0))
    example[0, :real, -1] = 1
    with torch.no_grad():
        traced = torch.jit.trace(forward, example, check_trace=False)
    with _numpy2_cast_fix():
        return ct.convert(
            traced,
            inputs=[ct.TensorType(name="x", shape=example.shape, dtype=np.float16)],
            outputs=[ct.TensorType(name="logits", dtype=np.float16)],
            convert_to="mlprogram",
            compute_precision=ct.precision.FLOAT16,
            minimum_deployment_target=ct.target.macOS15,
            skip_model_load=True,
        )


@contextlib.contextmanager
def _numpy2_cast_fix() -> Iterator[None]:
    """Work around a coremltools 9.0 bug with NumPy 2, for the duration of a conversion only.

    The torch frontend converts `int(x)` / `bool(x)` of a constant 1-element tensor with
    `int(array)`, which NumPy 2 rejects for arrays with ndim > 0. Such constants are cast
    through their single element; every other cast is unchanged.
    """
    from coremltools.converters.mil.frontend.torch import ops

    original = ops._cast

    def _cast(context: Any, node: Any, dtype: type, dtype_name: str) -> None:
        x = ops._get_inputs(context, node, expected=1)[0]
        if x.can_be_folded_to_const() and np.ndim(x.val) > 0 and np.size(x.val) == 1:
            context.add(ops.mb.const(val=dtype(np.asarray(x.val).reshape(-1)[0]), name=node.name), node.name)
            return
        original(context, node, dtype, dtype_name)

    ops._cast = _cast
    try:
        yield
    finally:
        ops._cast = original


def warm(path: str, buckets: Sequence[int] | None = None, log: Callable[[str], None] = logger.info) -> dict[int, float]:
    """Load every function once on the Neural Engine, so that macOS caches its compiled program.

    The cache is per executable name: warm with the same interpreter name (`python`,
    `python3`, ...) that runs the synthesis.

    Args:
        path: Compiled Core ML model from `export`.
        buckets: Buckets to warm; all of the model's by default.
        log: Progress messages.
    Returns:
        Load time in seconds per bucket (about 10 s when compiled, under 1 s when cached).
    """
    import coremltools as ct

    seconds = {}
    for bucket in prewarm_order(buckets or read_buckets(path)):
        start = time.perf_counter()
        function = ct.models.CompiledMLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE, function_name=f"L{bucket}")
        seconds[bucket] = time.perf_counter() - start
        logits = function.predict({"x": pack(np.zeros((1, HIDDEN_DIM), np.float16), bucket)})["logits"]
        if not np.isfinite(logits[0, 0]).all():
            raise RuntimeError(f"bucket {bucket}: non-finite logits for a zero input")
        log(f"bucket {bucket}: loaded in {seconds[bucket]:.1f} s")
    return seconds


def _size_gb(path: str) -> float:
    return sum(os.path.getsize(os.path.join(d, f)) for d, _, files in os.walk(path) for f in files) / 1e9


# -----------------------------
# Runtime
# -----------------------------
class CoreMLForward(FallbackForward):
    """`DeltaTTS.logits` on the Neural Engine (Core ML, float16), with a float32 fallback.

    An utterance runs in float32 (the PyTorch model) when its sequence is longer than the largest
    bucket or when its bucket is not loaded yet; it then moves to the front of the loading queue.
    Non-finite logits fall back to float32 for the rest of the utterance (`FallbackForward`).
    Call `reset` before every utterance.

    Args:
        model: The float32 DELTA-TTS model: computes the embeddings and is the fallback.
        path: Compiled Core ML model from `export`.
        prewarm: Load every bucket in the background, nearest to a typical length first.
            Otherwise a bucket is loaded when an utterance first needs it.
    """

    name = "Neural Engine float16"

    def __init__(self, model: DeltaTTS, path: str, prewarm: bool = True) -> None:
        super().__init__(model)
        self.path = os.path.abspath(path)
        self.buckets = read_buckets(self.path)
        self.loader = _BucketLoader(self.path, prewarm_order(self.buckets) if prewarm else [])
        self.in_exact_utterance = False  # this utterance's bucket was not ready, or it is too long

    def reset(self) -> None:
        """Start a new utterance on the Neural Engine."""
        super().reset()
        self.in_exact_utterance = False

    def wait(self, buckets: Iterable[int] | None = None, timeout: float | None = None) -> bool:
        """Block until `buckets` (all by default) are loaded; whether they all are."""
        return self.loader.wait(prewarm_order(self.buckets if buckets is None else buckets), timeout)

    def _fast_logits(self, text_token: torch.Tensor, speech_token: torch.Tensor) -> torch.Tensor | None:
        if self.in_exact_utterance:
            return None
        seq_len = 1 + text_token.shape[1] + 1 + speech_token.shape[1]
        bucket = select_bucket(seq_len, self.buckets)
        function = None if bucket is None else self.loader.get(bucket)
        if bucket is None or function is None:
            # The whole utterance runs in float32, so that its result does not depend on when
            # the bucket finishes loading.
            self.in_exact_utterance = True
            logger.info("Sequence length %d: %s; this utterance runs in float32", seq_len,
                        "longer than the largest Core ML bucket" if bucket is None else f"Core ML bucket {bucket} is not loaded yet")
            return None

        embeddings = self.exact.embed(text_token, speech_token)[0]  # (seq_len, hidden_dim)
        x = pack(embeddings.float().cpu().numpy(), bucket)  # (1, bucket, hidden_dim + 1)
        logits = function.predict({"x": x})["logits"][:, :seq_len]  # (1, seq_len, vocab_size), float16
        return torch.from_numpy(logits.astype(np.float32)).to(text_token.device)  # (1, seq_len, vocab_size)


class _BucketLoader:
    """Loads the Core ML function of each bucket in a background thread, requested buckets first.

    A helper process (the same interpreter, hence the same macOS cache) loads each function
    first, which compiles it if needed; this process then loads it from the cache. Loads run
    one at a time, as parallel loads are not faster. The helper exits when the queue is empty.

    Args:
        path: Compiled Core ML model.
        queue: Buckets to load in the background, in order.
    """

    def __init__(self, path: str, queue: list[int]) -> None:
        self.path = path
        self.ready: dict[int, Any] = {}  # bucket -> loaded function (ct.models.CompiledMLModel)
        self.failed: set[int] = set()
        self._queue = list(queue)
        self._loading: int | None = None
        self._cond = threading.Condition()
        self._helper: subprocess.Popen[str] | None = None
        threading.Thread(target=self._run, name="delta-tts-coreml-loader", daemon=True).start()

    def get(self, bucket: int) -> Any:
        """The loaded function of `bucket`, or None after moving it to the front of the queue."""
        with self._cond:
            if bucket in self.ready:
                return self.ready[bucket]
            if bucket not in self.failed and bucket != self._loading:
                if bucket in self._queue:
                    self._queue.remove(bucket)
                self._queue.insert(0, bucket)
                self._cond.notify_all()
            return None

    def wait(self, buckets: list[int], timeout: float | None = None) -> bool:
        """Queue `buckets` (behind the queue) and block until each is loaded or failed."""
        with self._cond:
            for bucket in buckets:
                if bucket not in self.ready and bucket not in self.failed and bucket != self._loading and bucket not in self._queue:
                    self._queue.append(bucket)
            self._cond.notify_all()
            self._cond.wait_for(lambda: all(b in self.ready or b in self.failed for b in buckets), timeout)
            return all(b in self.ready for b in buckets)

    def _run(self) -> None:
        while True:
            with self._cond:
                if not self._queue:
                    self._stop_helper()
                self._cond.wait_for(lambda: bool(self._queue))
                bucket = self._loading = self._queue.pop(0)
            start = time.perf_counter()
            function = None
            try:
                helper_seconds = self._load_in_helper(bucket)
                function = self._load(bucket)
                logger.info("Core ML bucket %d ready in %.1f s (helper load %.1f s)", bucket, time.perf_counter() - start, helper_seconds)
            except (OSError, RuntimeError, ValueError) as e:  # the bucket's utterances run in float32
                logger.warning("Core ML bucket %d failed to load; its utterances run in float32: %s", bucket, e)
            with self._cond:
                if function is None:
                    self.failed.add(bucket)
                else:
                    self.ready[bucket] = function
                self._loading = None
                self._cond.notify_all()

    def _load_in_helper(self, bucket: int) -> float:
        """Load `bucket` in the helper process, which compiles it into the cache if needed.
        Returns the helper's load time in seconds."""
        if self._helper is None or self._helper.poll() is not None:
            self._helper = subprocess.Popen([sys.executable, "-c", _HELPER, self.path], stdin=subprocess.PIPE,
                                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        assert self._helper.stdin is not None and self._helper.stdout is not None
        self._helper.stdin.write(f"L{bucket}\n")
        self._helper.stdin.flush()
        reply = self._helper.stdout.readline().split(maxsplit=2)
        if reply[:2] != [f"L{bucket}", "ok"]:
            raise RuntimeError(f"helper process: {' '.join(reply) or 'exited'}")
        return float(reply[2])

    def _stop_helper(self) -> None:
        if self._helper is not None:
            assert self._helper.stdin is not None
            self._helper.stdin.close()  # the helper exits at the end of its input
            self._helper.wait()
            self._helper = None

    def _load(self, bucket: int) -> Any:
        """Load `bucket` in this process, from the cache the helper filled (holds the GIL)."""
        # coremltools warns at import about untested torch and scikit-learn versions, which
        # matter for conversion only, not for running a compiled model.
        coremltools_logger = logging.getLogger("coremltools")
        level = coremltools_logger.level
        coremltools_logger.setLevel(logging.ERROR)
        try:
            import coremltools as ct
        finally:
            coremltools_logger.setLevel(level)
        return ct.models.CompiledMLModel(self.path, compute_units=ct.ComputeUnit.CPU_AND_NE, function_name=f"L{bucket}")


# Helper process: reads function names on stdin, loads each on the Neural Engine and replies
# "<name> ok <seconds>". Only coremltools is imported. It exits when stdin closes.
_HELPER = """
import sys, time
import coremltools as ct
for line in sys.stdin:
    name = line.strip()
    start = time.perf_counter()
    try:
        ct.models.CompiledMLModel(sys.argv[1], compute_units=ct.ComputeUnit.CPU_AND_NE, function_name=name)
        print(name, "ok", round(time.perf_counter() - start, 2), flush=True)
    except Exception as e:
        print(name, "error", repr(e).replace(chr(10), " "), flush=True)
"""


# -----------------------------
# CLI
# -----------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Export DELTA-TTS for the Apple Neural Engine (the `fast` compute mode on macOS).")
    sub = parser.add_subparsers(dest="command", required=True)
    exp = sub.add_parser("export", help="convert, compile and warm the Core ML model")
    exp.add_argument("--backbone_dir", default="pretrained_models/Fun-CosyVoice3-0.5B-2512")
    exp.add_argument("--checkpoint", default="pretrained_models/DELTA-TTS/delta_tts.pt")
    exp.add_argument("--out", help=f"output .mlmodelc (default: {MODEL_NAME} next to the checkpoint)")
    exp.add_argument("--buckets", type=int, nargs="+", default=list(BUCKETS), help="sequence-length buckets")
    exp.add_argument("--no_warm", action="store_true", help="skip loading every function once (about 10 s each)")
    wrm = sub.add_parser("warm", help="load every function once, so that macOS caches the compiled programs")
    wrm.add_argument("--checkpoint", default="pretrained_models/DELTA-TTS/delta_tts.pt")
    wrm.add_argument("--model", help=f"compiled model (default: {MODEL_NAME} next to the checkpoint)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("coremltools").setLevel(logging.WARNING)  # it logs every converted op at INFO
    if not is_available():
        raise SystemExit("Core ML needs macOS and coremltools: uv sync --extra mac")

    if args.command == "export":
        out = args.out or default_path(args.checkpoint)
        logger.info("Loading DELTA-TTS")
        model = DeltaTTS(args.backbone_dir).float().eval()
        model.load_checkpoint(args.checkpoint)
        model.merge_lora()
        export(model, out, args.buckets)
        if not args.no_warm:
            warm(out)
    else:
        warm(args.model or default_path(args.checkpoint))


if __name__ == "__main__":
    main()

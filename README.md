# DELTA-TTS

**Adapting Autoregressive Model into Diffusion Language Model for Text-to-Speech**

DELTA-TTS converts a pretrained autoregressive TTS model into a discrete diffusion language model with LoRA and a convolution module, so that speech tokens are decoded in confidence order instead of left to right.

[![Demo](https://img.shields.io/badge/Demo-Page-blue)](https://slpr-lab.github.io/DELTA-TTS/)
[![arXiv](https://img.shields.io/badge/arXiv-2607.04140-B31B1B)](https://arxiv.org/abs/2607.04140)
[![OpenReview](https://img.shields.io/badge/OpenReview-NeurIPS_2026-8C1B13)](https://openreview.net/forum?id=c4cM3P84zM)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-FFD21E)](https://huggingface.co/slpr-lab/DELTA-TTS)

## News

- ⭐ **[2026.09.25]** DELTA-TTS is accepted to NeurIPS 2026.
- ⭐ **[2026.05.24]** DELTA-TTS is accepted to the SPIGM Workshop at ICML 2026.

## Installation

```bash
git clone --recursive https://github.com/slpr-lab/DELTA-TTS.git
cd DELTA-TTS
uv sync
```

FFmpeg must be installed on the system, since torchaudio reads and writes audio through it (e.g. `apt install ffmpeg`).

### macOS (Apple Silicon)

```bash
brew install ffmpeg
uv sync --extra mac
```

- The `mac` extra installs `onnxruntime` in place of `onnxruntime-gpu`, which has no macOS build.
- On macOS, PyTorch 2.15 is installed instead of the paper's 2.10: its MPS backend is faster and generates the same speech tokens.
- DELTA-TTS and the CosyVoice 3 flow decoder and vocoder run on MPS; the ONNX frontend (speech tokenizer, speaker embedding) runs on the CPU. MPS has no float64, so the vocoder's F0 predictor runs in float32, which gives the same F0 when the whole utterance is vocoded at once.
- `--device` picks CUDA, then MPS, then CPU.

### Audio processing (optional)

`uv sync --extra audio` (on macOS, `uv sync --extra mac --extra audio`) adds [Silero VAD](https://github.com/snakers4/silero-vad) and [NVIDIA RE-USE](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-RE-USE) speech enhancement.
RE-USE runs on a pure-PyTorch port of its Mamba layers (`delta_tts/reuse.py`, with a Metal kernel on MPS), so `mamba-ssm` is not needed.

## Checkpoints

| Model | Backbone | Trainable modules | Training data | Download |
|---|---|---|---|---|
| DELTA-TTS | Fun-CosyVoice3-0.5B-2512 | LoRA (r=64) and depth-wise convolution (k=31), 94M | LibriTTS, 585 h | [Hugging Face](https://huggingface.co/slpr-lab/DELTA-TTS) |

```bash
uv run huggingface-cli download FunAudioLLM/Fun-CosyVoice3-0.5B-2512 --local-dir pretrained_models/Fun-CosyVoice3-0.5B-2512
uv run huggingface-cli download slpr-lab/DELTA-TTS --local-dir pretrained_models/DELTA-TTS
```

## Inference

Synthesize one sentence in the voice of a prompt:

```bash
uv run python infer.py \
    --prompt_wav prompt.wav \
    --prompt_text "Transcript of the prompt audio." \
    --text "Text to synthesize." \
    --output output.wav
```

Synthesize a test set given as a Seed-TTS style `meta.lst` (`utt_id|prompt_text|prompt_wav|target_text`):

```bash
uv run python infer.py --meta third_party/F5-TTS/data/seedtts_testset/en/meta.lst --output_dir outputs/seedtts_en
```

The output length is estimated with the length rule of [OmniVoice](https://github.com/k2-fsa/OmniVoice).

`--compute exact` (default) runs DELTA-TTS in float32. `--compute fast` runs it in float16; a step whose float16 logits overflow (inf/NaN) is recomputed in float32 and the rest of that utterance stays in float32. Fast-mode outputs differ numerically from exact ones. On MPS, the fast mode runs DELTA-TTS on the Apple Neural Engine when its Core ML model is exported (below), otherwise in float16 on MPS; the log says which.

`--length_scale` multiplies the estimated length (above 1: slower speech).

With the `audio` extra, the prompt can be cleaned before synthesis and the output trimmed after it (all off by default; the paper results use none):

| Flag | Effect |
|---|---|
| `--enhance_prompt` | Enhance the prompt with RE-USE (noise, reverberation, codec artifacts, ...) |
| `--trim_prompt` | Trim leading and trailing non-speech of the prompt with Silero VAD |
| `--trim_output` | Trim leading and trailing non-speech of the output with Silero VAD |

Only leading and trailing non-speech is removed, so the prompt transcript stays valid. The same processing on a single file:

```bash
uv run python -m delta_tts.audio noisy.wav clean.wav --enhance --trim
```

### Neural Engine (macOS)

Export once (about 5 min and 13 GB of RAM; writes `pretrained_models/DELTA-TTS/delta_tts_ane.mlmodelc`, 0.85 GB):

```bash
uv run python -m delta_tts.coreml export
uv run python infer.py --compute fast ...
```

- The Core ML model is float16, with one fixed-shape function per sequence-length bucket: 160 to 640 positions of text, prompt and output speech together, which is about 20 s of prompt and output audio. The embeddings stay in PyTorch and are padded to the bucket without changing the logits. Longer sequences run in float32 on MPS, where longer buckets would not be faster (below). `--coreml_model` points to a model elsewhere.
- Loading a bucket compiles it for the Neural Engine (8 to 12 s). macOS caches the result per user, model path and interpreter name (`python`, `python3`, ...), so the export loads every bucket once with the interpreter that ran it; `python -m delta_tts.coreml warm` does it again, e.g. for another interpreter name. macOS may delete this cache when the disk runs low; it is then rebuilt in the background.
- Buckets are loaded in the background, the one a request needs first. Until its bucket is ready, an utterance runs in float32 on MPS, so synthesis never waits for a compilation. Each load holds the Python interpreter for about 0.3 s.
- In Python, `DeltaTTSPipeline.set_compute("exact" | "fast")` switches a loaded pipeline (the fast backend is built once and kept; do not switch during `synthesize`), and `backend_status()` tells what runs DELTA-TTS, including the Neural Engine's loading state.

Measured on an M4 Pro (48 GB, macOS 27.2) with 12 to 24 GB of free disk; each extra bucket costs one more compilation and about 0.85 GB of macOS cache, so 128 (rarely reached) and 768 and 1024 (no faster than MPS) are not exported:

| Loading the buckets | `.mlpackage`, loaded in-process (prototype) | `.mlmodelc`, compiled in a helper process |
|---|---|---|
| First process | 13 to 15 s for bucket 320 alone, blocking Python | first bucket after 11 s, all 7 after 66 to 68 s, in the background |
| Later processes | the same: recompiled at every load | first bucket after 2.0 to 2.1 s, all 7 after 5.0 to 5.2 s (cache kept) |

DELTA-TTS forward per unmasking step (embeddings, Core ML call and logits back on MPS; median of 10, two runs, PyTorch 2.15):

| Sequence length (bucket) | Neural Engine | MPS float32 | MPS float16 |
|---|---|---|---|
| 120 (128, not exported) | 13 ms | 36 ms | 34 to 35 ms |
| 150 (160) | 22 ms | 42 ms | 40 ms |
| 150 (192) | 29 ms | 42 ms | 40 ms |
| 180 (192) | 29 ms | 48 ms | 44 ms |
| 300 (320) | 49 ms | 69 ms | 63 ms |
| 450 (512) | 82 ms | 102 ms | 92 ms |
| 640 (640) | 108 to 110 ms | 133 to 134 ms | 117 to 118 ms |
| 700 (768, not exported) | 140 ms | 143 ms | 125 to 129 ms |
| 1000 (1024, not exported) | 205 to 207 ms | 209 to 211 ms | 183 ms |

On the 8 zero-shot prompts of the demo page (150 to 452 positions; measured with buckets 192 to 1024), the Neural Engine gives 73% of the float32 speech tokens (float16 on MPS: 97%), and Whisper (base.en) finds no word errors in the float32, Neural Engine or float16 outputs. The Neural Engine saturates instead of producing inf/NaN, so its float32 fallback catches non-finite outputs but not a saturated activation; on these prompts the largest activation in float32 is 3115, 5% of the float16 range.

## Demo app

A macOS app for a live demo (SwiftUI; macOS 26 or newer) in `demo/DeltaTTSDemo`: a visitor reads one sentence into the microphone, types what their voice should say, and hears it.
It talks to a small local server, `demo/server.py`, which keeps the model loaded.
Both read `demo/config.json`: server settings, the header titles, the sentence to read and the default text, recording settings, sample voices (the prompts of the demo page), synthesis and processing defaults, and warning thresholds.

```bash
uv run python demo/server.py              # loads the model; GET /health reports "loading", then "ready"
uv run python demo/server.py --dry-run    # no model: returns a tone of the estimated output length
demo/DeltaTTSDemo/make_app.sh             # builds demo/DeltaTTSDemo/build/DeltaTTSDemo.app
open demo/DeltaTTSDemo/build/DeltaTTSDemo.app
```

- Use the app bundle: macOS grants microphone access only to apps whose Info.plist explains it. `make_app.sh` builds the release binary, writes that Info.plist and signs the bundle ad hoc; macOS asks for the microphone on the first recording (if access was denied, allow DeltaTTSDemo in System Settings > Privacy & Security > Microphone). A rebuilt bundle may be asked again. `swift run` in `demo/DeltaTTSDemo` also works, with sample voices only.
- The main screen shows four steps as a guide: read and record (a countdown, then the sentence shown large while recording, up to 12 s), type what to say, generate, listen. One button in a fixed place at the bottom does the next thing: Record your voice, Stop, Generate my voice, Play, Play again; Re-record, Play my recording and Edit text sit beside it. Shortcuts: Space (the next step, when no text field has focus), Command-R record or stop, Command-G or Return in the text generate, Command-P play, Command-N next visitor.
- As soon as a recording stops, the app sends the whole recording to `POST /prepare`, which enhances it (RE-USE, which estimates the noise from the edges without speech), then trims it (Silero VAD) and caches its features while the visitor reads step 2. It returns the processed audio, which "Play my recording" plays: what the model will use. Step 1 shows "Preparing your voice..." and then "Ready", with warnings such as no speech found; a failure offers Record again. The app trims silence itself only when the server's `trim_prompt` is off. Both prompt processing steps are on by default (`processing` in the config; needs the `audio` extra).
- Generate then sends the returned `prompt_id`, so it only generates. While it runs, the primary spot shows the stage over a progress bar fed by `GET /progress`, which follows the pipeline's progress reports: the prompt (only if it was not prepared), the DELTA-TTS tokens (one step per forward pass), the acoustic model (10 flow-matching steps) and the vocoder.
- After the first full listen the primary button reads "Play again", with "Regenerate" (Command-G) beside it: it generates again from the same prepared prompt and text, so it shows how fast generation alone is (at temperature 0 the result is the same audio). "Next visitor" (at the top right, in the action bar once the voice has been heard, and Command-N) cancels what is in flight, deletes the recording, clears the app and asks the server to forget it (`POST /reset`).
- The header has the microphone, the output device and the mode (exact or fast); Settings edits the same values. Recording uses the system default microphone unless one is chosen. Playback uses the system default output unless one is chosen; a chosen output gives way to the system default when that changes (e.g. headphones plugged in). Leading and trailing silence of a recording is cut before upload.
- Generation timings (RTF) are not shown to visitors; they are in Settings > General and on the app's standard output.
- Recordings are kept in a private temporary folder and deleted on Next visitor and when the app quits. The server keeps uploads only in its own temporary folder, removed by `/reset` and when it exits.
- Settings (Command-,) edit every section of the config, start and stop the server, and show its log. They are saved to `~/Library/Application Support/DeltaTTSDemo/config.json` (or `$DELTA_TTS_DEMO_CONFIG`), which the app passes to a server it starts; Reset to Defaults reloads `demo/config.json`. `app.ui_scale` sets the size of the main screen.
- `POST /synthesize` takes `text`, `prompt_text`, `prompt_wav_base64` (or `prompt_wav_path`) and optional synthesis and processing fields, and returns a base64 WAV with `duration_s`, `elapsed_s`, `rtf` and `warnings` (`{level, code, message}`): the library's logged warnings, the fast mode's float32 fallback, and checks of the text, transcript, prompt and estimated output length. `POST /prepare` takes the prompt part of that request and returns `prompt_id`, which `/synthesize` accepts instead of the audio, with the processed prompt audio (`audio_wav_base64`) and its duration. `GET /progress` reports the request in flight (state, stage, step, an overall fraction that never goes backwards). `POST /reset` deletes the uploaded and prepared prompts and clears the cached prompt features (speaker embeddings); a request that was waiting for the lock when a reset ran is rejected with 409 instead of storing anything.

Security: the server is for local use by the app and must not be exposed to a network.
It listens on 127.0.0.1 and accepts only loopback `Host` headers; it rejects requests with an `Origin` header (unless listed in `server.allowed_origins`), sends no CORS headers, and requires `Content-Type: application/json` on POST, so that web pages cannot use it.
`prompt_wav_path` must point inside the repository, a preset directory or the upload directory; uploads are limited to 18 MB and 60 s, checked before they are stored, and kept in a private temporary directory of at most 32 files and 200 MB.

## Evaluation

WER and SIM are computed with the [evaluation code of F5-TTS](https://github.com/SWivid/F5-TTS/tree/main/src/f5_tts/eval), included as a submodule and used without modification.
As F5-TTS expects, place the Seed-TTS test set at `third_party/F5-TTS/data/seedtts_testset/en` and the WavLM checkpoint at `third_party/checkpoints/UniSpeech/wavlm_large_finetune.pth`, then run:

```bash
bash eval.sh outputs/seedtts_en
```

## Acknowledgement

This work builds on [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) (backbone, speech tokenizer, flow-matching decoder and vocoder), [F5-TTS](https://github.com/SWivid/F5-TTS) (evaluation code) and [OmniVoice](https://github.com/k2-fsa/OmniVoice) (length estimation rule).
The optional audio processing uses [Silero VAD](https://github.com/snakers4/silero-vad) and [NVIDIA NemotronLabs RE-USE](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-RE-USE).
We thank the authors for releasing their work.

## Citation

```bibtex
@inproceedings{moon2026deltatts,
  title     = {{DELTA}-{TTS}: Adapting Autoregressive Model into Diffusion Language Model for Text-to-Speech},
  author    = {Junwon Moon and Yejin Lee and Seungbeom Kim and Hoseong Ahn and Sewoong Park and Heeseung Kim and Kyuhong Shim},
  booktitle = {The Fortieth Annual Conference on Neural Information Processing Systems},
  year      = {2026},
  url       = {https://openreview.net/forum?id=c4cM3P84zM}
}
```

## License

The code is released under the Apache-2.0 license.
The model weights are built on CosyVoice 3 and follow its Apache-2.0 license.
`delta_tts/reuse.py` is a port of the RE-USE code and follows its OpenMDW-1.1 license, as do the RE-USE weights.

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

`--compute exact` (default) runs DELTA-TTS in float32. `--compute fast` runs it in float16; a step whose float16 logits overflow (inf/NaN) is recomputed in float32 and the rest of that utterance stays in float32. Fast-mode outputs differ numerically from exact ones.

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

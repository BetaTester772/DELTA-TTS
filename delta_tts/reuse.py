# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
#
# Port of NVIDIA NemotronLabs RE-USE (https://huggingface.co/nvidia/NVIDIA-NemotronLabs-RE-USE),
# files models/{generator_SEMamba_time_d4, mamba_block2_SEMamba, codec_module_time_d4, stfts}.py
# and the enhancement step of inference.py. The network is unchanged and loads the released
# weights as is. The only difference is `Mamba`: the original imports it from mamba-ssm, whose
# kernels need CUDA, so it is re-implemented here in plain PyTorch (runs on CUDA, MPS and CPU),
# with fused Metal kernels on MPS.
"""RE-USE universal speech enhancement (9.6M parameters, 8-48 kHz input).

What this file contains
    Mamba            Mamba (selective state space) layer, weight-compatible with mamba-ssm: plain
                     PyTorch, with fused Metal kernels on MPS
    SEMamba          the RE-USE network: dense encoder, 30 time-frequency Mamba blocks, decoders
    ReUSE            loads the released checkpoint and enhances a waveform at its own sample rate
"""

import json
import math
import os
from typing import Any

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

__all__ = ["Mamba", "ReUSE", "SEMamba"]

REPO_ID = "nvidia/NVIDIA-NemotronLabs-RE-USE"


# -----------------------------
# Mamba (pure PyTorch)
# -----------------------------
class Mamba(nn.Module):
    """Mamba layer with the parameter layout of `mamba_ssm.Mamba` (mamba_simple.py).

    Args:
        d_model: Model dimension.
        d_state: SSM state size.
        d_conv: Kernel size of the causal depth-wise convolution.
        expand: Inner dimension is `expand * d_model`.
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2) -> None:
        super().__init__()
        self.d_state = d_state
        self.d_inner = expand * d_model
        self.dt_rank = math.ceil(d_model / 16)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv, groups=self.d_inner, padding=d_conv - 1)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner)
        self.A_log = nn.Parameter(torch.zeros(self.d_inner, d_state))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: Shape: (batch, seq_len, d_model)
        Returns:
            Shape: (batch, seq_len, d_model)
        """
        if hidden_states.device.type == "mps" and hidden_states.dtype == torch.float32 and hasattr(torch.mps, "compile_shader"):
            return self._forward_metal(hidden_states)
        seq_len = hidden_states.shape[1]
        x, z = self.in_proj(hidden_states).chunk(2, dim=-1)  # (batch, seq_len, d_inner) each
        x = self.conv1d(x.transpose(1, 2))[..., :seq_len].transpose(1, 2)  # causal convolution
        x = F.silu(x)  # (batch, seq_len, d_inner)
        dt, B, C = self.x_proj(x).split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))  # (batch, seq_len, d_inner); dt_proj.bias is mamba-ssm's delta_bias
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        y = selective_scan(x, dt, A, B, C)  # (batch, seq_len, d_inner)
        y = y + x * self.D
        y = y * F.silu(z)
        return self.out_proj(y)  # (batch, seq_len, d_model)

    def _forward_metal(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """`forward` on MPS: the per-channel steps between the projections run as two Metal kernels,
        which read the projection outputs in place, without transposes or intermediate tensors."""
        batch, seq_len, _ = hidden_states.shape
        library = _mamba_library(self.d_state, self.conv1d.kernel_size[0])
        threads = batch * self.d_inner
        xz = self.in_proj(hidden_states).contiguous()  # (batch, seq_len, 2 * d_inner): [x | z]
        u = torch.empty(batch, seq_len, self.d_inner, device=xz.device)  # silu(causal_conv(x))
        library.causal_conv_silu(xz, self.conv1d.weight.reshape(self.d_inner, -1).contiguous(),
                                 self.conv1d.bias.contiguous(), u, seq_len, self.d_inner, threads=threads)
        x_dbl = self.x_proj(u).contiguous()  # (batch, seq_len, dt_rank + 2 * d_state): [dt | B | C]
        dt = self.dt_proj(x_dbl[..., :self.dt_rank]).contiguous()  # before softplus
        A = -torch.exp(self.A_log.float()).contiguous()  # (d_inner, d_state)
        y = torch.empty_like(u)  # (scan + D u) * silu(z)
        library.selective_scan(u, dt, A, x_dbl, self.D.float().contiguous(), xz, y,
                               seq_len, self.d_inner, self.dt_rank, threads=threads)
        return self.out_proj(y)  # (batch, seq_len, d_model)


def selective_scan(u: torch.Tensor, delta: torch.Tensor, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
    """Selective scan h_t = exp(delta_t A) h_{t-1} + delta_t B_t u_t, y_t = C_t h_t (the
    `selective_scan_fn` of mamba-ssm without D, z and delta_bias, which `Mamba` applies itself),
    in plain PyTorch (any device).

    The recurrence runs step by step; the per-step terms are precomputed for a block of steps at
    a time, sized to keep each (batch, steps, d_inner, d_state) tensor around 32M elements.

    Args:
        u: Input. Shape: (batch, seq_len, d_inner)
        delta: Step sizes (after softplus). Shape: (batch, seq_len, d_inner)
        A: State matrix (diagonal). Shape: (d_inner, d_state)
        B: Input matrix. Shape: (batch, seq_len, d_state)
        C: Output matrix. Shape: (batch, seq_len, d_state)
    Returns:
        y. Shape: (batch, seq_len, d_inner)
    """
    batch, seq_len, d_inner = u.shape
    d_state = A.shape[1]
    block = max(1, (1 << 25) // (batch * d_inner * d_state))
    h = u.new_zeros(batch, d_inner, d_state)
    ys = []
    for start in range(0, seq_len, block):
        stop = min(start + block, seq_len)
        dt = delta[:, start:stop]  # (batch, steps, d_inner)
        dA = torch.exp(dt.unsqueeze(-1) * A)  # (batch, steps, d_inner, d_state)
        dBu = (dt * u[:, start:stop]).unsqueeze(-1) * B[:, start:stop].unsqueeze(2)  # (batch, steps, d_inner, d_state)
        hs = []
        for t in range(stop - start):
            h = torch.addcmul(dBu[:, t], dA[:, t], h)  # (batch, d_inner, d_state)
            hs.append(h)
        ys.append(torch.einsum("bsdn,bsn->bsd", torch.stack(hs, dim=1), C[:, start:stop]))
    return torch.cat(ys, dim=1)


# One thread per (batch, channel) walks the sequence with its convolution window and its d_state
# states in registers, so neither the (batch, seq_len, d_inner, d_state) states nor the
# intermediate per-channel tensors are written to memory. Rows are (batch, time) pairs.
_MAMBA_SHADER = """
#include <metal_stdlib>
using namespace metal;

// u = silu(bias + causal depth-wise convolution of x), x = first half of xz
kernel void causal_conv_silu(
    device const float* xz    [[buffer(0)]],  // (rows, 2 * d_inner): [x | z]
    device const float* w     [[buffer(1)]],  // (d_inner, D_CONV)
    device const float* bias  [[buffer(2)]],  // (d_inner)
    device float* u           [[buffer(3)]],  // (rows, d_inner)
    constant uint& seq_len    [[buffer(4)]],
    constant uint& d_inner    [[buffer(5)]],
    uint idx [[thread_position_in_grid]])
{
    const uint b = idx / d_inner;
    const uint d = idx % d_inner;
    float k[D_CONV];
    float window[D_CONV];  // window[D_CONV - 1] is x_t, window[0] is x_{t - D_CONV + 1}
    for (uint i = 0; i < D_CONV; ++i) {
        k[i] = w[d * D_CONV + i];
        window[i] = 0.0f;
    }
    for (uint t = 0; t < seq_len; ++t) {
        const uint row = b * seq_len + t;
        for (uint i = 0; i + 1 < D_CONV; ++i) {
            window[i] = window[i + 1];
        }
        window[D_CONV - 1] = xz[row * 2 * d_inner + d];
        float acc = bias[d];
        for (uint i = 0; i < D_CONV; ++i) {
            acc += k[i] * window[i];
        }
        u[row * d_inner + d] = acc / (1.0f + precise::exp(-acc));
    }
}

// y = (C_t h_t + D u_t) * silu(z_t), h_t = exp(dt_t A) h_{t-1} + dt_t B_t u_t, dt_t = softplus(dt_raw_t)
kernel void selective_scan(
    device const float* u      [[buffer(0)]],  // (rows, d_inner)
    device const float* dt_raw [[buffer(1)]],  // (rows, d_inner), before softplus
    device const float* A      [[buffer(2)]],  // (d_inner, D_STATE)
    device const float* x_dbl  [[buffer(3)]],  // (rows, dt_rank + 2 * D_STATE): [dt | B | C]
    device const float* Dskip  [[buffer(4)]],  // (d_inner)
    device const float* xz     [[buffer(5)]],  // (rows, 2 * d_inner): [x | z]
    device float* y            [[buffer(6)]],  // (rows, d_inner)
    constant uint& seq_len     [[buffer(7)]],
    constant uint& d_inner     [[buffer(8)]],
    constant uint& dt_rank     [[buffer(9)]],
    uint idx [[thread_position_in_grid]])
{
    const uint b = idx / d_inner;
    const uint d = idx % d_inner;
    const uint width = dt_rank + 2 * D_STATE;
    float a[D_STATE];
    float h[D_STATE];
    for (uint n = 0; n < D_STATE; ++n) {
        a[n] = A[d * D_STATE + n];
        h[n] = 0.0f;
    }
    const float skip = Dskip[d];
    for (uint t = 0; t < seq_len; ++t) {
        const uint row = b * seq_len + t;
        const float raw = dt_raw[row * d_inner + d];
        // softplus as in torch (threshold 20); exp(raw) where log(1 + exp(raw)) would lose precision
        const float dt = raw > 20.0f ? raw : (raw < -15.0f ? precise::exp(raw) : precise::log(1.0f + precise::exp(raw)));
        const float ut = u[row * d_inner + d];
        const float dt_u = dt * ut;
        device const float* Bt = x_dbl + row * width + dt_rank;
        device const float* Ct = Bt + D_STATE;
        float acc = 0.0f;
        for (uint n = 0; n < D_STATE; ++n) {
            h[n] = precise::exp(dt * a[n]) * h[n] + dt_u * Bt[n];
            acc += h[n] * Ct[n];
        }
        const float z = xz[row * 2 * d_inner + d_inner + d];
        y[row * d_inner + d] = (acc + skip * ut) * (z / (1.0f + precise::exp(-z)));
    }
}
"""
_metal_libraries: dict[tuple[int, int], Any] = {}


def _mamba_library(d_state: int, d_conv: int) -> Any:
    """The compiled Metal kernels for a (d_state, d_conv) configuration."""
    key = (d_state, d_conv)
    if key not in _metal_libraries:
        source = _MAMBA_SHADER.replace("D_STATE", str(d_state)).replace("D_CONV", str(d_conv))
        _metal_libraries[key] = torch.mps.compile_shader(source)
    return _metal_libraries[key]


# -----------------------------
# RE-USE network (models/mamba_block2_SEMamba.py)
# -----------------------------
class MambaBlock(nn.Module):
    """Bidirectional Mamba: one Mamba per direction, merged by a linear layer."""

    def __init__(self, d_model: int, cfg: dict) -> None:
        super().__init__()
        d_state, d_conv, expand = cfg["model_cfg"]["d_state"], cfg["model_cfg"]["d_conv"], cfg["model_cfg"]["expand"]
        self.forward_blocks = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.backward_blocks = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.output_proj = nn.Linear(2 * d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        out_fw = self.forward_blocks(x) + x
        x_flip = torch.flip(x, dims=[1])
        out_bw = torch.flip(self.backward_blocks(x_flip) + x_flip, dims=[1])
        out = self.output_proj(torch.cat([out_fw, out_bw], dim=-1))
        return self.norm(out)


class TFMambaBlock(nn.Module):
    """Mamba over time (per frequency bin), then over frequency (per frame)."""

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        hid_feature = cfg["model_cfg"]["hid_feature"]
        self.time_mamba = MambaBlock(d_model=hid_feature, cfg=cfg)
        self.freq_mamba = MambaBlock(d_model=hid_feature, cfg=cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T, F]
        b, c, t, f = x.size()
        x = x.permute(0, 3, 2, 1).contiguous().view(b * f, t, c)
        x = self.time_mamba(x) + x
        x = x.view(b, f, t, c).permute(0, 2, 1, 3).contiguous().view(b * t, f, c)
        x = self.freq_mamba(x) + x
        return x.view(b, t, f, c).permute(0, 3, 1, 2)


# -----------------------------
# Encoder and decoders (models/codec_module_time_d4.py)
# -----------------------------
def _padding_2d(kernel_size: tuple, dilation: tuple = (1, 1)) -> tuple:
    return (int((kernel_size[0] * dilation[0] - dilation[0]) / 2),
            int((kernel_size[1] * dilation[1] - dilation[1]) / 2))


class SPConvTranspose2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, r: int = 1) -> None:
        super().__init__()
        self.pad1 = nn.ConstantPad2d((1, 1, 0, 0), value=0.0)
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels * r, kernel_size=kernel_size, stride=(1, 1))
        self.r = r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(self.pad1(x))
        batch_size, nchannels, H, W = out.shape
        out = out.view((batch_size, self.r, nchannels // self.r, H, W)).permute(0, 2, 3, 4, 1)
        return out.contiguous().view((batch_size, nchannels // self.r, H, -1))


class DenseBlock(nn.Module):
    def __init__(self, cfg: dict, kernel_size: tuple = (3, 3), depth: int = 4) -> None:
        super().__init__()
        self.depth = depth
        hid = cfg["model_cfg"]["hid_feature"]
        self.dense_block = nn.ModuleList()
        for i in range(depth):
            dil = 2 ** i
            self.dense_block.append(nn.Sequential(
                nn.Conv2d(hid * (i + 1), hid, kernel_size, dilation=(dil, 1), padding=_padding_2d(kernel_size, (dil, 1))),
                nn.InstanceNorm2d(hid, affine=True),
                nn.PReLU(hid),
            ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = x
        for i in range(self.depth):
            x = self.dense_block[i](skip)
            skip = torch.cat([x, skip], dim=1)
        return x


class DenseEncoder(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        in_ch, hid = cfg["model_cfg"]["input_channel"], cfg["model_cfg"]["hid_feature"]
        self.dense_conv_1 = nn.Sequential(nn.Conv2d(in_ch, hid, (1, 1)), nn.InstanceNorm2d(hid, affine=True), nn.PReLU(hid))
        self.dense_block = DenseBlock(cfg, depth=4)
        self.dense_conv_2 = nn.Sequential(nn.Conv2d(hid, hid, (1, 3), stride=(4, 2)), nn.InstanceNorm2d(hid, affine=True), nn.PReLU(hid))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dense_conv_2(self.dense_block(self.dense_conv_1(x)))  # [B, hid, T/4, F/2]


class _Decoder(nn.Module):
    """Shared trunk of MagDecoder and PhaseDecoder: dense block and two sub-pixel upsamplings."""

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        hid = cfg["model_cfg"]["hid_feature"]
        self.dense_block = DenseBlock(cfg, depth=4)
        self.up_conv1 = nn.Sequential(SPConvTranspose2d(hid, hid, (1, 3), 2), nn.InstanceNorm2d(hid, affine=True), nn.PReLU(hid))
        self.up_conv2 = nn.Sequential(SPConvTranspose2d(hid, hid, (1, 3), 4), nn.InstanceNorm2d(hid, affine=True), nn.PReLU(hid))

    def trunk(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up_conv1(self.dense_block(x))
        return self.up_conv2(x.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)


class MagDecoder(_Decoder):
    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)
        self.final_conv = nn.Conv2d(cfg["model_cfg"]["hid_feature"], cfg["model_cfg"]["output_channel"], (1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.final_conv(self.trunk(x))


class PhaseDecoder(_Decoder):
    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)
        hid, out_ch = cfg["model_cfg"]["hid_feature"], cfg["model_cfg"]["output_channel"]
        self.phase_conv_r = nn.Conv2d(hid, out_ch, (1, 1))
        self.phase_conv_i = nn.Conv2d(hid, out_ch, (1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.trunk(x)
        return torch.atan2(self.phase_conv_i(x), self.phase_conv_r(x))


# -----------------------------
# SEMamba (models/generator_SEMamba_time_d4.py)
# -----------------------------
class SEMamba(nn.Module):
    """RE-USE network: compressed magnitude and phase in, enhanced magnitude and phase out."""

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.dense_encoder = DenseEncoder(cfg)
        self.TSMamba = nn.ModuleList([TFMambaBlock(cfg) for _ in range(cfg["model_cfg"]["num_tfmamba"])])
        self.mask_decoder = MagDecoder(cfg)
        self.phase_decoder = PhaseDecoder(cfg)

    def forward(self, noisy_mag: torch.Tensor, noisy_pha: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            noisy_mag: Compressed magnitude. Shape: (B, F, T)
            noisy_pha: Phase. Shape: (B, F, T)
        Returns:
            Enhanced compressed magnitude and phase. Shape: (B, F, T) each
        """
        num_freqs, num_frames = noisy_mag.shape[1:]
        x = torch.stack((noisy_mag, noisy_pha), dim=1).transpose(2, 3)  # [B, 2, T, F]
        # Two zero frames and two zero bins, as in the original ("prevent unpredictable errors").
        x = F.pad(x, (0, 2, 0, 2))  # [B, 2, T + 2, F + 2]
        x = self.dense_encoder(x)
        for block in self.TSMamba:
            x = block(x)
        mag = rearrange(self.mask_decoder(x), "b c t f -> b f t c").squeeze(-1)[:, :num_freqs, :num_frames]
        pha = rearrange(self.phase_decoder(x), "b c t f -> b f t c").squeeze(-1)[:, :num_freqs, :num_frames]
        return mag, pha


# -----------------------------
# STFT helpers (models/stfts.py), for compress_factor = relu_log1p
# -----------------------------
def _mag_phase_stft(y: torch.Tensor, n_fft: int, hop: int, win: int) -> tuple[torch.Tensor, torch.Tensor]:
    spec = torch.stft(y, n_fft, hop_length=hop, win_length=win, window=torch.hann_window(win, device=y.device),
                      center=True, pad_mode="reflect", normalized=False, return_complex=True)
    return torch.log1p(spec.abs()), torch.angle(spec)


def _mag_phase_istft(mag: torch.Tensor, pha: torch.Tensor, n_fft: int, hop: int, win: int) -> torch.Tensor:
    mag = torch.expm1(F.relu(mag))
    spec = torch.complex(mag * torch.cos(pha), mag * torch.sin(pha))
    return torch.istft(spec, n_fft, hop_length=hop, win_length=win, window=torch.hann_window(win, device=mag.device), center=True)


def _make_even(value: float) -> int:
    rounded = round(value)
    return rounded if rounded % 2 == 0 else rounded + 1


# -----------------------------
# Enhancer
# -----------------------------
class ReUSE:
    """RE-USE speech enhancement at the input's own sample rate (8-48 kHz).

    Args:
        model_dir: Local directory with `config.json` and `model.safetensors`, or a Hugging Face
            repo id (downloaded to the Hugging Face cache).
        device: Torch device.
    """

    def __init__(self, model_dir: str = REPO_ID, device: str | torch.device = "cpu") -> None:
        from safetensors.torch import load_file

        config_path, weights_path = _resolve_files(model_dir)
        with open(config_path, encoding="utf-8") as f:
            self.cfg = json.load(f)
        if self.cfg["model_cfg"]["compress_factor"] != "relu_log1p":
            raise ValueError(f"Unsupported compress_factor: {self.cfg['model_cfg']['compress_factor']}")
        self.device = torch.device(device)
        self.model = SEMamba(self.cfg)
        self.model.load_state_dict(load_file(weights_path), strict=True)
        self.model.to(self.device).eval()

    @torch.no_grad()
    def __call__(self, wav: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Enhance a mono waveform in one pass. RE-USE is trained on clips under 10 s; for
        recordings much longer than 30 s, its model card recommends chunked processing.

        Args:
            wav: Waveform. Shape: (1, num_samples)
            sample_rate: Sample rate of `wav`, 8000 to 48000.
        Returns:
            Enhanced waveform at `sample_rate`. Shape: (1, num_samples)
        """
        stft_cfg = self.cfg["stft_cfg"]
        scale = sample_rate / stft_cfg["sampling_rate"]  # the STFT is scaled with the sample rate
        n_fft = _make_even(stft_cfg["n_fft"] * scale)
        hop = _make_even(stft_cfg["hop_size"] * scale)
        win = _make_even(stft_cfg["win_size"] * scale)

        # The STFT and its inverse are cheap and run on the CPU (MPS complex support is partial);
        # the network runs on `self.device`.
        noisy = wav.detach().cpu().float()  # (1, num_samples)
        mag, pha = _mag_phase_stft(noisy, n_fft, hop, win)  # (1, F, T) each
        mag, pha = self.model(mag.to(self.device), pha.to(self.device))
        mag, pha = mag.cpu(), pha.cpu()

        # As in RE-USE inference.py: frames where most bins are zero are silenced entirely
        # ("strange sweep artifact").
        zero_portion = (torch.expm1(F.relu(mag)) == 0).float().mean(dim=1)  # (1, T)
        mag[:, :, zero_portion[0] > 0.5] = 0

        enhanced = _mag_phase_istft(mag, pha, n_fft, hop, win)  # (1, ~num_samples)
        num_samples = noisy.shape[1]
        if enhanced.shape[1] >= num_samples:
            return enhanced[:, :num_samples]
        return F.pad(enhanced, (0, num_samples - enhanced.shape[1]), value=1e-8)


def _resolve_files(model_dir: str) -> tuple[str, str]:
    if os.path.isdir(model_dir):
        return os.path.join(model_dir, "config.json"), os.path.join(model_dir, "model.safetensors")
    from huggingface_hub import hf_hub_download

    return hf_hub_download(model_dir, "config.json"), hf_hub_download(model_dir, "model.safetensors")

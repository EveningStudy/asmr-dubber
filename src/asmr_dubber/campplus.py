"""Apache-2.0 CAM++ (Context-Aware Masking TDNN) speaker embedding extractor.

Adapted from 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker,
licensed under Apache-2.0) with a standalone Kaldi Povey-window 80-bin Mel
filterbank so ``campplus_cn_common.bin`` runs without ``torchaudio`` or GPU VRAM.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np

from .platforms import portable_home


def find_campplus_checkpoint(explicit_dir: Path | None = None) -> Path | None:
    """Locate locally installed ``campplus_cn_common.bin`` without network access."""
    candidates: list[Path] = []
    if explicit_dir is not None:
        candidates.append(explicit_dir / "campplus_cn_common.bin")
        candidates.append(explicit_dir / "hf_cache" / "campplus_cn_common.bin")
    home = portable_home()
    candidates.extend(
        [
            home / "models" / "campplus" / "campplus_cn_common.bin",
            home / "runtimes" / "index-tts" / "checkpoints" / "hf_cache" / "campplus_cn_common.bin",
            home
            / "runtimes"
            / "index-tts-2.5"
            / "checkpoints"
            / "hf_cache"
            / "campplus_cn_common.bin",
        ]
    )
    for path in candidates:
        if path.is_file():
            return path
    return None


def kaldi_fbank_80(waveform_16k: np.ndarray, sample_rate: int = 16_000) -> np.ndarray:
    """Compute CMN-normalized 80-bin Kaldi log-Mel filterbank features ``(T, 80)``."""
    sig = np.asarray(waveform_16k, dtype=np.float32).reshape(-1)
    # Kaldi scales normalized [-1, 1] float waveforms to 16-bit PCM range before log-energy,
    # though CMN cancels additive constants; scaling avoids float32 floor clipping on whispers.
    sig = sig * 32768.0
    frame_len = round(sample_rate * 0.025)
    frame_shift = round(sample_rate * 0.010)
    n_fft = 1 << (frame_len - 1).bit_length()
    if len(sig) < frame_len:
        sig = np.pad(sig, (0, frame_len - len(sig)))
    num_frames = max(1, 1 + (len(sig) - frame_len) // frame_shift)
    idx = np.arange(frame_len)[None, :] + frame_shift * np.arange(num_frames)[:, None]
    frames = sig[idx].astype(np.float64)
    # Remove DC offset per frame, apply 0.97 pre-emphasis, then Kaldi Povey window.
    frames = frames - np.mean(frames, axis=1, keepdims=True)
    pre = np.empty_like(frames)
    pre[:, 0] = frames[:, 0] * 0.03
    pre[:, 1:] = frames[:, 1:] - 0.97 * frames[:, :-1]
    n = np.arange(frame_len, dtype=np.float64)
    povey = np.power(0.5 - 0.5 * np.cos(2.0 * math.pi * n / max(1, frame_len - 1)), 0.85)
    power = np.abs(np.fft.rfft(pre * povey[None, :], n=n_fft, axis=1)) ** 2

    def _mel(f: np.ndarray | float) -> np.ndarray | float:
        return 1127.0 * np.log(1.0 + np.asarray(f) / 700.0)

    low_mel = float(_mel(20.0))
    high_mel = float(_mel(sample_rate * 0.5))
    mel_points = np.linspace(low_mel, high_mel, 82)
    bin_freqs = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
    bin_mels = np.asarray(_mel(bin_freqs), dtype=np.float64)
    fb = np.zeros((80, len(bin_freqs)), dtype=np.float64)
    for m in range(80):
        left, center, right = mel_points[m], mel_points[m + 1], mel_points[m + 2]
        up = (bin_mels - left) / max(center - left, 1e-9)
        down = (right - bin_mels) / max(right - center, 1e-9)
        fb[m] = np.maximum(0.0, np.minimum(up, down))
    # Kaldi zeroes Nyquist bin in filterbank integration.
    fb[:, -1] = 0.0
    feats = np.log(np.maximum(power @ fb.T, 1.1920929e-7))
    feats -= np.mean(feats, axis=0, keepdims=True)
    return feats.astype(np.float32)


def _build_campplus_module(feat_dim: int = 80, embedding_size: int = 192) -> Any:
    import torch
    import torch.nn.functional as F
    from torch import nn

    def _nonlinear(cfg: str, ch: int) -> nn.Sequential:
        seq = nn.Sequential()
        for name in cfg.split("-"):
            if name == "relu":
                seq.add_module("relu", nn.ReLU(inplace=True))
            elif name == "prelu":
                seq.add_module("prelu", nn.PReLU(ch))
            elif name == "batchnorm":
                seq.add_module("batchnorm", nn.BatchNorm1d(ch))
            elif name == "batchnorm_":
                seq.add_module("batchnorm", nn.BatchNorm1d(ch, affine=False))
        return seq

    class _StatsPool(nn.Module):
        def forward(self, x: Any) -> Any:
            return torch.cat([x.mean(dim=-1), x.std(dim=-1, unbiased=True)], dim=-1)

    class _TDNNLayer(nn.Module):
        def __init__(
            self,
            in_ch: int,
            out_ch: int,
            k: int,
            stride: int = 1,
            padding: int = 0,
            dilation: int = 1,
            bias: bool = False,
            config_str: str = "batchnorm-relu",
        ) -> None:
            super().__init__()
            pad = (k - 1) // 2 * dilation if padding < 0 else padding
            self.linear = nn.Conv1d(
                in_ch, out_ch, k, stride=stride, padding=pad, dilation=dilation, bias=bias
            )
            self.nonlinear = _nonlinear(config_str, out_ch)

        def forward(self, x: Any) -> Any:
            return self.nonlinear(self.linear(x))

    class _CAMLayer(nn.Module):
        def __init__(
            self,
            bn_ch: int,
            out_ch: int,
            k: int,
            stride: int,
            padding: int,
            dilation: int,
            bias: bool,
        ) -> None:
            super().__init__()
            self.linear_local = nn.Conv1d(
                bn_ch, out_ch, k, stride=stride, padding=padding, dilation=dilation, bias=bias
            )
            self.linear1 = nn.Conv1d(bn_ch, bn_ch // 2, 1)
            self.relu = nn.ReLU(inplace=True)
            self.linear2 = nn.Conv1d(bn_ch // 2, out_ch, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x: Any) -> Any:
            y = self.linear_local(x)
            seg = F.avg_pool1d(x, kernel_size=100, stride=100, ceil_mode=True)
            shape = seg.shape
            seg = seg.unsqueeze(-1).expand(*shape, 100).reshape(*shape[:-1], -1)[..., : x.shape[-1]]
            ctx = self.relu(self.linear1(x.mean(-1, keepdim=True) + seg))
            return y * self.sigmoid(self.linear2(ctx))

    class _CAMDenseTDNNLayer(nn.Module):
        def __init__(self, in_ch: int, out_ch: int, bn_ch: int, k: int, dilation: int) -> None:
            super().__init__()
            pad = (k - 1) // 2 * dilation
            self.nonlinear1 = _nonlinear("batchnorm-relu", in_ch)
            self.linear1 = nn.Conv1d(in_ch, bn_ch, 1, bias=False)
            self.nonlinear2 = _nonlinear("batchnorm-relu", bn_ch)
            self.cam_layer = _CAMLayer(bn_ch, out_ch, k, 1, pad, dilation, False)

        def forward(self, x: Any) -> Any:
            h = self.linear1(self.nonlinear1(x))
            return self.cam_layer(self.nonlinear2(h))

    class _CAMDenseTDNNBlock(nn.ModuleList):
        def __init__(
            self, n_layers: int, in_ch: int, out_ch: int, bn_ch: int, k: int, dil: int
        ) -> None:
            super().__init__()
            for i in range(n_layers):
                self.add_module(
                    f"tdnnd{i + 1}",
                    _CAMDenseTDNNLayer(in_ch + i * out_ch, out_ch, bn_ch, k, dil),
                )

        def forward(self, x: Any) -> Any:
            for layer in self:
                x = torch.cat([x, layer(x)], dim=1)
            return x

    class _TransitLayer(nn.Module):
        def __init__(self, in_ch: int, out_ch: int) -> None:
            super().__init__()
            self.nonlinear = _nonlinear("batchnorm-relu", in_ch)
            self.linear = nn.Conv1d(in_ch, out_ch, 1, bias=False)

        def forward(self, x: Any) -> Any:
            return self.linear(self.nonlinear(x))

    class _DenseLayer(nn.Module):
        def __init__(self, in_ch: int, out_ch: int) -> None:
            super().__init__()
            self.linear = nn.Conv1d(in_ch, out_ch, 1, bias=False)
            self.nonlinear = _nonlinear("batchnorm_", out_ch)

        def forward(self, x: Any) -> Any:
            x = self.linear(x.unsqueeze(-1)).squeeze(-1) if x.ndim == 2 else self.linear(x)
            return self.nonlinear(x)

    class _BasicResBlock(nn.Module):
        def __init__(self, in_p: int, planes: int, stride: int = 1) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(in_p, planes, 3, stride=(stride, 1), padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)
            self.shortcut = (
                nn.Sequential(
                    nn.Conv2d(in_p, planes, 1, stride=(stride, 1), bias=False),
                    nn.BatchNorm2d(planes),
                )
                if stride != 1 or in_p != planes
                else nn.Sequential()
            )

        def forward(self, x: Any) -> Any:
            out = F.relu(self.bn1(self.conv1(x)))
            return F.relu(self.bn2(self.conv2(out)) + self.shortcut(x))

    class _FCM(nn.Module):
        def __init__(self, m_ch: int = 32) -> None:
            super().__init__()
            self.in_p = m_ch
            self.conv1 = nn.Conv2d(1, m_ch, 3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(m_ch)
            self.layer1 = nn.Sequential(
                _BasicResBlock(m_ch, m_ch, 2), _BasicResBlock(m_ch, m_ch, 1)
            )
            self.layer2 = nn.Sequential(
                _BasicResBlock(m_ch, m_ch, 2), _BasicResBlock(m_ch, m_ch, 1)
            )
            self.conv2 = nn.Conv2d(m_ch, m_ch, 3, stride=(2, 1), padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(m_ch)
            self.out_channels = m_ch * (feat_dim // 8)

        def forward(self, x: Any) -> Any:
            out = F.relu(self.bn1(self.conv1(x.unsqueeze(1))))
            out = F.relu(self.bn2(self.conv2(self.layer2(self.layer1(out)))))
            return out.reshape(out.shape[0], out.shape[1] * out.shape[2], out.shape[3])

    class _CAMPPlus(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.head = _FCM()
            ch = self.head.out_channels
            self.xvector = nn.Sequential(
                OrderedDict([("tdnn", _TDNNLayer(ch, 128, 5, stride=2, padding=-1))])
            )
            ch = 128
            for idx, (nl, ks, dil) in enumerate(
                zip((12, 24, 16), (3, 3, 3), (1, 2, 2), strict=True)
            ):
                self.xvector.add_module(
                    f"block{idx + 1}", _CAMDenseTDNNBlock(nl, ch, 32, 128, ks, dil)
                )
                ch += nl * 32
                self.xvector.add_module(f"transit{idx + 1}", _TransitLayer(ch, ch // 2))
                ch //= 2
            self.xvector.add_module("out_nonlinear", _nonlinear("batchnorm-relu", ch))
            self.xvector.add_module("stats", _StatsPool())
            self.xvector.add_module("dense", _DenseLayer(ch * 2, embedding_size))

        def forward(self, x: Any) -> Any:
            return self.xvector(self.head(x.permute(0, 2, 1)))

    return _CAMPPlus()


_CACHED_CAMPPLUS: tuple[Path, Any] | None = None


def extract_campplus_embeddings(
    clips_16k: list[np.ndarray],
    checkpoint_path: Path | None = None,
) -> np.ndarray | None:
    """Return L2-normalized ``(N, 192)`` CAM++ embeddings on CPU, or ``None`` if unavailable."""
    global _CACHED_CAMPPLUS
    ckpt = checkpoint_path or find_campplus_checkpoint()
    if ckpt is None or not clips_16k:
        return None
    try:
        import torch
    except ImportError:
        return None

    if _CACHED_CAMPPLUS is None or _CACHED_CAMPPLUS[0] != ckpt:
        model = _build_campplus_module(feat_dim=80, embedding_size=192)
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()
        _CACHED_CAMPPLUS = (ckpt, model)
    else:
        model = _CACHED_CAMPPLUS[1]

    vectors: list[np.ndarray] = []
    with torch.inference_mode():
        for clip in clips_16k:
            feat = kaldi_fbank_80(clip, 16_000)
            if feat.shape[0] < 16:
                feat = np.pad(feat, ((0, 16 - feat.shape[0]), (0, 0)), mode="edge")
            emb = model(torch.from_numpy(feat).unsqueeze(0)).squeeze(0).cpu().numpy()
            norm = float(np.linalg.norm(emb))
            vectors.append(emb / max(norm, 1e-9))
    return np.stack(vectors, axis=0).astype(np.float32)

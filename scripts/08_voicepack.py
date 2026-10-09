#!/usr/bin/env python3
"""Phase 8.4: voicepacks uf_rasa / um_rasa (plan 10.4), two-checkpoint mode.

    python scripts/08_voicepack.py --style-ckpt work/best_s1.pth --predictor-ckpt work/best_s2.pth \
        --audio-dir data/audio --refs data/voicepack_refs.json --bandwidth data/bandwidth.json --out export/voices

* style_encoder from the Stage-1 checkpoint (Stage 2 can degrade it), predictor_encoder from
  the Stage-2 checkpoint (untrained after Stage 1 alone -> falls back to style_encoder).
* Reference clips: that speaker's neutral-style, full-bandwidth clips, top-N by DNSMOS.
* Output [510, 1, 256] float32 (+ .npy copies for the ONNX runner).
* --length-rows: experimental length-dependent rows (plan 10.4 optional); default off.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

SPEAKER_NAMES = {"0": "uf_rasa", "1": "um_rasa"}


# ── StyleEncoder (standalone copy of StyleTTS2 models.py, Kokoro dims) ───────────
class LearnedDownSample(nn.Module):
    def __init__(self, layer_type, dim_in):
        super().__init__()
        self.layer_type = layer_type
        if layer_type == "none":
            self.conv = nn.Identity()
        elif layer_type == "timepreserve":
            self.conv = spectral_norm(nn.Conv2d(dim_in, dim_in, kernel_size=(3, 1), stride=(2, 1), groups=dim_in, padding=(1, 0)))
        elif layer_type == "half":
            self.conv = spectral_norm(nn.Conv2d(dim_in, dim_in, kernel_size=(3, 3), stride=(2, 2), groups=dim_in, padding=1))

    def forward(self, x):
        return self.conv(x)


class DownSample(nn.Module):
    def __init__(self, layer_type):
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x):
        if self.layer_type == "none":
            return x
        if self.layer_type == "timepreserve":
            return F.avg_pool2d(x, (2, 1))
        if x.shape[-1] % 2 != 0:
            x = torch.cat([x, x[..., -1].unsqueeze(-1)], dim=-1)
        return F.avg_pool2d(x, 2)


class ResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2), normalize=False, downsample="none"):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = DownSample(downsample)
        self.downsample_res = LearnedDownSample(downsample, dim_in)
        self.learned_sc = dim_in != dim_out
        self.conv1 = spectral_norm(nn.Conv2d(dim_in, dim_in, 3, 1, 1))
        self.conv2 = spectral_norm(nn.Conv2d(dim_in, dim_out, 3, 1, 1))
        if normalize:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = spectral_norm(nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False))

    def forward(self, x):
        sc = self.conv1x1(x) if self.learned_sc else x
        sc = self.downsample(sc)
        h = x
        if self.normalize:
            h = self.norm1(h)
        h = self.conv1(self.actv(h))
        h = self.downsample_res(h)
        if self.normalize:
            h = self.norm2(h)
        h = self.conv2(self.actv(h))
        return (sc + h) / math.sqrt(2)


class StyleEncoder(nn.Module):
    def __init__(self, dim_in=64, style_dim=128, max_conv_dim=512):
        super().__init__()
        blocks = [spectral_norm(nn.Conv2d(1, dim_in, 3, 1, 1))]
        for _ in range(4):
            dim_out = min(dim_in * 2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample="half")]
            dim_in = dim_out
        blocks += [nn.LeakyReLU(0.2), spectral_norm(nn.Conv2d(dim_out, dim_out, 5, 1, 0)), nn.AdaptiveAvgPool2d(1), nn.LeakyReLU(0.2)]
        self.shared = nn.Sequential(*blocks)
        self.unshared = nn.Linear(dim_out, style_dim)

    def forward(self, x):
        h = self.shared(x)
        return self.unshared(h.view(h.size(0), -1))


def _strip(sd):
    return {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}


def load_encoders(style_ckpt: str, predictor_ckpt: str | None, device: str):
    se = StyleEncoder()
    pe = StyleEncoder()
    s_net = torch.load(style_ckpt, map_location="cpu", weights_only=False)["net"]
    se.load_state_dict(_strip(s_net["style_encoder"]))
    trained = False
    if predictor_ckpt:
        p_net = torch.load(predictor_ckpt, map_location="cpu", weights_only=False)["net"]
        try:
            pe.load_state_dict(_strip(p_net["predictor_encoder"]))
            with torch.no_grad():
                trained = pe(torch.randn(1, 1, 80, 200)).norm().item() < 1e3
        except Exception:
            trained = False
    if not trained:
        print("predictor_encoder untrained or missing -> using style_encoder for both halves (Stage-1 mode)")
        pe.load_state_dict(_strip(s_net["style_encoder"]))
    return se.to(device).eval(), pe.to(device).eval(), trained


def mel_of(path: str, device: str):
    import soundfile as sf
    import torchaudio
    y, sr = sf.read(path, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    w = torch.from_numpy(y).unsqueeze(0)
    if sr != 24000:
        w = torchaudio.functional.resample(w, sr, 24000)
    mel = torchaudio.transforms.MelSpectrogram(sample_rate=24000, n_fft=2048, win_length=1200, hop_length=300, n_mels=80)(w)
    return ((torch.log(1e-5 + mel) - (-4)) / 4).to(device)


def pick_refs(refs: list[str], audio_dir: str, n: int, rank_by_mos: bool) -> list[str]:
    refs = [r for r in refs if os.path.exists(os.path.join(audio_dir, r))]
    rng = random.Random(42)
    if not rank_by_mos or len(refs) <= n:
        rng.shuffle(refs)
        return refs[:n]
    from eval.metrics import dnsmos
    import soundfile as sf
    cand = rng.sample(refs, min(len(refs), n * 3))
    scored = []
    for r in cand:
        y, sr = sf.read(os.path.join(audio_dir, r), dtype="float32")
        scored.append((dnsmos(y, sr), r))
    scored.sort(reverse=True)
    return [r for _, r in scored[:n]]


def build_voicepack(se, pe, paths: list[str], device: str, length_rows: bool = False, phon_lens: dict | None = None) -> torch.Tensor:
    acoustic, prosodic, lens = [], [], []
    with torch.no_grad():
        for p in paths:
            mel = mel_of(p, device)
            if mel.shape[-1] < 80:
                continue
            x = mel.unsqueeze(1)
            acoustic.append(se(x).cpu())
            prosodic.append(pe(x).cpu())
            lens.append(phon_lens.get(os.path.basename(p), 0) if phon_lens else 0)
    assert acoustic, "no usable reference clips"
    A = torch.cat(acoustic)
    P = torch.cat(prosodic)
    combined = torch.cat([A.mean(0), P.mean(0)])
    vp = combined.unsqueeze(0).unsqueeze(0).expand(510, 1, 256).clone()
    if length_rows and phon_lens:
        L = torch.tensor(lens, dtype=torch.float32)
        for i in range(510):
            w = torch.exp(-0.5 * ((L - (i + 1)) / 40.0) ** 2)
            if w.sum() > 1e-3:
                w = w / w.sum()
                vp[i, 0, :128] = (A * w[:, None]).sum(0)
                vp[i, 0, 128:] = (P * w[:, None]).sum(0)
    print(f"  acoustic norm {A.mean(0).norm():.3f} prosodic norm {P.mean(0).norm():.3f} from {len(acoustic)} clips")
    return vp.float()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--style-ckpt", required=True)
    ap.add_argument("--predictor-ckpt", default=None)
    ap.add_argument("--audio-dir", required=True)
    ap.add_argument("--refs", required=True, help="voicepack_refs.json: {speaker_id: [relative wav paths]}")
    ap.add_argument("--bandwidth", default=None, help="bandwidth.json from the audit; band-limited speakers are refused")
    ap.add_argument("--out", required=True)
    ap.add_argument("--num", type=int, default=200)
    ap.add_argument("--no-mos", action="store_true")
    ap.add_argument("--length-rows", action="store_true")
    ap.add_argument("--train-list", default=None, help="for --length-rows: phoneme lengths per clip")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.out, exist_ok=True)
    refs = json.load(open(a.refs, encoding="utf-8"))
    bw = json.load(open(a.bandwidth)) if a.bandwidth and os.path.exists(a.bandwidth) else {}
    se, pe, trained = load_encoders(a.style_ckpt, a.predictor_ckpt, device)
    phon_lens = None
    if a.length_rows and a.train_list:
        phon_lens = {}
        for ln in open(a.train_list, encoding="utf-8"):
            parts = ln.rstrip("\n").split("|")
            if len(parts) >= 2:
                phon_lens[os.path.basename(parts[0])] = len(parts[1])
    report = {"predictor_encoder_trained": trained, "voices": {}}
    for spk, name in SPEAKER_NAMES.items():
        if spk not in refs or not refs[spk]:
            print(f"speaker {spk} ({name}): no reference clips; skipped")
            continue
        if bw.get(spk, {}).get("band_limited"):
            raise SystemExit(f"speaker {spk} is band-limited (rolloff {bw[spk]['rolloff99_hz']} Hz); it must not be a voicepack reference")
        paths = [os.path.join(a.audio_dir, r) for r in pick_refs(refs[spk], a.audio_dir, a.num, not a.no_mos)]
        print(f"{name}: {len(paths)} reference clips")
        vp = build_voicepack(se, pe, paths, device, a.length_rows, phon_lens)
        assert tuple(vp.shape) == (510, 1, 256)
        torch.save(vp, os.path.join(a.out, f"{name}.pt"))
        np.save(os.path.join(a.out, f"{name}.npy"), vp.numpy())
        report["voices"][name] = {"clips": len(paths), "acoustic_norm": float(vp[0, 0, :128].norm()), "prosodic_norm": float(vp[0, 0, 128:].norm())}
    with open(os.path.join(a.out, "voicepack_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

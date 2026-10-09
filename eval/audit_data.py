#!/usr/bin/env python3
"""Per-speaker bandwidth (and optional DNSMOS) audit (plan 3.2.7, pitfall 6).

rolloff99 = frequency below which 99 % of the energy lies (true 24 kHz speech: > ~9 kHz;
16 kHz-native audio: ~7.4 kHz). The 95 % roll-off of normal speech is only ~4-6 kHz and must
NOT be used for this test.

Ported from Kokoro-Indic-Fine-Tuning/eval/audit_data.py. 16 kHz-native audio shipped as
24 kHz has no energy above 8 kHz; a model conditioned on it as the style reference sounds
dull/metallic. Such a speaker may still be used for TRAINING but never as the VOICEPACK
reference.

    python eval/audit_data.py --audio-dir data/audio --out data/bandwidth.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys

import numpy as np
import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from eval.metrics import spectral_rolloff, hf_energy_ratio, dnsmos  # noqa: E402


def audit_speaker(clips: list[str], per_spk: int = 30, seed: int = 0, with_mos: bool = False) -> dict:
    rng = random.Random(seed)
    sample = rng.sample(clips, min(per_spk, len(clips)))
    roll, hfr, mos = [], [], []
    for p in sample:
        try:
            y, sr = sf.read(p, dtype="float32")
            if y.ndim > 1:
                y = y.mean(axis=1)
            if len(y) < sr * 0.5:
                continue
            roll.append(spectral_rolloff(y, sr, 0.99))
            hfr.append(hf_energy_ratio(y, sr, 7000.0))
            if with_mos:
                mos.append(dnsmos(y, sr))
        except Exception:
            continue
    r95 = float(np.nanmean(roll)) if roll else float("nan")
    return {
        "n_sampled": len(roll),
        "n_clips": len(clips),
        "rolloff99_hz": round(r95, 1),
        "hf_ratio_pct_above_7k": round(100 * float(np.nanmean(hfr)), 3) if hfr else float("nan"),
        "dnsmos": round(float(np.nanmean(mos)), 3) if mos else None,
        "band_limited": bool(r95 < 8000.0) if roll else None,
    }


def audit_dir(audio_dir: str, per_spk: int = 30, with_mos: bool = False) -> dict:
    out = {}
    for spk in sorted(os.listdir(audio_dir)):
        d = os.path.join(audio_dir, spk)
        if not os.path.isdir(d):
            continue
        clips = sorted(glob.glob(os.path.join(d, "*.wav")))
        if clips:
            out[spk] = audit_speaker(clips, per_spk, with_mos=with_mos)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-dir", required=True, help="dir containing <speaker>/*.wav")
    ap.add_argument("--per-spk", type=int, default=30)
    ap.add_argument("--mos", action="store_true", help="also compute DNSMOS (slow)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = audit_dir(a.audio_dir, a.per_spk, a.mos)
    print(f"{'speaker':<10}{'clips':>7}{'roll99kHz':>11}{'HF%>7k':>9}  verdict")
    for spk, v in res.items():
        flag = "BAND-LIMITED (never a voicepack reference)" if v["band_limited"] else "ok"
        print(f"{spk:<10}{v['n_clips']:>7}{v['rolloff99_hz']/1000:>11.2f}{v['hf_ratio_pct_above_7k']:>9.3f}  {flag}")
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()

"""Naturalness, speaker-similarity and signal-quality metrics (plan 9.3-9.5 and the gates).

All functions take 24 kHz float32 numpy audio unless stated otherwise. Heavy models are
loaded lazily and cached. Every metric degrades gracefully (returns NaN) if its optional
dependency is missing, so the evaluation never crashes a Kaggle session.
"""
from __future__ import annotations

import math
import time
from functools import lru_cache
from typing import Optional

import numpy as np

SR = 24000


def _resample(y: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return y.astype(np.float32)
    try:
        import soxr  # type: ignore
        return soxr.resample(y, sr_in, sr_out, quality="HQ").astype(np.float32)
    except Exception:
        import librosa
        return librosa.resample(y.astype(np.float32), orig_sr=sr_in, target_sr=sr_out)


# ---------------------------------------------------------------------------
# naturalness
# ---------------------------------------------------------------------------
def dnsmos(y: np.ndarray, sr: int = SR) -> float:
    try:
        from speechmos import dnsmos as _d  # type: ignore
        y16 = _resample(y, sr, 16000)
        y16 = np.clip(y16, -1, 1)
        return float(_d.run(y16, sr=16000)["ovrl_mos"])
    except Exception:
        return float("nan")


@lru_cache(maxsize=1)
def _utmos_model():
    try:
        import torch
        model = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
        return model.eval()
    except Exception:
        return None


def utmos(y: np.ndarray, sr: int = SR) -> float:
    """UTMOS22 strong learner via tarepan/SpeechMOS (torch.hub). NaN if unavailable."""
    m = _utmos_model()
    if m is None:
        return float("nan")
    try:
        import torch
        y16 = torch.from_numpy(_resample(y, sr, 16000)).unsqueeze(0)
        with torch.no_grad():
            return float(m(y16, 16000).item())
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# speaker similarity
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _ecapa():
    try:
        from speechbrain.inference.speaker import EncoderClassifier  # type: ignore
        return EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": "cpu"})
    except Exception:
        try:
            from speechbrain.pretrained import EncoderClassifier  # type: ignore
            return EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": "cpu"})
        except Exception:
            return None


def ecapa_embedding(y: np.ndarray, sr: int = SR) -> Optional[np.ndarray]:
    enc = _ecapa()
    if enc is None:
        return None
    import torch
    y16 = torch.from_numpy(_resample(y, sr, 16000)).unsqueeze(0)
    with torch.no_grad():
        e = enc.encode_batch(y16).squeeze().cpu().numpy()
    return e / (np.linalg.norm(e) + 1e-9)


def ecapa_similarity(a: np.ndarray, b: np.ndarray, sr: int = SR) -> float:
    ea, eb = ecapa_embedding(a, sr), ecapa_embedding(b, sr)
    if ea is None or eb is None:
        return float("nan")
    return float(np.dot(ea, eb))


# ---------------------------------------------------------------------------
# signal checks
# ---------------------------------------------------------------------------
def spectral_rolloff(y: np.ndarray, sr: int = SR, percent: float = 0.95, n_fft: int = 2048) -> float:
    """Frequency (Hz) below which `percent` of the spectral energy lies (mean over frames)."""
    if len(y) < n_fft:
        return float("nan")
    hop = n_fft // 4
    win = np.hanning(n_fft)
    frames = range(0, len(y) - n_fft, hop)
    S = np.stack([np.abs(np.fft.rfft(y[i:i + n_fft] * win)) ** 2 for i in frames]) if len(y) > n_fft else np.abs(np.fft.rfft(y[:n_fft] * win))[None] ** 2
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    tot = S.sum(axis=1, keepdims=True) + 1e-12
    cum = np.cumsum(S, axis=1) / tot
    idx = (cum >= percent).argmax(axis=1)
    return float(np.mean(freqs[idx]))


def hf_energy_ratio(y: np.ndarray, sr: int = SR, cutoff: float = 8000.0, n_fft: int = 2048) -> float:
    if len(y) < n_fft:
        return float("nan")
    win = np.hanning(n_fft)
    S = np.abs(np.fft.rfft(y[: (len(y) // n_fft) * n_fft].reshape(-1, n_fft) * win, axis=1)) ** 2
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    return float(S[:, freqs > cutoff].sum() / (S.sum() + 1e-12))


def clipping_ratio(y: np.ndarray, thresh: float = 0.99) -> float:
    return float(np.mean(np.abs(y) >= thresh)) if len(y) else 0.0


def max_silence_seconds(y: np.ndarray, sr: int = SR, thresh_db: float = -45.0, frame_ms: float = 20.0) -> float:
    n = int(sr * frame_ms / 1000)
    if len(y) < n:
        return 0.0
    frames = y[: (len(y) // n) * n].reshape(-1, n)
    rms_db = 20 * np.log10(np.sqrt(np.mean(frames ** 2, axis=1)) + 1e-9)
    quiet = rms_db < thresh_db
    # ignore leading/trailing silence
    idx = np.where(~quiet)[0]
    if len(idx) == 0:
        return len(y) / sr
    quiet = quiet[idx[0]: idx[-1] + 1]
    best = run = 0
    for q in quiet:
        run = run + 1 if q else 0
        best = max(best, run)
    return best * frame_ms / 1000


def comb_cepstral_peak(y: np.ndarray, sr: int = SR, min_ms: float = 12.5, max_ms: float = 30.0, n_fft: int = 4096) -> float:
    """Height (in MAD units above the median) of the strongest cepstral peak between 12.5 and 30 ms
    (delays below the voice pitch period, which lives at 4-12 ms): a comb/echo artefact shows as a
    sharp peak there. Compare against real recordings; absolute thresholds are unreliable."""
    if len(y) < n_fft * 2:
        return float("nan")
    hop = n_fft // 2
    win = np.hanning(n_fft)
    peaks = []
    lo, hi = int(sr * min_ms / 1000), int(sr * max_ms / 1000)
    for i in range(0, len(y) - n_fft, hop):
        spec = np.log(np.abs(np.fft.rfft(y[i:i + n_fft] * win)) + 1e-9)
        cep = np.abs(np.fft.irfft(spec))
        seg = cep[lo:hi]
        med = np.median(seg) + 1e-9
        mad = np.median(np.abs(seg - np.median(seg))) + 1e-9
        peaks.append((seg.max() - med) / mad)
    return float(np.median(peaks)) if peaks else float("nan")


def real_time_factor(fn, text: str, repeats: int = 3) -> float:
    """RTF = synthesis time / audio duration for callable fn(text)->audio@24k."""
    t = time.perf_counter()
    dur = 0.0
    for _ in range(repeats):
        a = fn(text)
        dur += len(a) / SR
    return (time.perf_counter() - t) / max(dur, 1e-6)


__all__ = ["dnsmos", "utmos", "ecapa_similarity", "ecapa_embedding", "spectral_rolloff", "hf_energy_ratio",
           "clipping_ratio", "max_silence_seconds", "comb_cepstral_peak", "real_time_factor"]

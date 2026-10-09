#!/usr/bin/env python3
"""Phase 8.5: ONNX export (fp32 + fp16) with parity checks (plan 10.5).

    python scripts/09_onnx.py --config export/config.json --model export/lughaat-tts-82m.pth \
        --voice export/voices/uf_rasa.pt --out export/onnx

Based on kokoro/examples/export.py: KModelForONNX(KModel(..., disable_complex=True)), opset 17,
dynamic axes. Parity = waveform correlation >= 0.99 against PyTorch on 20 sentences; the fp16
model is kept only if it also passes. Dynamic INT8 is NOT shipped (it slows this vocoder down).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from eval.testsets import APPENDIX_A, APPENDIX_A2  # noqa: E402
from lughaat_tts.vocab import VOCAB  # noqa: E402


def _ids(ps: str) -> torch.LongTensor:
    ids = [VOCAB[c] for c in ps if c in VOCAB][:510]
    return torch.LongTensor([[0, *ids, 0]])


def export(model_onnx, path: str) -> None:
    import onnx
    input_ids = torch.randint(1, 100, (48,))
    input_ids = torch.LongTensor([[0, *input_ids.tolist(), 0]])
    style = torch.randn(1, 256)
    speed = torch.ones(1, dtype=torch.float32)
    torch.onnx.export(
        model_onnx, (input_ids, style, speed), path, export_params=True, verbose=False,
        input_names=["input_ids", "style", "speed"], output_names=["waveform", "duration"], opset_version=17,
        dynamic_axes={"input_ids": {0: "batch_size", 1: "input_ids_len"}, "style": {0: "batch_size"}, "speed": {0: "batch_size"}},
        do_constant_folding=True, dynamo=False,
    )
    onnx.checker.check_model(onnx.load(path))


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    if n < 1000:
        return 0.0
    a, b = a[:n] - a[:n].mean(), b[:n] - b[:n].mean()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def mel_correlation(a: np.ndarray, b: np.ndarray, sr: int = 24000) -> float:
    """Correlation of log-mel spectrograms (what the ear hears; phase-insensitive)."""
    import librosa
    n = min(len(a), len(b))
    ma = np.log(librosa.feature.melspectrogram(y=a[:n], sr=sr, n_fft=1024, hop_length=256, n_mels=80) + 1e-5).ravel()
    mb = np.log(librosa.feature.melspectrogram(y=b[:n], sr=sr, n_fft=1024, hop_length=256, n_mels=80) + 1e-5).ravel()
    ma, mb = ma - ma.mean(), mb - mb.mean()
    return float(np.dot(ma, mb) / (np.linalg.norm(ma) * np.linalg.norm(mb) + 1e-9))


def parity(onnx_path: str, model, pack, sentences: list[str], frontend, complex_model=None) -> dict:
    """Waveform correlation between onnxruntime and the exported PyTorch model (same real-valued
    iSTFT math; this is the gate), plus informational numbers against the default complex-STFT
    KModel path and a log-mel correlation."""
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    corrs, len_ratio, corr_complex, mel_corrs = [], [], [], []
    for s in sentences:
        ps = frontend(s, english_accent="pakistani")[:510]
        ids = _ids(ps)
        ref_s = pack[len(ps) - 1]
        with torch.no_grad():
            wav_pt = model(ps, ref_s, 1.0).numpy()
        out = sess.run(None, {"input_ids": ids.numpy(), "style": ref_s.numpy().astype(np.float32), "speed": np.array([1.0], dtype=np.float32)})
        wav_ox = out[0].reshape(-1).astype(np.float32)
        corrs.append(correlation(wav_pt, wav_ox))
        len_ratio.append(len(wav_ox) / max(len(wav_pt), 1))
        mel_corrs.append(mel_correlation(wav_pt, wav_ox))
        if complex_model is not None:
            with torch.no_grad():
                wav_c = complex_model(ps, ref_s, 1.0).numpy()
            corr_complex.append(correlation(wav_c, wav_ox))
    rep = {"min_corr": float(min(corrs)), "mean_corr": float(np.mean(corrs)), "mel_corr_min": float(min(mel_corrs)),
           "len_ratio_mean": float(np.mean(len_ratio)), "n": len(corrs)}
    if corr_complex:
        rep["corr_vs_complex_stft_path"] = {"min": float(min(corr_complex)), "mean": float(np.mean(corr_complex))}
    return rep


def fp16_only(a) -> None:
    """Convert the fp32 graph in <out> to fp16 and run the parity check; writes <out>/fp16_report.json."""
    import onnx
    from onnxconverter_common import float16  # type: ignore
    from kokoro import KModel
    from lughaat_tts.codeswitch import MixedFrontend
    fp32 = os.path.join(a.out, "lughaat-tts-82m.onnx")
    fp16 = os.path.join(a.out, "lughaat-tts-82m-fp16.onnx")
    rep: dict = {"pass": False}
    try:
        # explicit Cast / index-arithmetic / normalisation nodes stay fp32, otherwise the graph fails to load
        block =list(getattr(float16, "DEFAULT_OP_BLOCK_LIST", [])) + ["Cast", "Range", "CumSum", "Gather", "GatherElements",
                                                                         "ScatterND", "ScatterElements", "Where", "Round", "Clip",
                                                                         "LayerNormalization", "InstanceNormalization", "Softmax",
                                                                         "STFT", "DFT", "Loop", "If"]
        t0 = time.time()
        # shape inference is required so the converter can place casts around blocked ops correctly
        # (without it, LayerNorm/MatMul end up with mixed float/float16 inputs); it takes a few minutes
        m16 = float16.convert_float_to_float16(onnx.load(fp32), keep_io_types=True, op_block_list=sorted(set(block)), disable_shape_infer=False)
        onnx.save(m16, fp16)
        rep["convert_seconds"] = round(time.time() - t0, 1)
        model = KModel(repo_id="hexgrad/Kokoro-82M", config=a.config, model=a.model, disable_complex=True).eval()
        pack = torch.load(a.voice, weights_only=True).float()
        sentences = (APPENDIX_A + APPENDIX_A2)[: a.fp16_sentences]
        t0 = time.time()
        rep.update(parity(fp16, model, pack, sentences, MixedFrontend()))
        rep["parity_seconds"] = round(time.time() - t0, 1)
        rep["pass"] = rep["min_corr"] >= a.threshold
    except Exception as e:
        rep["error"] = repr(e)[:400]
    with open(os.path.join(a.out, "fp16_report.json"), "w") as f:
        json.dump(rep, f, indent=2)
    print("fp16:", rep)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--voice", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, default=0.99)
    ap.add_argument("--fp16-sentences", type=int, default=6, help="sentences for the (slow, CPU-emulated) fp16 parity check")
    ap.add_argument("--fp16-timeout-min", type=float, default=25.0, help="give up on fp16 after this long (fp32 is always shipped)")
    ap.add_argument("--fp16-only", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.fp16_only:
        return fp16_only(a)
    os.makedirs(a.out, exist_ok=True)
    from kokoro import KModel
    from kokoro.model import KModelForONNX
    from lughaat_tts.codeswitch import MixedFrontend
    fe = MixedFrontend()
    model = KModel(repo_id="hexgrad/Kokoro-82M", config=a.config, model=a.model, disable_complex=True).eval()
    ref_model = KModel(repo_id="hexgrad/Kokoro-82M", config=a.config, model=a.model).eval()
    pack = torch.load(a.voice, weights_only=True).float()
    sentences = (APPENDIX_A + APPENDIX_A2)[:20]
    fp32 = os.path.join(a.out, "lughaat-tts-82m.onnx")
    export(KModelForONNX(model), fp32)
    # tracing perturbs the exported module's eager behaviour (iSTFT buffers); compare against a FRESH model
    model = KModel(repo_id="hexgrad/Kokoro-82M", config=a.config, model=a.model, disable_complex=True).eval()
    rep = {"fp32": parity(fp32, model, pack, sentences, fe, complex_model=ref_model), "fp32_mb": round(os.path.getsize(fp32) / 1e6, 1)}
    rep["fp32"]["pass"] = rep["fp32"]["min_corr"] >= a.threshold
    print("fp32:", rep["fp32"])
    fp16 = os.path.join(a.out, "lughaat-tts-82m-fp16.onnx")
    # fp16: conversion + CPU-emulated parity can take very long -> isolated subprocess with a timeout
    fp16_json = os.path.join(a.out, "fp16_report.json")
    cmd = [sys.executable, os.path.abspath(__file__), "--fp16-only", "--config", a.config, "--model", a.model, "--voice", a.voice,
           "--out", a.out, "--threshold", str(a.threshold), "--fp16-sentences", str(a.fp16_sentences)]
    try:
        subprocess.run(cmd, timeout=a.fp16_timeout_min * 60, check=False)
        rep["fp16"] = json.load(open(fp16_json)) if os.path.exists(fp16_json) else {"error": "fp16 subprocess produced no report", "pass": False}
    except subprocess.TimeoutExpired:
        rep["fp16"] = {"error": f"fp16 conversion/parity exceeded {a.fp16_timeout_min} min; skipped", "pass": False}
    if not rep["fp16"].get("pass") and os.path.exists(fp16):
        os.remove(fp16)
        print("fp16 not shipped:", rep["fp16"].get("error", rep["fp16"]))
    else:
        rep["fp16_mb"] = round(os.path.getsize(fp16) / 1e6, 1) if os.path.exists(fp16) else None
        print("fp16:", rep["fp16"])
    with open(os.path.join(a.out, "onnx_report.json"), "w") as f:
        json.dump(rep, f, indent=2)
    if not rep["fp32"]["pass"]:
        raise SystemExit(f"ONNX fp32 parity failed: {rep['fp32']}")


if __name__ == "__main__":
    main()

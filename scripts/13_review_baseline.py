#!/usr/bin/env python3
"""Audit all saved baseline WAVs and optionally regenerate changed pronunciations.

Candidates use the same base Kokoro voices. This measures signal integrity and
frontend changes, not perceptual pronunciation accuracy. Originals remain intact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def signal_checks(path: Path) -> dict:
    import numpy as np
    import soundfile as sf
    from eval.metrics import clipping_ratio, max_silence_seconds

    y, sr = sf.read(path, dtype="float32")
    issues = []
    if y.ndim != 1:
        issues.append("not_mono")
        y = y.mean(axis=1)
    finite = bool(np.isfinite(y).all())
    if not finite:
        issues.append("nonfinite_audio")
    if not len(y):
        issues.append("empty_audio")
    if sr != 24000:
        issues.append("unexpected_sample_rate")
    clip = clipping_ratio(y) if finite else None
    silence = max_silence_seconds(y, sr) if finite else None
    peak = float(np.max(np.abs(y))) if len(y) and finite else None
    if peak is not None and peak < 1e-4:
        issues.append("silent_audio")
    if clip is not None and clip >= 0.001:
        issues.append("clipping")
    if silence is not None and silence > 1.5:
        issues.append("long_internal_silence")
    return {"sample_rate": sr, "seconds": len(y) / sr, "peak": peak,
            "clipping_ratio": clip, "max_internal_silence_s": silence, "issues": issues}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", type=Path, default=ROOT / "baseline_samples")
    ap.add_argument("--out", type=Path, default=ROOT / "pronunciation_review")
    ap.add_argument("--text-edits", type=Path, help="JSON mapping exact original text to explicitly corrected text")
    ap.add_argument("--regenerate-changed", action="store_true")
    ap.add_argument("--offline", action="store_true")
    a = ap.parse_args()
    baseline, out = a.baseline.resolve(), a.out.resolve()
    if out == baseline or baseline in out.parents:
        ap.error("--out must be outside the original baseline directory")
    if a.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
    import torch
    import soundfile as sf
    from lughaat_tts.codeswitch import MixedFrontend
    from lughaat_tts.phoneset import assert_in_vocab
    from lughaat_tts.version import FRONTEND_VERSION
    from eval.listening_test import write_listening_test
    from misaki import en

    torch.set_num_threads(4)
    meta = json.loads((baseline / "baseline_metrics.json").read_text(encoding="utf-8"))
    edits = json.loads(a.text_edits.read_text(encoding="utf-8")) if a.text_edits else {}
    if not isinstance(edits, dict) or any(not isinstance(k, str) or not isinstance(v, str) or not v.strip() for k, v in edits.items()):
        ap.error("--text-edits must contain nonempty text-to-text strings")
    unused = set(edits) - {r["text"] for r in meta["rows"]}
    if unused:
        ap.error(f"text edits do not match baseline sentences: {sorted(unused)}")
    fe, en_g2p = MixedFrontend(), en.G2P(trf=False, british=False)
    out.mkdir(parents=True, exist_ok=True)
    rows, listening, cache = [], [], {}
    model, packs = None, {}
    for i, original in enumerate(meta["rows"]):
        filename = PureWindowsPath(original["wav"]).name
        path = baseline / filename
        text = original["text"]
        spoken = edits.get(text, text)
        key = (original["set"], spoken)
        if key not in cache:
            cache[key] = en_g2p(spoken)[0] if original["set"] == "A3" else fe(spoken, english_accent="pakistani")
        ps = assert_in_vocab(cache[key])
        if not ps.strip() or len(ps) > 510:
            raise ValueError(f"Invalid phoneme length for {filename}: {len(ps)}")
        r = {"file": filename, "text": text, "spoken_text": spoken, "voice": original["voice"],
             "set": original["set"], "old_phonemes": original["phonemes"], "current_phonemes": ps,
             "phonemes_changed": ps != original["phonemes"], "explicit_text_edit": spoken != text,
             "original_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
             "original_signal": signal_checks(path)}
        listening.append({**original, "wav": str(path), "accent": "original baseline"})
        if a.regenerate_changed and r["phonemes_changed"]:
            from huggingface_hub import hf_hub_download
            if model is None:
                from kokoro import KModel
                config = hf_hub_download("hexgrad/Kokoro-82M", "config.json", local_files_only=a.offline)
                weights = hf_hub_download("hexgrad/Kokoro-82M", "kokoro-v1_0.pth", local_files_only=a.offline)
                model = KModel(repo_id="hexgrad/Kokoro-82M", config=config, model=weights).eval()
            voice = original["voice"]
            if voice not in packs:
                vp = hf_hub_download("hexgrad/Kokoro-82M", f"voices/{voice}.pt", local_files_only=a.offline)
                packs[voice] = torch.load(vp, map_location="cpu", weights_only=True)
            torch.manual_seed(0)
            with torch.inference_mode():
                audio = model(ps, packs[voice][len(ps) - 1], 1.0).cpu().numpy()
            candidate = out / "candidates" / filename
            candidate.parent.mkdir(parents=True, exist_ok=True)
            sf.write(candidate, audio, 24000)
            r["candidate"] = candidate.relative_to(out).as_posix()
            r["candidate_signal"] = signal_checks(candidate)
            listening.append({**original, "wav": str(candidate), "accent": "corrected phonemes; needs listening"})
        rows.append(r)
        if (i + 1) % 20 == 0:
            print(f"Audited {i+1}/{len(meta['rows'])} files", flush=True)
    listed = {r["file"] for r in rows}
    # Also inspect WAVs missing from the manifest, so "all" includes extra files.
    extras = [{"file": p.name, "signal": signal_checks(p)}
              for p in sorted(baseline.glob("*.wav")) if p.name not in listed]
    summary = {
        "frontend_version": FRONTEND_VERSION,
        "manifest_files": len(rows), "extra_files": len(extras),
        "total_audio_seconds": sum(r["original_signal"]["seconds"] for r in rows) + sum(r["signal"]["seconds"] for r in extras),
        "original_files_with_signal_issues": sum(bool(r["original_signal"]["issues"]) for r in rows) + sum(bool(r["signal"]["issues"]) for r in extras),
        "files_with_changed_phonemes": sum(r["phonemes_changed"] for r in rows),
        "generated_candidates": sum("candidate" in r for r in rows),
        "candidates_with_signal_issues": sum(bool(r.get("candidate_signal", {}).get("issues")) for r in rows),
        "asr_evaluated": False, "human_listening_validated": False,
        "quality_note": "Signal checks and phoneme changes do not establish spoken pronunciation accuracy. Candidates still use base Hindi voices.",
    }
    (out / "audit.json").write_text(json.dumps({"summary": summary, "rows": rows, "extras": extras}, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    page = write_listening_test(str(out), listening, title="Urdu baseline pronunciation review")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Open: {page}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Final verification in a brand-new venv (plan 11.2.5): pip install from the HF repo, synthesise
Appendix A / A2 / A3 on CPU, recompute CER / PCER / WER and compare with the Phase-7 numbers (±0.01).

    HF_TOKEN=... python scripts/12_verify_install.py --model-repo USER/lughaat-tts-82m --out verify_out \
        [--reference export/v1.0/eval/metrics.json] [--asr-model openai/whisper-large-v3-turbo]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SYNTH = r'''
import json, os, sys, time
import soundfile as sf
from lughaat_tts import UrduPipeline
sys.path.insert(0, %(root)r)
from eval.testsets import APPENDIX_A, APPENDIX_A2, APPENDIX_A3
out = %(out)r
tts = UrduPipeline(voice="uf_rasa", device="cpu", repo_id=%(repo)r)
rows = []
t0 = time.time(); dur = 0.0
for name, sents, acc in (("A", APPENDIX_A, "pakistani"), ("A2", APPENDIX_A2, "pakistani"), ("A3", APPENDIX_A3, "native")):
    for i, s in enumerate(sents):
        a = tts(s, english_accent=acc)
        dur += len(a) / 24000
        p = os.path.join(out, f"{name}_{i+1:02d}.wav")
        sf.write(p, a, 24000)
        rows.append({"set": name, "text": s, "wav": p})
json.dump({"rows": rows, "rtf_cpu": (time.time() - t0) / dur}, open(os.path.join(out, "synth.json"), "w", encoding="utf-8"), ensure_ascii=False)
print("synthesised", len(rows), "RTF", (time.time() - t0) / dur)
'''


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-repo", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reference", default=None)
    ap.add_argument("--asr-model", default=os.environ.get("ASR_EVAL_MODEL", "openai/whisper-large-v3"))
    ap.add_argument("--keep-venv", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    venv = tempfile.mkdtemp(prefix="lt_verify_venv_")
    py = os.path.join(venv, "Scripts" if os.name == "nt" else "bin", "python" + (".exe" if os.name == "nt" else ""))
    token = os.environ.get("HF_TOKEN", "")
    url = f"git+https://{('user:' + token + '@') if token else ''}huggingface.co/{a.model_repo}"
    subprocess.run([sys.executable, "-m", "venv", venv], check=True)
    subprocess.run([py, "-m", "pip", "install", "-q", "--upgrade", "pip"], check=True)
    r = subprocess.run([py, "-m", "pip", "install", "-q", url, "soundfile"], capture_output=True, text=True)
    report = {"model_repo": a.model_repo, "pip_install_ok": r.returncode == 0, "pip_tail": (r.stdout + r.stderr)[-1500:]}
    if r.returncode != 0:
        json.dump(report, open(os.path.join(a.out, "verify.json"), "w"), indent=2)
        raise SystemExit("pip install from the HF repo failed:\n" + report["pip_tail"])
    subprocess.run([py, "-m", "spacy", "download", "en_core_web_sm", "-q"], check=False)
    env = {**os.environ, "HF_TOKEN": token, "PYTHONUTF8": "1"}
    r = subprocess.run([py, "-c", SYNTH % {"root": ROOT, "out": a.out, "repo": a.model_repo}], env=env, capture_output=True, text=True)
    report["synth_ok"] = r.returncode == 0
    report["synth_tail"] = (r.stdout + r.stderr)[-1500:]
    if r.returncode != 0:
        json.dump(report, open(os.path.join(a.out, "verify.json"), "w"), indent=2)
        raise SystemExit("synthesis in the fresh venv failed:\n" + report["synth_tail"])
    synth = json.load(open(os.path.join(a.out, "synth.json"), encoding="utf-8"))
    report["rtf_cpu"] = synth["rtf_cpu"]
    # score with the project's own evaluation code (this interpreter)
    sys.path.insert(0, ROOT)
    from eval.asr import WhisperASR, cer, wer
    from eval.pcer import pcer_detail
    import numpy as np
    asr = WhisperASR(a.asr_model)
    cers, pcers, wers = [], [], []
    for row in synth["rows"]:
        lang = "en" if row["set"] == "A3" else "ur"
        hyp = asr.transcribe_one(row["wav"], language=lang)
        if row["set"] == "A":
            cers.append(cer(row["text"], hyp))
        elif row["set"] == "A2":
            pcers.append(pcer_detail(row["text"], hyp).pcer)
        else:
            wers.append(wer(row["text"], hyp, lang="en"))
    report.update({"appendixA_cer": float(np.nanmean(cers)), "mixed_pcer_pakistani": float(np.nanmean(pcers)), "english_wer_native": float(np.nanmean(wers))})
    if a.reference and os.path.exists(a.reference):
        ref = json.load(open(a.reference))
        diffs = {k: abs(report[k] - ref.get(k, float("nan"))) for k in ("appendixA_cer", "mixed_pcer_pakistani", "english_wer_native") if ref.get(k) is not None}
        report["diff_vs_phase7"] = diffs
        report["within_0_01"] = all(d <= 0.01 + 1e-9 for d in diffs.values()) if diffs else None
    json.dump(report, open(os.path.join(a.out, "verify.json"), "w"), indent=2)
    print(json.dumps(report, indent=2))
    if not a.keep_venv:
        import shutil
        shutil.rmtree(venv, ignore_errors=True)


if __name__ == "__main__":
    main()

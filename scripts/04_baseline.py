#!/usr/bin/env python3
"""Phase 3: zero-training baseline with base Kokoro-82M and the Hindi voices (plan 5).

    python scripts/04_baseline.py --out /kaggle/tmp/baseline [--asr-model openai/whisper-large-v3]

* Appendix A (Urdu) and A2 (mixed) through OUR frontend -> base KModel with hf_alpha,
  hf_beta, hm_omega, hm_psi.
* Appendix A3 (English) with af_heart: its English WER is the "no-forgetting" reference used
  by the Phase-7 gate.
* Writes baseline_samples/*.wav and baseline_metrics.json (CER / PCER / WER + ASR floor note).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from eval.testsets import APPENDIX_A, APPENDIX_A2, APPENDIX_A3  # noqa: E402

HINDI_VOICES = ["hm_psi", "hf_alpha", "hf_beta", "hm_omega"]


def log(m: str) -> None:
    print(f"[baseline {time.strftime('%H:%M:%S')}] {m}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--asr-model", default=os.environ.get("ASR_EVAL_MODEL", "openai/whisper-large-v3"))
    ap.add_argument("--skip-asr", action="store_true")
    ap.add_argument("--voices", default=",".join(HINDI_VOICES))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    import soundfile as sf
    import torch
    from kokoro import KModel
    from lughaat_tts.codeswitch import MixedFrontend
    from huggingface_hub import hf_hub_download

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = KModel(repo_id="hexgrad/Kokoro-82M").to(device).eval()
    fe = MixedFrontend()

    def voice(name: str):
        return torch.load(hf_hub_download("hexgrad/Kokoro-82M", f"voices/{name}.pt"), weights_only=True)

    def synth(ps: str, pack) -> np.ndarray:
        ps = ps[:510]
        with torch.no_grad():
            return model(ps, pack[len(ps) - 1].to(device), 1.0).cpu().numpy()

    rows = []
    try:
        from misaki import en  # type: ignore
        en_g2p = en.G2P(trf=False, british=False)
    except Exception:
        en_g2p = None
    for vname in a.voices.split(","):
        pack = voice(vname)
        for i, s in enumerate(APPENDIX_A):
            ps = fe(s, english_accent="pakistani")
            wav = synth(ps, pack)
            p = os.path.join(a.out, f"A_{i+1:02d}_{vname}.wav")
            sf.write(p, wav, 24000)
            rows.append({"set": "A", "voice": vname, "text": s, "phonemes": ps, "wav": p})
        for i, s in enumerate(APPENDIX_A2):
            ps = fe(s, english_accent="pakistani")
            wav = synth(ps, pack)
            p = os.path.join(a.out, f"A2_{i+1:02d}_{vname}.wav")
            sf.write(p, wav, 24000)
            rows.append({"set": "A2", "voice": vname, "text": s, "phonemes": ps, "wav": p})
    pack = voice("af_heart")
    for i, s in enumerate(APPENDIX_A3):
        ps = en_g2p(s)[0] if en_g2p else fe(s, english_accent="native")
        wav = synth(ps, pack)
        p = os.path.join(a.out, f"A3_{i+1:02d}_af_heart.wav")
        sf.write(p, wav, 24000)
        rows.append({"set": "A3", "voice": "af_heart", "text": s, "phonemes": ps, "wav": p})
    log(f"synthesised {len(rows)} baseline samples")

    metrics = {"asr_model": a.asr_model, "voices": a.voices.split(",")}
    if not a.skip_asr:
        from eval.asr import WhisperASR, cer, wer
        from eval.pcer import pcer_detail
        asr = WhisperASR(a.asr_model)
        for r in rows:
            lang = "en" if r["set"] == "A3" else "ur"
            r["hyp"] = asr.transcribe_one(r["wav"], language=lang)
            if r["set"] == "A":
                r["cer"] = cer(r["text"], r["hyp"])
            elif r["set"] == "A2":
                d = pcer_detail(r["text"], r["hyp"], fe)
                r["pcer"], r["pcer_en"], r["lost"] = d.pcer, d.pcer_en, d.english_words_lost
            else:
                r["wer"] = wer(r["text"], r["hyp"], lang="en")
        for vname in a.voices.split(","):
            cs = [r["cer"] for r in rows if r["set"] == "A" and r["voice"] == vname]
            ps = [r["pcer"] for r in rows if r["set"] == "A2" and r["voice"] == vname]
            metrics[f"A_cer_{vname}"] = float(np.nanmean(cs))
            metrics[f"A2_pcer_{vname}"] = float(np.nanmean(ps))
        ws = [r["wer"] for r in rows if r["set"] == "A3"]
        metrics["A3_wer_af_heart"] = float(np.nanmean(ws))
        metrics["best_hindi_voice_A_cer"] = min(metrics[f"A_cer_{v}"] for v in a.voices.split(","))
    with open(os.path.join(a.out, "baseline_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "rows": rows}, f, ensure_ascii=False, indent=2)
    log(f"baseline metrics: {metrics}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Phase 7 evaluation and the quality gates (plan 9).

    # light (per-epoch, ~10 Urdu / 5 mixed / 5 English sentences)
    python eval/evaluate.py --config export/config.json --model export/lughaat-tts-82m.pth \
        --voices export/voices --out eval_out --light
    # full
    python eval/evaluate.py ... --data /kaggle/tmp/data --baseline baseline/baseline_metrics.json

Test sets: val texts (in-domain), 300 FLEURS ur_pk test sentences (OOD), Appendix A (hard Urdu),
Appendix A2 + 200 synthetic code-switched, Appendix A3 + 200 LibriTTS-R test sentences.
Metrics: Urdu CER/WER (whisper-large-v3, ur), English WER (en), PCER for mixed text
(pakistani / native / auto), DNSMOS (+UTMOS when available), ECAPA similarity, spectral
rolloff, clipping / long-silence / comb checks, RTF. Writes metrics.json, per_sentence.jsonl,
samples/*.wav and (full mode) listening_test.html.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from eval.testsets import APPENDIX_A, APPENDIX_A2, APPENDIX_A3  # noqa: E402

GATES_DOC = """Quality gates (plan 9):
  ood_cer <= asr_floor + 0.03 and clearly better than the Hindi-voice baseline
  appendixA_cer <= asr_floor + 0.06
  mixed pcer (pakistani) <= 0.10 overall; english-part pcer <= 0.15; no English word lost
  english wer (native) <= base af_heart wer + 0.05
  utmos/dnsmos >= 3.5 or within 0.3 of real Rasa recordings
  no clipping, no silence > 1.5 s, no comb/echo peak"""


def log(m: str) -> None:
    print(f"[eval {time.strftime('%H:%M:%S')}] {m}", flush=True)


def read_lines(p: str, n: int | None = None) -> list[str]:
    if not p or not os.path.exists(p):
        return []
    with open(p, encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    return lines[:n] if n else lines


def nanmean(xs) -> float:
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(np.mean(xs)) if xs else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--voices", required=True, help="dir with uf_rasa.pt / um_rasa.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=None, help="data dir (eval/*.txt, audio/, manifest.jsonl) for the full test sets")
    ap.add_argument("--baseline", default=None, help="baseline_metrics.json from scripts/04_baseline.py")
    ap.add_argument("--asr-model", default=os.environ.get("ASR_EVAL_MODEL", "openai/whisper-large-v3"))
    ap.add_argument("--asr-floor", type=float, default=None, help="override the FLEURS ASR floor")
    ap.add_argument("--light", action="store_true")
    ap.add_argument("--no-mos", action="store_true")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "samples"), exist_ok=True)

    import soundfile as sf
    import torch
    from lughaat_tts.pipeline import UrduPipeline
    from eval.asr import WhisperASR, cer, wer
    from eval.pcer import pcer_detail
    from eval import metrics as M

    voices = [v for v in ("uf_rasa", "um_rasa") if os.path.exists(os.path.join(a.voices, f"{v}.pt"))]
    assert voices, "no voicepacks found"
    tts = UrduPipeline(voice=voices[0], config_path=a.config, model_path=a.model, voices_dir=a.voices, repo_id="local")
    fe = tts.frontend

    # ---- test sets ------------------------------------------------------------
    rng = random.Random(0)
    sets: dict[str, list[tuple[str, str]]] = {}   # name -> [(text, kind)] kind in ur|mixed|en
    if a.light:
        sets["A"] = [(s, "ur") for s in APPENDIX_A[:10]]
        sets["A2"] = [(s, "mixed") for s in APPENDIX_A2[:5]]
        sets["A3"] = [(s, "en") for s in APPENDIX_A3[:5]]
    else:
        sets["A"] = [(s, "ur") for s in APPENDIX_A]
        sets["A2"] = [(s, "mixed") for s in APPENDIX_A2]
        sets["A3"] = [(s, "en") for s in APPENDIX_A3]
        if a.data:
            ev = os.path.join(a.data, "eval")
            sets["val_ur"] = [(s, "ur") for s in read_lines(os.path.join(ev, "val_texts_ur.txt"), 100)]
            sets["fleurs"] = [(s, "ur") for s in read_lines(os.path.join(ev, "fleurs_test.txt"), 300)]
            sets["mixed_syn"] = [(s, "mixed") for s in read_lines(os.path.join(ev, "synthetic_mixed_test.txt"), 200)]
            sets["en_libritts"] = [(s, "en") for s in read_lines(os.path.join(ev, "english_test.txt"), 200)]
    log("test sets: " + ", ".join(f"{k}={len(v)}" for k, v in sets.items()))

    asr = WhisperASR(a.asr_model, batch_size=8)
    floor = a.asr_floor
    if floor is None and a.data and os.path.exists(os.path.join(a.data, "data_report.json")):
        floor = json.load(open(os.path.join(a.data, "data_report.json"))).get("asr", {}).get("fleurs_floor_cer")
    if floor is None:
        floor = 0.10
    base = json.load(open(a.baseline)).get("metrics", {}) if a.baseline and os.path.exists(a.baseline) else {}

    rows = []
    per_sentence = open(os.path.join(a.out, "per_sentence.jsonl"), "w", encoding="utf-8")
    t_synth, dur_synth = 0.0, 0.0
    accents = ["pakistani"] if a.light else ["pakistani", "native", "auto"]
    for voice in voices:
        for set_name, items in sets.items():
            for i, (text, kind) in enumerate(items):
                acc_list = accents if kind == "mixed" else (["native"] if kind == "en" else ["pakistani"])
                for accent in acc_list:
                    t0 = time.perf_counter()
                    audio = tts(text, voice=voice, english_accent=accent)
                    t_synth += time.perf_counter() - t0
                    dur_synth += len(audio) / 24000
                    wav = os.path.join(a.out, "samples", f"{set_name}_{i+1:03d}_{voice}_{accent}.wav")
                    if set_name in ("A", "A2", "A3") or i < 5:
                        sf.write(wav, audio, 24000)
                    else:
                        wav = None
                    lang = "en" if kind == "en" else "ur"
                    hyp = asr.transcribe_one(audio, language=lang, sr=24000)
                    r = {"set": set_name, "kind": kind, "voice": voice, "accent": accent, "text": text, "hyp": hyp, "wav": wav,
                         "dur": round(len(audio) / 24000, 2)}
                    if kind == "ur":
                        r["cer"] = cer(text, hyp)
                        r["wer"] = wer(text, hyp, lang="ur")
                    elif kind == "en":
                        r["wer"] = wer(text, hyp, lang="en")
                        r["cer"] = cer(text, hyp, lang="en")
                    else:
                        d = pcer_detail(text, hyp, fe)
                        r.update({"pcer": d.pcer, "pcer_ur": d.pcer_ur, "pcer_en": d.pcer_en, "lost": d.english_words_lost})
                    if not a.no_mos:
                        r["dnsmos"] = M.dnsmos(audio)
                        if not a.light:
                            r["utmos"] = M.utmos(audio)
                    r["rolloff99"] = M.spectral_rolloff(audio, percent=0.99)
                    r["clipping"] = M.clipping_ratio(audio)
                    r["max_silence"] = M.max_silence_seconds(audio)
                    r["comb"] = M.comb_cepstral_peak(audio)
                    rows.append(r)
                    per_sentence.write(json.dumps(r, ensure_ascii=False) + "\n")
            log(f"  {voice} {set_name}: done")
    per_sentence.close()

    # ---- aggregate -------------------------------------------------------------
    def agg(pred, key):
        return nanmean([r.get(key) for r in rows if pred(r)])

    m = {"tag": a.tag, "asr_model": a.asr_model, "asr_floor_cer": floor, "voices": voices, "light": a.light,
         "n_sentences": len(rows), "rtf_gpu_or_cpu": round(t_synth / max(dur_synth, 1e-6), 4), "device": tts.device}
    m["appendixA_cer"] = agg(lambda r: r["set"] == "A", "cer")
    m["ood_cer"] = agg(lambda r: r["set"] == "fleurs", "cer") if "fleurs" in sets else float("nan")
    m["val_cer"] = agg(lambda r: r["set"] == "val_ur", "cer") if "val_ur" in sets else float("nan")
    for acc in accents:
        m[f"mixed_pcer_{acc}"] = agg(lambda r, acc=acc: r["kind"] == "mixed" and r["accent"] == acc, "pcer")
        m[f"mixed_pcer_en_{acc}"] = agg(lambda r, acc=acc: r["kind"] == "mixed" and r["accent"] == acc, "pcer_en")
        m[f"mixed_pcer_ur_{acc}"] = agg(lambda r, acc=acc: r["kind"] == "mixed" and r["accent"] == acc, "pcer_ur")
        m[f"english_words_lost_{acc}"] = int(sum(len(r.get("lost", [])) for r in rows if r["kind"] == "mixed" and r["accent"] == acc))
    m["english_wer_native"] = agg(lambda r: r["kind"] == "en", "wer")
    m["dnsmos"] = agg(lambda r: True, "dnsmos")
    m["utmos"] = agg(lambda r: True, "utmos")
    m["rolloff99_hz"] = agg(lambda r: True, "rolloff99")
    m["clipping_max"] = float(max([r["clipping"] for r in rows] or [0]))
    m["max_silence_s"] = float(max([r["max_silence"] for r in rows] or [0]))
    m["comb_peak_median"] = agg(lambda r: True, "comb")
    m["baseline"] = base

    # speaker similarity against real reference clips
    if a.data and os.path.exists(os.path.join(a.data, "voicepack_refs.json")):
        refs = json.load(open(os.path.join(a.data, "voicepack_refs.json")))
        sims = {}
        for spk, voice in (("0", "uf_rasa"), ("1", "um_rasa")):
            if voice not in voices or not refs.get(spk):
                continue
            real = [os.path.join(a.data, "audio", p) for p in refs[spk][:5] if os.path.exists(os.path.join(a.data, "audio", p))]
            synth = [r["wav"] for r in rows if r["voice"] == voice and r["wav"] and r["set"] == "A"][:5]
            vals = []
            for rp in real:
                ry, rs = sf.read(rp, dtype="float32")
                for sp in synth:
                    sy, ss = sf.read(sp, dtype="float32")
                    vals.append(M.ecapa_similarity(ry, sy))
            sims[voice] = nanmean(vals)
            if real and not a.no_mos:
                m[f"real_dnsmos_{voice}"] = nanmean([M.dnsmos(sf.read(p, dtype="float32")[0]) for p in real])
            if real:
                m[f"real_comb_{voice}"] = nanmean([M.comb_cepstral_peak(sf.read(p, dtype="float32")[0]) for p in real])
                m[f"real_rolloff99_{voice}"] = nanmean([M.spectral_rolloff(sf.read(p, dtype="float32")[0], percent=0.99) for p in real])
        m["ecapa_similarity"] = sims

    # ---- gates -------------------------------------------------------------------
    g = {}
    g["ood_cer"] = (m["ood_cer"] <= floor + 0.03) if not math.isnan(m["ood_cer"]) else None
    if base.get("best_hindi_voice_A_cer") is not None:
        g["beats_baseline"] = m["appendixA_cer"] < base["best_hindi_voice_A_cer"] - 0.05
    g["appendixA_cer"] = m["appendixA_cer"] <= floor + 0.06
    g["mixed_pcer"] = m["mixed_pcer_pakistani"] <= 0.10
    g["mixed_pcer_en"] = (m["mixed_pcer_en_pakistani"] <= 0.15) if not math.isnan(m["mixed_pcer_en_pakistani"]) else None
    g["no_english_word_lost"] = m["english_words_lost_pakistani"] == 0
    if base.get("A3_wer_af_heart") is not None:
        g["english_wer"] = m["english_wer_native"] <= base["A3_wer_af_heart"] + 0.05
    mos = m["utmos"] if not math.isnan(m["utmos"]) else m["dnsmos"]
    real_mos = nanmean([v for k, v in m.items() if k.startswith("real_dnsmos_")])
    g["naturalness"] = (mos >= 3.5 or (not math.isnan(real_mos) and mos >= real_mos - 0.3)) if not math.isnan(mos) else None
    g["no_clipping"] = m["clipping_max"] < 0.001
    g["no_long_silence"] = m["max_silence_s"] <= 1.5
    real_comb = nanmean([v for k, v in m.items() if k.startswith("real_comb_")])
    # the comb/echo gate is relative to real recordings of the same voices (absolute values track pitch and voicing)
    g["no_comb"] = (m["comb_peak_median"] <= real_comb + 3.0) if not (math.isnan(real_comb) or math.isnan(m["comb_peak_median"])) else None
    m["gates"] = g
    m["gates_passed"] = all(v for v in g.values() if v is not None)
    m["gates_doc"] = GATES_DOC
    with open(os.path.join(a.out, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=2)
    log("metrics: " + json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items() if k not in ("baseline", "gates_doc")}, ensure_ascii=False))

    if not a.light:
        from eval.listening_test import write_listening_test
        write_listening_test(a.out, rows, title=f"Lughaat-TTS listening test {a.tag}")


if __name__ == "__main__":
    main()

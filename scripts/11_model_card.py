#!/usr/bin/env python3
"""Generate the Hugging Face model card README.md (plan 11.2.3)."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lughaat_tts.version import FRONTEND_VERSION, __version__  # noqa: E402


def fmt(x) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return "n/a" if math.isnan(x) else f"{x:.3f}"
    return str(x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--state", required=True)
    ap.add_argument("--model-repo", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    st = json.load(open(a.state, encoding="utf-8"))
    m = st.get("final_metrics", {}) or {}
    base = (m.get("baseline") or st.get("baseline") or {})
    data = st.get("data_report", {}) or {}
    hours = data.get("hours_final", {})
    ur_h = sum(v for k, v in hours.items() if k.endswith("_ur"))
    en_h = sum(v for k, v in hours.items() if k.endswith("_en"))
    onnx_rep = {}
    p = os.path.join(a.export, "onnx", "onnx_report.json")
    if os.path.exists(p):
        onnx_rep = json.load(open(p))
    samples = []
    si = os.path.join(a.export, "samples", "index.json")
    if os.path.exists(si):
        samples = json.load(open(si, encoding="utf-8"))
    user = a.model_repo.split("/")[0]
    gates = m.get("gates", {})
    card = f"""---
license: apache-2.0
language:
- ur
- en
base_model: hexgrad/Kokoro-82M
pipeline_tag: text-to-speech
tags:
- kokoro
- styletts2
- urdu
- tts
- code-switching
library_name: kokoro
---

# Lughaat-TTS-82M (`{a.model_repo}`) — Urdu + English Kokoro-82M fine-tune

A fine-tune of [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) (82 M parameters, StyleTTS2 + ISTFTNet)
that speaks **Urdu**, **English** and **Urdu–English code-switched** text in two Urdu voices (`uf_rasa` female, `um_rasa` male).
It loads with the standard `kokoro.KModel`; the Urdu text frontend (normaliser, lexicon, neural G2P, English accent mapping) ships
in the `lughaat_tts` package in this repo. Frontend version: `{FRONTEND_VERSION}`; package version `{__version__}`.

## Usage

```bash
pip install git+https://huggingface.co/{a.model_repo}
```

```python
from lughaat_tts import UrduPipeline
tts = UrduPipeline(voice="uf_rasa", device="cpu")            # weights are downloaded from this repo
audio = tts("پاکستان ایک خوبصورت ملک ہے۔")                     # np.float32 @ 24 kHz
tts.save("آج کی meeting کینسل ہو گئی ہے، please email check کریں۔", "mixed.wav")
tts.save("Good morning, this is an English sentence.", "en.wav", english_accent="native")
print(tts.phonemize("آج کی meeting کینسل ہو گئی ہے۔"))      # phonemes + per-token language tags
```

`english_accent` is `"auto"` (default: Pakistani-English phones inside Urdu sentences, native American phones for English-only
sentences), `"pakistani"` or `"native"`. `retroflex_td=True` makes English t/d retroflex, as Urdu speakers do.

ONNX (`onnxruntime` + `numpy` only):

```bash
python -m lughaat_tts.onnx_infer --onnx onnx/lughaat-tts-82m.onnx --voice voices/uf_rasa.pt --text "پاکستان ایک خوبصورت ملک ہے۔" --out out.wav
```

Highest-naturalness path with the trained diffusion prosody sampler: see `diffusion/synthesize_diffusion.py`.

## Samples

{chr(10).join(f'- `{s["file"]}` ({s["voice"]}, {s["set"]}): {s["text"]}' for s in samples[:12])}

All Appendix sentences for both voices are in `samples/`.

## Metrics

ASR: `{m.get("asr_model", "openai/whisper-large-v3")}`. The ASR floor (its own CER on real FLEURS ur_pk speech) is reported next to every number.

| metric | value | gate |
|---|---|---|
| ASR floor CER (real FLEURS ur_pk audio) | {fmt(m.get("asr_floor_cer"))} | – |
| Out-of-domain Urdu CER (FLEURS test sentences) | {fmt(m.get("ood_cer"))} | ≤ floor + 0.03: {gates.get("ood_cer")} |
| Appendix A (hard Urdu) CER | {fmt(m.get("appendixA_cer"))} | ≤ floor + 0.06: {gates.get("appendixA_cer")} |
| Hindi-voice baseline Appendix A CER (base Kokoro) | {fmt(base.get("best_hindi_voice_A_cer"))} | beaten: {gates.get("beats_baseline")} |
| Code-switched PCER, pakistani accent (overall / English words) | {fmt(m.get("mixed_pcer_pakistani"))} / {fmt(m.get("mixed_pcer_en_pakistani"))} | ≤ 0.10 / ≤ 0.15: {gates.get("mixed_pcer")} / {gates.get("mixed_pcer_en")} |
| Code-switched PCER, native / auto | {fmt(m.get("mixed_pcer_native"))} / {fmt(m.get("mixed_pcer_auto"))} | – |
| English words lost (pakistani) | {m.get("english_words_lost_pakistani", "n/a")} | 0: {gates.get("no_english_word_lost")} |
| English-only WER, Urdu voices, native accent | {fmt(m.get("english_wer_native"))} | ≤ base af_heart WER {fmt(base.get("A3_wer_af_heart"))} + 0.05: {gates.get("english_wer")} |
| DNSMOS / UTMOS | {fmt(m.get("dnsmos"))} / {fmt(m.get("utmos"))} | ≥ 3.5 or within 0.3 of real recordings: {gates.get("naturalness")} |
| Spectral roll-off 95 % (Hz) | {fmt(m.get("rolloff99_hz"))} | – |
| ECAPA speaker similarity to reference | {fmt((m.get("ecapa_similarity") or {}).get("uf_rasa"))} (uf) / {fmt((m.get("ecapa_similarity") or {}).get("um_rasa"))} (um) | – |
| Real-time factor | {fmt(m.get("rtf_gpu_or_cpu"))} ({m.get("device", "")}) | – |
| ONNX fp32 / fp16 parity (min waveform corr.) | {fmt((onnx_rep.get("fp32") or {}).get("min_corr"))} / {fmt((onnx_rep.get("fp16") or {}).get("min_corr"))} | ≥ 0.99 |

All gates passed: **{m.get("gates_passed", "n/a")}**. Per-epoch numbers and the full evaluation are in `REPORT.md`.

## Training data

| data | hours | role |
|---|---|---|
| [AI4Bharat Rasa](https://huggingface.co/datasets/ai4bharat/Rasa) Urdu (2 studio speakers, 48 kHz → 24 kHz) | {ur_h:.1f} | primary (speaker ids 0 = female, 1 = male) |
| [LibriTTS-R](https://huggingface.co/datasets/mythicinfinity/libritts_r) train.clean.100 (English, 24 kHz) | {en_h:.1f} | English replay (keeps ɹ θ ð æ ɜ in the model) |

Mix by hours: {data.get("mix_percent_by_hours")}. ASR consistency filter: {json.dumps(data.get("asr", {}), ensure_ascii=False)[:300]}.
Bandwidth audit: {json.dumps({k: v.get("rolloff99_hz") for k, v in (data.get("bandwidth") or {}).items()})} (99 % roll-off in Hz per speaker).

Recipe: Stage 1 (acoustic) {len(st.get("stage1", {}).get("epochs", []))} epochs, Stage 2 (adversarial + diffusion prosody) {len(st.get("stage2", {}).get("epochs", []))} epochs,
variant `{st.get("stage2", {}).get("variant")}`, trained on Kaggle free GPUs (T4 x2) with the
[kikiri-tts](https://github.com/semidark/kikiri-tts) / [Kokoro-Indic-Fine-Tuning](https://github.com/sammy4321/Kokoro-Indic-Fine-Tuning) recipe.
Best Stage-1 epoch {st.get("best_s1_epoch")}, best Stage-2 epoch {st.get("best_s2_epoch")}.

## Limitations

- The accent is that of the two Rasa Urdu speakers (Indian studio Urdu); Pakistani-English words are rendered with Urdu phones.
- Out-of-lexicon words go through a neural G2P (PER {fmt((data.get("g2p") or {}).get("per"))} on held-out WikiPron words);
  short-vowel choice (ə/ɪ/ʊ) and the ر/ڑ distinction in rare words can be wrong. Add words to `lughaat_tts/data/gold_overrides.tsv`.
- Roman Urdu (Latin-script Urdu) is out of scope: such tokens are spoken as English (see `RomanUrduHook`).
- Code-switching prosody at language boundaries is learned from synthetic mixed text only (no real code-switched audio).
- Very short inputs (1–2 words) sound rushed; the pipeline merges short sentences with neighbours.
- Dynamic INT8 ONNX is not shipped: it slows this convolution-heavy vocoder down.

## Licences and attributions

- Model weights and code: Apache-2.0 (inherits Kokoro-82M). Training data licences apply to the data, not the weights, but attribution is required:
  AI4Bharat **Rasa** (CC-BY-4.0) and **LibriTTS-R** (CC-BY-4.0).
- `lughaat_tts/data/lexicon.tsv` is derived from [WikiPron](https://github.com/CUNY-CL/wikipron) / Wiktionary and is **CC-BY-SA-4.0**.
- Built with [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) (Apache-2.0), [misaki](https://github.com/hexgrad/misaki) (Apache-2.0),
  [StyleTTS2](https://github.com/yl4579/StyleTTS2) (MIT), [kikiri-tts](https://github.com/semidark/kikiri-tts) and
  [Kokoro-Indic-Fine-Tuning](https://github.com/sammy4321/Kokoro-Indic-Fine-Tuning) (Apache-2.0).

### Prohibited use

Do not use this model to impersonate real people, to generate deceptive audio, or to produce content that violates the rights of
the Rasa voice talents. The voices are synthetic averages of studio recordings released for TTS research under CC-BY-4.0.

### Citations

```
@inproceedings{{rasa2024, title={{Rasa: Building Expressive Speech Synthesis Systems for Indian Languages in Low-resource Settings}}, author={{AI4Bharat}}, year={{2024}}}}
@inproceedings{{koizumi2023librittsr, title={{LibriTTS-R: A Restored Multi-Speaker Text-to-Speech Corpus}}, author={{Koizumi, Yuma and others}}, booktitle={{Interspeech}}, year={{2023}}}}
```
"""
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(card)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()

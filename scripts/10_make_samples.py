#!/usr/bin/env python3
"""Synthesise samples/ for the model card: Appendix A, A2, A3 for both voices (plan 0.1)."""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from eval.testsets import APPENDIX_A, APPENDIX_A2, APPENDIX_A3  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--voices", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    import soundfile as sf
    from lughaat_tts.pipeline import UrduPipeline
    voices = [v for v in ("uf_rasa", "um_rasa") if os.path.exists(os.path.join(a.voices, f"{v}.pt"))]
    tts = UrduPipeline(voice=voices[0], config_path=a.config, model_path=a.model, voices_dir=a.voices, repo_id="local")
    index = []
    for voice in voices:
        for name, sents, accent in (("A", APPENDIX_A, "auto"), ("A2", APPENDIX_A2, "auto"), ("A3", APPENDIX_A3, "native")):
            for i, s in enumerate(sents):
                fn = f"{name}_{i+1:02d}_{voice}.wav"
                audio = tts(s, voice=voice, english_accent=accent)
                sf.write(os.path.join(a.out, fn), audio, 24000)
                index.append({"file": fn, "set": name, "voice": voice, "text": s, "phonemes": tts.phonemize(s, english_accent=accent).phonemes})
    with open(os.path.join(a.out, "index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=1)
    print(f"wrote {len(index)} samples to {a.out}")


if __name__ == "__main__":
    main()

"""Command line: ``python -m lughaat_tts "متن" -o out.wav`` or ``lughaat-tts --phonemes "متن"``."""
from __future__ import annotations

import argparse
import sys
from typing import Optional


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="lughaat-tts", description="Kokoro-82M Urdu text-to-speech")
    ap.add_argument("text", nargs="?", help="text to speak (Urdu, English or mixed); reads stdin if omitted")
    ap.add_argument("-o", "--out", default="out.wav")
    ap.add_argument("-v", "--voice", default="uf_rasa")
    ap.add_argument("-s", "--speed", type=float, default=1.0)
    ap.add_argument("--accent", default="auto", choices=["auto", "pakistani", "native"])
    ap.add_argument("--no-retroflex", action="store_true", help="keep English t/d alveolar in Pakistani mode")
    ap.add_argument("--repo", default=None, help="HF repo id (default: baked-in or $LUGHAAT_TTS_REPO)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--phonemes", action="store_true", help="only print phonemes and language tags; no model needed")
    a = ap.parse_args(argv)
    text = a.text if a.text is not None else sys.stdin.read()
    if a.phonemes:
        from .codeswitch import default_frontend
        res = default_frontend().phonemize(text, english_accent=a.accent, retroflex_td=not a.no_retroflex)
        print(res.phonemes)
        print(" ".join(f"{t.text}/{t.tag}" for t in res.tokens))
        return 0
    from .pipeline import UrduPipeline
    tts = UrduPipeline(voice=a.voice, device=a.device, repo_id=a.repo, english_accent=a.accent, retroflex_td=not a.no_retroflex)
    tts.save(text, a.out, speed=a.speed)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

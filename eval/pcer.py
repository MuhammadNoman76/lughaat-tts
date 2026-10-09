"""Phonetic CER (PCER) for code-switched sentences (plan 9.2).

Whisper writes English words in mixed speech inconsistently (Latin or Urdu script), so
both the reference text and the ASR output are passed through OUR frontend
(Pakistani accent, so English words become Urdu phones either way), collapsed to coarse
phone classes and compared with a phone-unit edit distance.

Classes merged: t/ʈ, d/ɖ, v/w/ʋ, e/ɛ, o/ɔ, ə/ʌ/ɐ; length marks, aspiration and stress are
ignored; misaki diphthong letters are expanded (A->eɪ, I->aɪ, W->aʊ, Y->ɔɪ, O->oʊ).

``pcer_detail`` also attributes each reference phone to its source word (Urdu / English)
and reports PCER per part plus whether every English reference word survived.
"""
from __future__ import annotations

import os
import sys
import unicodedata
from dataclasses import dataclass
from typing import Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from lughaat_tts.codeswitch import MixedFrontend, default_frontend  # noqa: E402
from lughaat_tts.phoneset import split_units  # noqa: E402

_DIPH = {"A": "eɪ", "I": "aɪ", "W": "aʊ", "Y": "ɔɪ", "O": "oʊ", "Q": "əʊ"}
_MERGE = {"ʈ": "t", "ɖ": "d", "v": "ʋ", "w": "ʋ", "ɛ": "e", "ɔ": "o", "ʌ": "ə", "ɐ": "ə", "ɾ": "r", "ɹ": "r",
          "T": "t", "ᵊ": "ə", "ᵻ": "ɪ", "æ": "e", "ɑ": "ɑ", "a": "ɑ", "i": "i", "u": "u"}
_DROP = set("ːʰˈˌ̃")


def collapse(ph: str) -> list[str]:
    s = unicodedata.normalize("NFD", ph)
    s = "".join(_DIPH.get(c, c) for c in s)
    out = []
    for c in s:
        if c in _DROP or c.isspace() or c in ",.!?;:…—\"()“”":
            continue
        out.append(_MERGE.get(c, c))
    return out


def _align(ref: list[str], hyp: list[str]) -> tuple[int, list[tuple[int, int, str]]]:
    """Edit distance with back-trace. Returns (distance, ops) with ops (i, j, 'M'|'S'|'D'|'I')."""
    n, m = len(ref), len(hyp)
    D = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        D[i][0] = i
    for j in range(1, m + 1):
        D[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i][j] = min(D[i - 1][j] + 1, D[i][j - 1] + 1, D[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))
    ops = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i][j] == D[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            ops.append((i - 1, j - 1, "M" if ref[i - 1] == hyp[j - 1] else "S"))
            i, j = i - 1, j - 1
        elif i > 0 and D[i][j] == D[i - 1][j] + 1:
            ops.append((i - 1, j, "D"))
            i -= 1
        else:
            ops.append((i, j - 1, "I"))
            j -= 1
    return D[n][m], ops[::-1]


@dataclass
class PCERResult:
    pcer: float
    pcer_ur: float
    pcer_en: float
    n_ref: int
    english_words_lost: list[str]
    ref_phones: str
    hyp_phones: str


def pcer_detail(ref_text: str, hyp_text: str, frontend: Optional[MixedFrontend] = None, accent: str = "pakistani") -> PCERResult:
    fe = frontend or default_frontend()
    r = fe.phonemize(ref_text, english_accent=accent)
    h = fe.phonemize(hyp_text, english_accent=accent)
    # per-word phones of the reference with language labels
    ref_units: list[str] = []
    ref_lang: list[str] = []
    word_spans: list[tuple[str, str, int, int]] = []  # (word, lang, start, end)
    for tk in r.tokens:
        if tk.tag == "punct" or not tk.phonemes:
            continue
        units = collapse(tk.phonemes)
        lang = "en" if (tk.tag.startswith("en") or tk.tag in ("roman_ur?", "num_en")) else "ur"
        word_spans.append((tk.text, lang, len(ref_units), len(ref_units) + len(units)))
        ref_units += units
        ref_lang += [lang] * len(units)
    hyp_units = collapse(h.phonemes)
    if not ref_units:
        return PCERResult(float("nan"), float("nan"), float("nan"), 0, [], r.phonemes, h.phonemes)
    dist, ops = _align(ref_units, hyp_units)
    err = {"ur": 0, "en": 0}
    cnt = {"ur": ref_lang.count("ur"), "en": ref_lang.count("en")}
    matched = [False] * len(ref_units)
    for i, j, op in ops:
        if op == "M":
            matched[i] = True
        elif op in ("S", "D"):
            err[ref_lang[i]] += 1
        else:  # insertion: charge to the language of the nearest reference phone
            k = min(max(i, 0), len(ref_units) - 1)
            err[ref_lang[k]] += 1
    lost = []
    for word, lang, s, e in word_spans:
        if lang == "en" and e > s and sum(matched[s:e]) < 0.5 * (e - s):
            lost.append(word)
    return PCERResult(
        pcer=dist / len(ref_units),
        pcer_ur=err["ur"] / cnt["ur"] if cnt["ur"] else float("nan"),
        pcer_en=err["en"] / cnt["en"] if cnt["en"] else float("nan"),
        n_ref=len(ref_units),
        english_words_lost=lost,
        ref_phones=r.phonemes,
        hyp_phones=h.phonemes,
    )


def pcer(ref_text: str, hyp_text: str, frontend: Optional[MixedFrontend] = None) -> float:
    return pcer_detail(ref_text, hyp_text, frontend).pcer


__all__ = ["pcer", "pcer_detail", "PCERResult", "collapse"]

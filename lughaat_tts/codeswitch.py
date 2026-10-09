"""Language tagging and the mixed Urdu/English frontend (plan 4.7).

``MixedFrontend.phonemize(text)`` returns a :class:`FrontendResult` with the final
phoneme string (NFD, vocabulary-checked) and per-token language tags.

Tags: ``ur`` (Arabic script), ``en`` (Latin, English), ``en_acronym`` (spelled out),
``en_acronym_word`` (pronounced as a word), ``roman_ur?`` (Latin token not in the English
dictionary: spoken as English and logged; see :class:`RomanUrduHook`), ``num`` (digits,
resolved by the surrounding language), ``punct``.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Callable, Optional

from .english import EnglishG2P, classify_latin, default_english_g2p
from .g2p import UrduG2P, default_urdu_g2p
from .normalize import (expand_numbers, expand_symbols, has_arabic_script, has_latin,
                        normalize_chars, normalize_punctuation)
from .phoneset import assert_in_vocab, to_lughaat_tts
from .vocab import VOCAB

log = logging.getLogger("lughaat_tts")

_TOKEN_RE = re.compile(
    # Urdu word: Arabic-script letters + harakat + ZWNJ, excluding the punctuation and symbol
    # code points of the block (، ؛ ؟ ٪ ٫ ٬ ٭ ۔) and the digits (converted earlier)
    r"[ؠ-ٟٮ-ۓەۡ-ۯۺ-ۿݐ-ݿﭐ-﷿ﹰ-﻿‌]+"
    r"|[A-Za-z][A-Za-z'’\-]*"                                           # Latin word
    r"|\d+(?:[.:,]\d+)*"                                                 # number
    r"|[^\s]"                                                            # punctuation / other
)
_PUNCT_KEEP = set(",.!?;:—…\"()“”")
_URDU_WORD_RE = re.compile(r"[ؠ-ٟٮ-ۓەۡ-ۯۺ-ۿݐ-ݿﭐ-﷿ﹰ-﻿]")


@dataclass
class Token:
    text: str
    tag: str
    phonemes: str = ""


@dataclass
class FrontendResult:
    text: str
    phonemes: str
    tokens: list[Token] = field(default_factory=list)
    accent: str = "pakistani"

    def __str__(self) -> str:  # pragma: no cover
        tags = " ".join(f"{t.text}/{t.tag}" for t in self.tokens)
        return f"{self.phonemes}\n[{self.accent}] {tags}"


class RomanUrduHook:
    """Plug-in point for Roman-Urdu transliteration (out of scope for v1).

    Subclass and override :meth:`to_urdu` to return Urdu script for a Latin token (or
    None to keep treating it as English)."""

    def to_urdu(self, token: str) -> Optional[str]:
        return None


def tag_tokens(text: str, english: Optional[EnglishG2P] = None, roman_hook: Optional[RomanUrduHook] = None) -> list[Token]:
    toks: list[Token] = []
    for m in _TOKEN_RE.finditer(text):
        t = m.group(0)
        if _URDU_WORD_RE.search(t):
            toks.append(Token(t, "ur"))
        elif has_latin(t):
            kind = classify_latin(t)
            if kind == "en" and english is not None:
                core = t.strip("'’-")
                if core and not english.is_known_word(core):
                    if roman_hook is not None:
                        ur = roman_hook.to_urdu(core)
                        if ur:
                            toks.append(Token(ur, "ur"))
                            continue
                    kind = "roman_ur?"
            toks.append(Token(t, kind))
        elif t[0].isdigit():
            toks.append(Token(t, "num"))
        else:
            toks.append(Token(t, "punct"))
    # resolve digits by neighbouring language (majority of the sentence as a fallback)
    n_ur = sum(1 for t in toks if t.tag == "ur")
    n_en = sum(1 for t in toks if t.tag.startswith("en") or t.tag == "roman_ur?")
    default = "en" if n_en > n_ur else "ur"
    for i, t in enumerate(toks):
        if t.tag != "num":
            continue
        left = next((x.tag for x in reversed(toks[:i]) if x.tag not in ("punct", "num")), None)
        right = next((x.tag for x in toks[i + 1:] if x.tag not in ("punct", "num")), None)
        votes = [x for x in (left, right) if x]
        votes = ["en" if v.startswith("en") or v == "roman_ur?" else "ur" for v in votes]
        if votes.count("en") > votes.count("ur"):
            t.tag = "num_en"
        elif votes.count("ur") > votes.count("en"):
            t.tag = "num_ur"
        else:
            t.tag = "num_" + default
    return toks


class MixedFrontend:
    def __init__(self, urdu: Optional[UrduG2P] = None, english: Optional[EnglishG2P] = None,
                 roman_hook: Optional[RomanUrduHook] = None, british_base: bool = False):
        self._urdu = urdu
        self._english = english
        self.british_base = british_base
        self.roman_hook = roman_hook

    @property
    def urdu(self) -> UrduG2P:
        if self._urdu is None:
            self._urdu = default_urdu_g2p()
        return self._urdu

    @property
    def english(self) -> EnglishG2P:
        if self._english is None:
            self._english = default_english_g2p(self.british_base)
        return self._english

    def choose_accent(self, tokens: list[Token], english_accent: str) -> str:
        if english_accent in ("pakistani", "native"):
            return english_accent
        words = [t for t in tokens if t.tag != "punct"]
        n_en = sum(1 for t in words if t.tag.startswith("en") or t.tag in ("roman_ur?", "num_en"))
        return "native" if words and n_en / len(words) >= 0.8 else "pakistani"

    def phonemize(self, text: str, english_accent: str = "auto", retroflex_td: bool = True) -> FrontendResult:
        s = normalize_chars(text)
        s = expand_symbols(s)
        s = expand_numbers(s)          # Urdu-context digits -> Urdu words; English digits stay
        s = normalize_chars(s)
        # tag before punctuation normalisation so Urdu punctuation survives tokenisation
        tokens = tag_tokens(s, self.english, self.roman_hook)
        accent = self.choose_accent(tokens, english_accent)
        out: list[str] = []
        i = 0
        while i < len(tokens):
            t = tokens[i]
            if t.tag == "ur":
                t.phonemes = self.urdu.word(t.text)
                out.append(t.phonemes)
                i += 1
            elif t.tag.startswith("en") or t.tag in ("roman_ur?", "num_en"):
                # group a run of English tokens (and the punctuation between them) into one span
                j = i
                span: list[Token] = []
                while j < len(tokens) and (tokens[j].tag.startswith("en") or tokens[j].tag in ("roman_ur?", "num_en")
                                           or (tokens[j].tag == "punct" and tokens[j].text in "'’-,"
                                               and j + 1 < len(tokens) and (tokens[j + 1].tag.startswith("en") or tokens[j + 1].tag == "num_en"))):
                    span.append(tokens[j])
                    j += 1
                for tk in span:
                    if tk.tag == "roman_ur?":
                        log.info("Roman-Urdu? token spoken as English: %s", tk.text)
                span_text = " ".join(tk.text for tk in span)
                ps = self.english.phonemize_span(span_text, accent=accent, retroflex_td=retroflex_td)
                if span:
                    span[0].phonemes = ps
                out.append(ps)
                i = j
            elif t.tag == "num_ur":
                # digits that survived (glued to Latin): expand now in Urdu
                t.phonemes = self.urdu(expand_numbers(t.text))
                out.append(t.phonemes)
                i += 1
            else:  # punct
                p = normalize_punctuation(t.text)
                if p and all(ch in _PUNCT_KEEP for ch in p):
                    t.phonemes = p
                    if out and p in ",.!?;:…":
                        out[-1] = out[-1] + p      # attach to the previous word (Kokoro style)
                    else:
                        out.append(p)
                i += 1
        ph = " ".join(x for x in out if x)
        ph = unicodedata.normalize("NFD", ph)
        ph = re.sub(r"\s+", " ", ph).strip()
        ph = re.sub(r"\s+([,.!?;:…])", r"\1", ph)
        bad = [c for c in ph if c not in VOCAB]
        if bad:
            log.warning("dropping out-of-vocabulary symbols %r from %r", sorted(set(bad)), ph)
            ph = "".join(c for c in ph if c in VOCAB)
        return FrontendResult(text=text, phonemes=ph, tokens=tokens, accent=accent)

    def __call__(self, text: str, english_accent: str = "auto", retroflex_td: bool = True) -> str:
        return self.phonemize(text, english_accent, retroflex_td).phonemes


_DEFAULT: Optional[MixedFrontend] = None


def default_frontend() -> MixedFrontend:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = MixedFrontend()
    return _DEFAULT


def phonemize(text: str, english_accent: str = "auto", retroflex_td: bool = True) -> str:
    return default_frontend()(text, english_accent, retroflex_td)


__all__ = ["MixedFrontend", "FrontendResult", "Token", "RomanUrduHook", "tag_tokens", "phonemize", "default_frontend"]

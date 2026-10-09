"""English G2P (misaki) and the Pakistani-English accent mapping (plan 4.7).

* ``EnglishG2P``: lazy wrapper around ``misaki.en.G2P`` (American by default, British
  optional). misaki's gold/silver lexicon covers common words; unknown words go to its
  fallback network (or espeak-ng if available). Nothing here runs Urdu text through misaki.
* ``to_pakistani``: re-maps misaki phonemes onto the Urdu phone set so that an English word
  inside an Urdu sentence is made of sounds the fine-tuned model has heard thousands of times.
* ``to_native``: keeps misaki's phonemes (all already in Kokoro's vocabulary) for English-only
  sentences; works because of the English replay data.
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from typing import Optional

from .vocab import VOCAB

# misaki symbol -> Pakistani-Urdu phone. Longest keys first. (US base; GB extras included.)
_PAK_MAP: list[tuple[str, str]] = sorted({
    "ˈ": "", "ˌ": "",
    "ɜɹ": "ər", "ɜː": "ər", "ɜ": "ə", "ɚ": "ər",
    "θ": "tʰ", "ð": "d",
    "ɹ": "r",
    "ɾ": "ʈ", "T": "ʈ",
    "w": "ʋ", "v": "ʋ",
    "æ": "ɛː", "a": "ɛː",
    "ʌ": "ə", "ə": "ə", "ᵊ": "ə", "ᵻ": "ə", "ɐ": "ə",
    "ɑ": "ɑː", "ɒ": "ɔː", "ɔ": "ɔː",
    "i": "iː", "u": "uː",
    "A": "eː", "O": "oː", "Q": "oː", "I": "ɑɪ", "W": "ɑʊ", "Y": "ɔɪ",
    "ɛ": "ɛ", "ɪ": "ɪ", "ʊ": "ʊ",
    "ʔ": "ʈ",
}.items(), key=lambda kv: -len(kv[0]))

_PAK_T_MAP = {"t": "ʈ", "d": "ɖ"}
_VOWELS_PAK = "əɪʊiueoɑɛɔ"

# English letter names (misaki symbols) for spelled acronyms: USB -> "ju ɛs bi"
LETTER_NAMES = {
    "A": "A", "B": "bi", "C": "si", "D": "di", "E": "i", "F": "ɛf", "G": "ʤi", "H": "Aʧ", "I": "I",
    "J": "ʤA", "K": "kA", "L": "ɛl", "M": "ɛm", "N": "ɛn", "O": "O", "P": "pi", "Q": "kju", "R": "ɑɹ",
    "S": "ɛs", "T": "ti", "U": "ju", "V": "vi", "W": "dʌbᵊlju", "X": "ɛks", "Y": "wI", "Z": "zɛd",
}


def spell_letters(word: str) -> str:
    return " ".join(LETTER_NAMES[c] for c in word.upper() if c in LETTER_NAMES)


def to_pakistani(ps: str, retroflex_td: bool = True, british_base: bool = False) -> str:
    """Map a misaki phoneme string to the Urdu phone set (plan 4.7 step 3 table)."""
    s = ps
    if british_base:
        # drop the British length marks first; the table re-adds Urdu length where it applies
        s = s.replace("ː", "")
    if retroflex_td:
        # English alveolar t/d -> retroflex, BEFORE θ/ð become dental tʰ/d
        s = "".join(_PAK_T_MAP.get(c, c) for c in s)
    for old, new in _PAK_MAP:
        if old in s:
            s = s.replace(old, new)
    s = unicodedata.normalize("NFD", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def to_native(ps: str) -> str:
    """Keep misaki's American phonemes; just drop anything not in Kokoro's vocabulary."""
    s = unicodedata.normalize("NFD", ps)
    return "".join(ch for ch in s if ch in VOCAB or ch == " ")


_ACRONYM_RE = re.compile(r"^[A-Z]{2,6}$")
_PRONOUNCEABLE = re.compile(r"[AEIOU]")
ACRONYM_AS_WORD = {"NADRA", "FIA", "PIA", "WAPDA", "NASA", "UNESCO", "NATO", "OPEC", "FIFA", "PEMRA", "HEC", "NUST", "LUMS", "GIKI", "ISI", "PSL", "IPL", "NAB", "FBR", "KESC", "LESCO", "UET", "PTV", "SIM", "PIN", "RAM", "ROM", "USB"}
ACRONYM_SPELL = {"PTI", "PPP", "PML", "MQM", "USB", "SIM", "PIN", "TV", "CD", "DVD", "PC", "AI", "IT", "HR", "CEO", "CNIC", "NIC", "ID", "UK", "USA", "US", "UAE", "KSA", "PM", "AM", "GPS", "ATM", "SMS", "MBA", "BBA", "PhD", "MA", "BA", "BSc", "MSc", "LLB", "MBBS", "FM", "DJ", "VIP", "PTA", "HD", "UPS", "LED", "LCD", "WiFi", "OK", "ISI", "PSL"}


def classify_latin(token: str) -> str:
    """'en' | 'en_acronym' | 'en_acronym_word' for a Latin-script token."""
    t = token.strip("'\".,!?;:()")
    if t in ACRONYM_SPELL:
        return "en_acronym"
    if t in ACRONYM_AS_WORD:
        return "en_acronym_word"
    if _ACRONYM_RE.match(t):
        if len(t) >= 4 and _PRONOUNCEABLE.search(t[1:-1]) and not t.endswith("S"):
            return "en_acronym_word"
        return "en_acronym"
    return "en"


class EnglishG2P:
    """misaki wrapper; ``__call__(text) -> phoneme string``."""

    def __init__(self, british: bool = False, use_espeak_fallback: Optional[bool] = None):
        self.british = british
        self._g2p = None
        self._fallback = None
        self.use_espeak_fallback = use_espeak_fallback

    def _load(self):
        if self._g2p is not None:
            return self._g2p
        from misaki import en  # type: ignore
        fallback = None
        if self.use_espeak_fallback is not False:
            try:
                from misaki import espeak  # type: ignore
                fallback = espeak.EspeakFallback(british=self.british)
            except Exception:
                fallback = None
        self._g2p = en.G2P(trf=False, british=self.british, fallback=fallback)
        return self._g2p

    @property
    def lexicon(self):
        return self._load().lexicon

    def is_known_word(self, word: str) -> bool:
        lex = self.lexicon
        w = word.strip("'\".,!?;:()")
        if not w:
            return False
        try:
            if lex.is_known(w, "NN"):
                return True
            return lex.is_known(w.lower(), "NN") or lex.is_known(w.capitalize(), "NNP")
        except Exception:
            return w.lower() in lex.golds or w.lower() in lex.silvers

    @lru_cache(maxsize=4096)
    def __call__(self, text: str) -> str:
        g = self._load()
        ps, _ = g(text)
        return ps.strip()

    def phonemize_span(self, text: str, accent: str = "native", retroflex_td: bool = True) -> str:
        """Phonemize an English span and apply the accent mapping."""
        words = text.split()
        parts: list[str] = []
        buf: list[str] = []

        def flush():
            if buf:
                parts.append(self(" ".join(buf)))
                buf.clear()

        for w in words:
            kind = classify_latin(w)
            if kind == "en_acronym":
                flush()
                core = re.sub(r"[^A-Za-z]", "", w)
                parts.append(spell_letters(core))                  # USB -> 'ju ɛs bi'
            elif kind == "en_acronym_word":
                flush()
                lead = re.match(r"^[^A-Za-z]*", w).group(0)
                trail = re.search(r"[^A-Za-z]*$", w).group(0)
                parts.append(self(lead + w.strip("'\".,!?;:()").capitalize() + trail))
            else:
                buf.append(w)
        flush()
        ps = " ".join(p for p in parts if p)
        if accent == "pakistani":
            return to_pakistani(ps, retroflex_td=retroflex_td, british_base=self.british)
        return to_native(ps)


@lru_cache(maxsize=2)
def default_english_g2p(british: bool = False) -> EnglishG2P:
    return EnglishG2P(british=british)


__all__ = ["EnglishG2P", "to_pakistani", "to_native", "classify_latin", "default_english_g2p"]

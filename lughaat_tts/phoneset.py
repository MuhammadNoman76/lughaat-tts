"""Canonical Urdu phone set and the mandatory mapping onto Kokoro's vocabulary.

Plan section 4.3. Everything that leaves the Urdu G2P passes through
``to_lughaat_tts`` and then ``assert_in_vocab``.  KModel silently drops symbols
that are not in its 178-token vocabulary (pitfall 15), so this module is the last
line of defence.

Canonical set
-------------
Consonants  : p b t d ʈ ɖ k ɡ q ʔ f v ʋ s z ʃ ʒ x ɣ h m n ŋ l r ɽ j ʧ ʤ
Aspiration  : a following ʰ  (pʰ bʰ tʰ dʰ ʈʰ ɖʰ kʰ ɡʰ ʧʰ ʤʰ ɽʰ lʰ mʰ nʰ rʰ)
Vowels      : ə ɪ ʊ ɑː iː uː eː oː ɛː ɔː   (+ short i u e o ɑ ɛ ɔ only from the lexicon)
Nasalisation: combining tilde U+0303 after the vowel (before the length mark): ɑ̃ː ẽː ĩː ũː õː
Gemination  : consonant written twice (bb, tt, ...)
Separators  : space between words; punctuation , . ! ? ; :
No stress marks.
"""
from __future__ import annotations

import re
import unicodedata

from .vocab import VOCAB, oov_chars

NASAL = "̃"  # combining tilde
LENGTH = "ː"
ASP = "ʰ"

CONSONANTS = set("pbtdʈɖkɡqʔfvʋszʃʒxɣhmnŋlrɽjʧʤ")
SHORT_VOWELS = set("əɪʊiueoɑɛɔ")
LONG_VOWELS = {"ɑː", "iː", "uː", "eː", "oː", "ɛː", "ɔː"}
VOWEL_CHARS = set("əɪʊiueoɑɛɔ")
PUNCT = set(",.!?;:")

# Multi-character replacements, applied longest-first. Keys may contain
# combining characters; everything is NFD-normalised before matching.
_REPLACEMENTS: list[tuple[str, str]] = [
    # affricates (WikiPron / espeak write them as digraphs, with or without a tie bar)
    ("d͡ʒ", "ʤ"), ("t͡ʃ", "ʧ"), ("dʒ", "ʤ"), ("tʃ", "ʧ"),
    ("t͡s", "ts"), ("d͡z", "dz"),
    # geminated affricates are written d+ʤ / t+ʧ in WikiPron
    ("dʤ", "ʤʤ"), ("tʧ", "ʧʧ"),
    # dental diacritic (t̪ d̪ n̪ l̪) -> plain
    ("̪", ""),
    # breathy-voiced aspiration -> plain aspiration (Urdu has no bʰ/bʱ contrast)
    ("ʱ", "ʰ"),
    # h variants
    ("ɦ", "h"),
    # espeak-ng mistakes for ص ظ ض ذ (retroflex sibilants do not exist in Urdu)
    ("ʂ", "s"), ("ʐ", "z"),
    # single rhotic for ر
    ("ɾ", "r"), ("ɹ", "r"),
    # ascii g -> IPA ɡ
    ("g", "ɡ"),
    # pharyngeal for ع -> glottal stop (then usually deleted by the rules)
    ("ʕ", "ʔ"),
    # labio-velar approximant -> Urdu ʋ
    ("w", "ʋ"),
    # nasal consonant variants
    ("ɳ", "n"), ("ɲ", "n"), ("ɴ", "n"),
    # vowel variants seen in WikiPron / espeak
    ("aː", "ɑː"), ("ä", "ɑ"), ("a", "ə"), ("ʌ", "ə"), ("ɐ", "ə"), ("ᵊ", "ə"), ("ᵻ", "ɪ"),
    ("æː", "ɛː"), ("æ", "ɛ"), ("ɒ", "ɔ"), ("ō", "oː"), ("ɨ", "ɪ"),
    # British-style length on schwa is meaningless in Urdu
    ("əː", "ə"),
    # consonant length marks -> gemination handled below
]

# Characters that are simply deleted (diacritics espeak/WikiPron emit that Kokoro lacks).
_DELETE = set(
    "͡"   # tie bar
    "̯"   # non-syllabic
    "̤"   # breathy voice
    "̈"   # diaeresis
    "̄"   # macron
    "̩"   # syllabic
    "̥"   # voiceless
    "ʷ"   # ʷ labialisation
    "ᵑ"   # ᵑ
    "ˈˌ"  # stress marks ˈ ˌ
    "~‿◌|‖ˑ"
    "ʲ"   # ʲ (in vocab, but Urdu never uses it)
    "​‌‍﻿"
)

_GEMINATE_RE = re.compile(r"([pbtdʈɖkɡqfvʋszʃʒxɣhmnŋlrɽjʧʤ])ː")
_WORD_DOT_RE = re.compile(r"(?<=\S)\.(?=\S)")  # espeak syllable separator inside a word
_MULTISPACE_RE = re.compile(r"\s+")


def to_lughaat_tts(ph: str) -> str:
    """Apply the plan 4.3 mapping to an Urdu phoneme string (one word or a sentence).

    Idempotent: ``to_lughaat_tts(to_lughaat_tts(x)) == to_lughaat_tts(x)``.
    """
    s = unicodedata.normalize("NFD", ph)
    # espeak writes the syllable separator as a dot inside words
    s = _WORD_DOT_RE.sub("", s)
    for old, new in _REPLACEMENTS:
        if old in s:
            s = s.replace(old, new)
    # consonant length mark -> doubled consonant
    s = _GEMINATE_RE.sub(r"\1\1", s)
    # aspiration must directly follow its consonant; collapse doubled marks
    s = s.replace("ʰʰ", "ʰ")
    # nasal tilde must precede the length mark:  ɑ̃ː not ɑ̃ ː / ɑː̃
    s = s.replace(LENGTH + NASAL, NASAL + LENGTH)
    s = "".join(ch for ch in s if ch not in _DELETE)
    s = _MULTISPACE_RE.sub(" ", s).strip()
    return s


def assert_in_vocab(ph: str, context: str = "") -> str:
    """Raise ValueError if any character of *ph* is not in Kokoro's vocabulary."""
    bad = oov_chars(ph)
    if bad:
        detail = ", ".join(f"{c!r} (U+{ord(c):04X})" for c in bad)
        raise ValueError(f"out-of-vocabulary phoneme symbols {detail} in {ph!r} {context}".strip())
    return ph


def is_canonical(ph: str) -> bool:
    """True if *ph* uses only canonical Urdu symbols (plus space and punctuation)."""
    allowed = CONSONANTS | VOWEL_CHARS | PUNCT | {" ", NASAL, LENGTH, ASP}
    return all(ch in allowed for ch in unicodedata.normalize("NFD", ph))


# ---------------------------------------------------------------------------
# Phone-unit tokenisation (used by the neural G2P, the skeleton validator and PCER)
# ---------------------------------------------------------------------------
_UNIT_RE = re.compile(
    r"[pbtdʈɖkɡqʔfvʋszʃʒxɣhmnŋlrɽjʧʤ]ʰ?"     # consonant (+ aspiration)
    r"|[əɪʊiueoɑɛɔ]̃?ː?"                  # vowel (+ nasal) (+ length)
    r"|[,.!?;:]"                                 # punctuation
    r"|\s+"                                      # whitespace
    r"|."                                        # anything else (will be flagged)
)


def split_units(ph: str) -> list[str]:
    """Split a canonical phoneme string into phone units, e.g. 'bʰɑːiː' -> ['bʰ','ɑː','iː']."""
    return [m.group(0) for m in _UNIT_RE.finditer(unicodedata.normalize("NFD", ph)) if not m.group(0).isspace()]


def consonant_units(ph: str) -> list[str]:
    return [u for u in split_units(ph) if u[0] in CONSONANTS]


def vowel_units(ph: str) -> list[str]:
    return [u for u in split_units(ph) if u[0] in VOWEL_CHARS]


__all__ = [
    "to_lughaat_tts", "assert_in_vocab", "is_canonical", "split_units", "consonant_units",
    "vowel_units", "CONSONANTS", "VOWEL_CHARS", "LONG_VOWELS", "NASAL", "LENGTH", "ASP", "VOCAB",
]

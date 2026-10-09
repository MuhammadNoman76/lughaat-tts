"""Consonant-skeleton validator (plan 4.4.5).

Urdu writes every consonant explicitly, so the consonant sequence of a correct
pronunciation is almost fully determined by the letters. A candidate pronunciation
(from espeak, an LLM or the neural G2P) is rejected if its consonant sequence cannot
be aligned to the letters. Vowels are not checked: they are what the lexicon and the
neural model are for.

    >>> skeleton_ok("لڑکی", "ləɽkiː")
    True
    >>> skeleton_ok("لڑکی", "lərkiː")   # espeak collapses ڑ into r
    False
    >>> skeleton_ok("گھر", "ɡər")       # aspiration dropped
    False

(espeak's ʂ/ʐ for ص/ظ are repaired by ``to_lughaat_tts`` before the check, so they pass.)
"""
from __future__ import annotations

from functools import lru_cache

from .normalize import HARAKAT, strip_diacritics
from .phoneset import consonant_units, to_lughaat_tts

# letter -> set of consonant phones it must produce (one of them)
REQUIRED: dict[str, frozenset[str]] = {
    "ب": frozenset({"b"}), "پ": frozenset({"p"}), "ت": frozenset({"t"}), "ٹ": frozenset({"ʈ"}),
    "ث": frozenset({"s"}), "ج": frozenset({"ʤ"}), "چ": frozenset({"ʧ"}), "ح": frozenset({"h"}),
    "خ": frozenset({"x"}), "د": frozenset({"d"}), "ڈ": frozenset({"ɖ"}), "ذ": frozenset({"z"}),
    "ر": frozenset({"r"}), "ڑ": frozenset({"ɽ"}), "ز": frozenset({"z"}), "ژ": frozenset({"ʒ", "z"}),
    "س": frozenset({"s"}), "ش": frozenset({"ʃ"}), "ص": frozenset({"s"}), "ض": frozenset({"z"}),
    "ط": frozenset({"t"}), "ظ": frozenset({"z"}), "غ": frozenset({"ɣ"}), "ف": frozenset({"f"}),
    "ق": frozenset({"q", "k"}), "ک": frozenset({"k"}), "گ": frozenset({"ɡ"}), "ل": frozenset({"l"}),
    "م": frozenset({"m"}), "ن": frozenset({"n", "ŋ", "m"}),
}
# letters that MAY surface as a consonant (or as a vowel / nothing)
OPTIONAL: dict[str, frozenset[str]] = {
    "ہ": frozenset({"h"}), "ۃ": frozenset({"t", "h"}), "ء": frozenset({"ʔ"}), "ئ": frozenset({"ʔ", "j"}),
    "ؤ": frozenset({"ʋ", "ʔ"}), "و": frozenset({"ʋ"}), "ی": frozenset({"j"}), "ے": frozenset(),
    "ع": frozenset({"ʔ"}), "ا": frozenset({"ʔ"}), "آ": frozenset({"ʔ"}), "ں": frozenset({"n"}),
    "ۂ": frozenset(), "ھ": frozenset({"h"}),
}
# phones that may appear without any letter (hiatus glottal stop, glides between vowels)
INSERTABLE = frozenset({"ʔ", "j", "ʋ"})
ASPIRATABLE = set("بپتٹجچدڈکگڑلمنر")


def letter_pattern(word: str) -> list[tuple[frozenset[str], bool]]:
    """Return [(allowed_phones, optional)] for the consonant letters of *word*."""
    tanwin = any(c in word for c in "ًٌٍ")
    w = strip_diacritics(word)
    pattern: list[tuple[frozenset[str], bool]] = []
    i = 0
    while i < len(w):
        ch = w[i]
        nxt = w[i + 1] if i + 1 < len(w) else ""
        if ch in REQUIRED:
            allowed = REQUIRED[ch]
            optional = False
            if nxt == "ھ" and ch in ASPIRATABLE and ch not in "لمنر":
                allowed = frozenset(p + "ʰ" for p in allowed)
                i += 1
            elif ch == "ن" and nxt and nxt not in "اآیےئ":
                # ن before a consonant (or و/ہ) often surfaces only as nasalisation (آنسو ɑ̃ːsuː, بانہوں bɑ̃ːhõː)
                optional = True
            pattern.append((allowed, optional))
        elif ch in OPTIONAL:
            pattern.append((OPTIONAL[ch], True))
        # vowels/diacritics/other characters contribute nothing
        i += 1
    if tanwin or (w.endswith("ا") and len(w) > 2):
        pattern.append((frozenset({"n"}), True))   # tanwin alif: اتفاقاً ɪttɪfɑːqən
    return pattern


@lru_cache(maxsize=65536)
def skeleton_ok(word: str, pron: str) -> bool:
    """True if the consonants of *pron* can be aligned with the letters of *word*."""
    pattern = letter_pattern(word)
    phones = [u for u in consonant_units(to_lughaat_tts(pron))]
    # geminates (shadda / doubled letters) -> allow repeating the previous match
    n, m = len(pattern), len(phones)

    @lru_cache(maxsize=None)
    def match(i: int, j: int) -> bool:
        if i == n and j == m:
            return True
        if j < m and phones[j] in INSERTABLE and match(i, j + 1):
            return True  # phone without letter (hiatus ʔ, glide)
        if i < n:
            allowed, optional = pattern[i]
            if optional and match(i + 1, j):
                return True
            # geminated aspirate written as base + aspirated (اتھان ʊttʰɑːn, بچھو bɪʧʧʰuː)
            if (j + 1 < m and phones[j + 1] == phones[j] + "ʰ" and phones[j + 1] in allowed
                    and match(i + 1, j + 2)):
                return True
            if j < m and (phones[j] in allowed or (phones[j].rstrip("ʰ") in allowed and not optional and phones[j].endswith("ʰ"))):
                # consume one phone, then any identical repeats (gemination: bb, ʧʧ, or t+tʰ)
                k = j + 1
                if match(i + 1, k):
                    return True
                while k < m and (phones[k] == phones[j] or phones[k] == phones[j] + "ʰ"):
                    k += 1
                    if match(i + 1, k):
                        return True
        return False

    return match(0, 0)


def skeleton_report(word: str, pron: str) -> str:
    pattern = letter_pattern(word)
    want = " ".join("/".join(sorted(a)) + ("?" if o else "") for a, o in pattern)
    have = " ".join(consonant_units(to_lughaat_tts(pron)))
    return f"letters want: [{want}]  pron has: [{have}]  ok={skeleton_ok(word, pron)}"


__all__ = ["skeleton_ok", "skeleton_report", "letter_pattern", "REQUIRED", "OPTIONAL"]

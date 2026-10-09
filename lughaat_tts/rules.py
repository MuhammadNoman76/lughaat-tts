"""Rule-based Urdu letter-to-phone converter (the last fallback, plan 4.4.3 / 4.5).

Pure Python, no espeak. Urdu orthography writes consonants and long vowels and
omits most short vowels, so the rules (a) map letters, diacritics and the aspiration
digraphs deterministically and (b) insert short vowels with a syllable heuristic.
Expect ~85 % phone accuracy on unseen words: good enough as a fallback after the
lexicon and the neural G2P, and good enough to generate the consonant skeleton.

An optional espeak-ng candidate (``espeak_candidate``) is provided for lexicon
building (candidate A in plan 4.4) with the plan's fixes applied; it returns None when
espeak-ng is not installed.
"""
from __future__ import annotations

import re
from typing import Optional

from .normalize import HARAKAT
from .phoneset import to_lughaat_tts

CONS = {
    "ب": "b", "پ": "p", "ت": "t", "ٹ": "ʈ", "ث": "s", "ج": "ʤ", "چ": "ʧ", "ح": "h", "خ": "x",
    "د": "d", "ڈ": "ɖ", "ذ": "z", "ر": "r", "ڑ": "ɽ", "ز": "z", "ژ": "ʒ", "س": "s", "ش": "ʃ",
    "ص": "s", "ض": "z", "ط": "t", "ظ": "z", "غ": "ɣ", "ف": "f", "ق": "q", "ک": "k", "گ": "ɡ",
    "ل": "l", "م": "m", "ن": "n", "ء": "ʔ", "ۃ": "t",
}
ASPIRATABLE = set("بپتٹجچدڈکگڑلمنر")
VOWEL_LETTERS = set("اآویےئؤ")
ZABAR, ZER, PESH, SHADDA, JAZM, KHARI = "َ", "ِ", "ُ", "ّ", "ْ", "ٰ"
TANWIN_F, TANWIN_K, TANWIN_D = "ً", "ٍ", "ٌ"

NASAL = {"ɑː": "ɑ̃ː", "iː": "ĩː", "eː": "ẽː", "oː": "õː", "uː": "ũː", "ɛː": "ɛ̃ː", "ɔː": "ɔ̃ː",
         "ə": "ə̃", "ɪ": "ɪ̃", "ʊ": "ʊ̃", "ɑ": "ɑ̃"}

# consonant pairs that can close a syllable (C1C2 . C3) or end a word after a short vowel
LEGAL_CODA = {
    "nd", "nt", "nk", "nɡ", "nʧ", "nʤ", "mp", "mb", "st", "ʃt", "ft", "xt", "qt", "kt", "sk", "sp",
    "rd", "rt", "rk", "rɡ", "rm", "rn", "rb", "rz", "rs", "rʃ", "lk", "lm", "ln", "ld", "lt", "lb",
    "zm", "ʃn", "sm", "hm", "hn", "xm", "ʃk", "ʃq", "sq", "fz", "ʤm", "ʧm", "tm", "dm", "km", "ɡm",
    "bz", "bs", "ks", "ps", "ts", "ʈs", "ɖs", "ns", "nz", "nʃ", "ms", "ls", "rʈ", "rɖ", "nʈ", "nɖ", "ŋk", "ŋɡ",
}
# clusters that may close a word after a LONG vowel (دوست doːst, گوشت ɡoːʃt); others get ə (موسم mɔːsəm)
LONG_OK_FINAL = {
    "st", "ʃt", "nd", "nt", "nʈ", "nɖ", "nʧ", "nʤ", "ŋk", "ŋɡ", "mp", "rd", "rt", "rk", "rn", "rm", "rʈ", "rɖ",
    "ld", "lt", "kt", "ft", "xt", "qt", "ns", "nz", "ks", "ps",
}


class _Unit:
    __slots__ = ("kind", "ph", "src")

    def __init__(self, kind: str, ph: str, src: str = ""):
        self.kind = kind  # 'C' or 'V'
        self.ph = ph
        self.src = src

    def __repr__(self) -> str:  # pragma: no cover
        return f"{self.kind}:{self.ph}"


def _graphemes(word: str) -> list[tuple[str, str]]:
    """Split into (base_letter, diacritics) pairs; ھ after an aspiratable consonant is merged."""
    out: list[tuple[str, str]] = []
    for ch in word:
        if ch in HARAKAT and out:
            out[-1] = (out[-1][0], out[-1][1] + ch)
        else:
            out.append((ch, ""))
    merged: list[tuple[str, str]] = []
    i = 0
    while i < len(out):
        base, dia = out[i]
        if base in ASPIRATABLE and i + 1 < len(out) and out[i + 1][0] == "ھ":
            merged.append((base + "ھ", dia + out[i + 1][1]))
            i += 2
        else:
            merged.append((base, dia))
            i += 1
    return merged


def _diacritic_vowel(dia: str, final: bool) -> Optional[str]:
    if KHARI in dia:
        return "ɑː"
    if ZABAR in dia:
        return "ə"
    if ZER in dia:
        return "eː" if final else "ɪ"   # zer on the last letter = izafat
    if PESH in dia:
        return "ʊ"
    return None


def _letters_to_units(word: str) -> list[_Unit]:
    g = _graphemes(word)
    n = len(g)
    units: list[_Unit] = []
    i = 0

    def base_at(k: int) -> str:
        return g[k][0] if 0 <= k < n else ""

    def last_is_vowel() -> bool:
        return bool(units) and units[-1].kind == "V"

    while i < n:
        base, dia = g[i]
        prev_b, next_b = base_at(i - 1), base_at(i + 1)
        first, last = i == 0, i == n - 1
        core = base.rstrip("ھ")

        if core in CONS and base != "ھ":
            ph = CONS[core] + ("ʰ" if base.endswith("ھ") and core in ASPIRATABLE else "")
            if core == "ء":
                # hamza: glottal stop only between vowels; otherwise silent
                if last_is_vowel() and next_b in VOWEL_LETTERS:
                    units.append(_Unit("C", "ʔ", base))
                i += 1
                continue
            if core == "ۃ":
                units.append(_Unit("V", "ɑː", base))
                i += 1
                continue
            units.append(_Unit("C", ph, base))
            if SHADDA in dia:
                units.append(_Unit("C", ph, base))
            v = _diacritic_vowel(dia, last)
            if v:
                units.append(_Unit("V", v, "dia"))
            elif TANWIN_F in dia:
                units += [_Unit("V", "ə", "dia"), _Unit("C", "n", "dia")]
            elif TANWIN_K in dia:
                units += [_Unit("V", "ɪ", "dia"), _Unit("C", "n", "dia")]
            elif TANWIN_D in dia:
                units += [_Unit("V", "ʊ", "dia"), _Unit("C", "n", "dia")]
            elif JAZM in dia:
                units.append(_Unit("C", "", "jazm"))  # marker: no vowel after
            i += 1
            continue

        if base == "ہ":
            if last and not first and not last_is_vowel() and units:
                units.append(_Unit("V", "ɑː", base))         # کمرہ -> kəmrɑː
            elif last and last_is_vowel() and units[-1].ph in ("ə",):
                units[-1] = _Unit("V", "ɑː", base)            # (C ə) + ہ -> ɑː
            else:
                units.append(_Unit("C", "h", base))
                if SHADDA in dia:
                    units.append(_Unit("C", "h", base))
                v = _diacritic_vowel(dia, last)
                if v:
                    units.append(_Unit("V", v, "dia"))
            i += 1
            continue

        if base == "ۂ":
            units.append(_Unit("V", "ɑː", base))
            units.append(_Unit("V", "eː", base))
            i += 1
            continue

        if base == "آ":
            units.append(_Unit("V", "ɑː", base))
            i += 1
            continue

        if base == "ا":
            if first:
                v = _diacritic_vowel(dia, False)
                if v:
                    units.append(_Unit("V", v, base))
                elif next_b == "ی" and base_at(i + 2) != "" and base_at(i + 2) not in VOWEL_LETTERS and base_at(i + 2) != "ں":
                    units.append(_Unit("V", "iː", base)); i += 1     # ایک, ایسا (eː/iː ambiguous)
                elif next_b == "ی":
                    units.append(_Unit("V", "iː", base)); i += 1
                elif next_b == "و":
                    nxt2 = base_at(i + 2)
                    units.append(_Unit("V", "ɔː" if nxt2 == "ر" and i + 3 == n else "oː", base)); i += 1  # اور
                elif next_b == "ے":
                    units.append(_Unit("V", "eː", base)); i += 1
                else:
                    units.append(_Unit("V", "ə", base))
            else:
                if last_is_vowel() and units[-1].ph == "ə" and units[-1].src == "dia":
                    units[-1] = _Unit("V", "ɑː", base)
                else:
                    units.append(_Unit("V", "ɑː", base))
            i += 1
            continue

        if base == "ع":
            if first:
                v = _diacritic_vowel(dia, False)
                if v:
                    units.append(_Unit("V", v, base))
                elif next_b == "ا":
                    units.append(_Unit("V", "ɑː", base)); i += 1
                elif next_b == "ی":
                    units.append(_Unit("V", "iː", base)); i += 1
                elif next_b == "و":
                    units.append(_Unit("V", "uː", base)); i += 1
                elif next_b == "ے":
                    units.append(_Unit("V", "eː", base)); i += 1
                else:
                    units.append(_Unit("V", "ə", base))
            elif last:
                if last_is_vowel():
                    pass                                      # شروع -> ʃʊruː
                else:
                    units.append(_Unit("V", "ɑː", base))      # موقع -> mɔːqɑː
            else:
                v = _diacritic_vowel(dia, False)
                if v:
                    units.append(_Unit("V", v, base))
                elif last_is_vowel() and units[-1].ph == "ə" and units[-1].src == "ا":
                    units[-1] = _Unit("V", "ɑː", base)        # اعظم -> ɑːzəm, اعلان -> eːlɑːn (lexicon)
                elif last_is_vowel():
                    if next_b not in VOWEL_LETTERS:
                        units.append(_Unit("V", "ə", base))   # شاعر -> ʃɑːər
                elif next_b in VOWEL_LETTERS:
                    units.append(_Unit("V", "ʊ" if next_b == "ا" else "ə", base))  # دعا -> dʊɑː
                else:
                    prev_dia = g[i - 1][1] if i > 0 else ""
                    units.append(_Unit("V", "eː" if ZER in prev_dia else "ɑː", base))  # بعد -> bɑːd
            i += 1
            continue

        if base == "و":
            v = _diacritic_vowel(dia, last)
            if first:
                units.append(_Unit("C", "ʋ", base))
                if v:
                    units.append(_Unit("V", v, "dia"))
            elif prev_b == "خ" and next_b in ("ا", "آ", "ی", "ے", "ش", "د", "ب"):
                # خو: silent wāw (خواب xɑːb, خود xʊd, خوش xʊʃ, خوب xuːb)
                if next_b in ("ا", "آ"):
                    pass
                elif next_b in ("ی", "ے"):
                    units.append(_Unit("V", "iː" if next_b == "ی" else "eː", base)); i += 1
                else:
                    units.append(_Unit("V", "ʊ" if next_b in ("د", "ش") else "uː", base))
            elif last_is_vowel() and (next_b in VOWEL_LETTERS or next_b == "ں" or (not last and next_b not in VOWEL_LETTERS and units[-1].ph.endswith("ː"))):
                units.append(_Unit("C", "ʋ", base))          # دیوار, آواز, ہوا
                if v:
                    units.append(_Unit("V", v, "dia"))
            elif next_b in ("ا", "آ", "ی", "ے") and not last:
                units.append(_Unit("C", "ʋ", base))          # سوال, حوالہ, جوان
                if v:
                    units.append(_Unit("V", v, "dia"))
            elif v == "ʊ" or PESH in (g[i - 1][1] if i > 0 else ""):
                units.append(_Unit("V", "uː", base))
            elif next_b == "ں" and last_is_vowel():
                units.append(_Unit("C", "ʋ", base))
            else:
                units.append(_Unit("V", "oː", base))
            i += 1
            continue

        if base == "ؤ":
            if next_b == "ں":
                units.append(_Unit("V", "uː", base))
            elif last_is_vowel():
                units.append(_Unit("V", "oː", base))          # جاؤ
            else:
                units.append(_Unit("C", "ʋ", base))
            i += 1
            continue

        if base == "ی":
            v = _diacritic_vowel(dia, last)
            if first:
                units.append(_Unit("C", "j", base))
                if v:
                    units.append(_Unit("V", v, "dia"))
            elif last:
                if last_is_vowel() and units[-1].ph in ("ə", "ɪ") and units[-1].src == "dia":
                    units[-1] = _Unit("V", "iː", base)
                else:
                    units.append(_Unit("V", "iː", base))
            elif next_b in ("ا", "آ", "ے", "ؤ", "ئ") or (next_b == "ی"):
                units.append(_Unit("C", "j", base))          # دریا, کیا, پیار
                if v:
                    units.append(_Unit("V", v, "dia"))
            elif last_is_vowel() and units[-1].ph.endswith("ː"):
                units.append(_Unit("C", "j", base))          # after a long vowel: dɑːjə
            elif next_b == "ں":
                units.append(_Unit("V", "eː", base))         # یں -> ẽː (plural / verb endings)
            else:
                units.append(_Unit("V", "iː", base))         # کریم, تین
            i += 1
            continue

        if base == "ے":
            units.append(_Unit("V", "eː", base))
            i += 1
            continue

        if base == "ئ":
            if next_b in ("ے",):
                pass                                          # جائے -> ʤɑːeː
            elif next_b == "ی":
                pass                                          # کوئی -> koːiː
            elif last_is_vowel():
                if next_b and next_b not in VOWEL_LETTERS and next_b != "ں":
                    units.append(_Unit("V", "ɪ", base))      # لائن -> lɑːɪn
            elif next_b in VOWEL_LETTERS:
                units.append(_Unit("V", "ɪ", base))          # لئے -> lɪeː
            else:
                units.append(_Unit("V", "ə", base))          # مسئلہ -> məsələ
            i += 1
            continue

        if base == "ں":
            if last_is_vowel():
                ph = units[-1].ph
                units[-1] = _Unit("V", NASAL.get(ph, ph + "̃"), base)
            elif last:
                units.append(_Unit("V", "ə̃", base))
            else:
                units.append(_Unit("C", "n", base))
            i += 1
            continue

        if base == "ھ":
            units.append(_Unit("C", "h", base))
            i += 1
            continue

        # unknown character (Latin, digit, symbol): ignore
        i += 1
    return units


def _insert_short_vowels(units: list[_Unit]) -> list[str]:
    """Syllable heuristic: break consonant runs with ə."""
    # drop jazm markers but remember them
    seq: list[_Unit] = []
    no_vowel_after: set[int] = set()
    for u in units:
        if u.src == "jazm":
            no_vowel_after.add(len(seq) - 1)
            continue
        seq.append(u)
    if not seq:
        return []
    out: list[str] = []
    i = 0
    n = len(seq)
    # find runs
    runs: list[tuple[int, int]] = []  # [start, end) of consonant runs
    while i < n:
        if seq[i].kind == "C":
            j = i
            while j < n and seq[j].kind == "C":
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    insert_after: set[int] = set()
    for (s, e) in runs:
        length = e - s
        at_start = s == 0
        at_end = e == n
        cons = [seq[k].ph.rstrip("ʰ") for k in range(s, e)]
        if at_start and at_end:
            # all-consonant word: C ə C, C ə C C, C ə C ə C ...
            k = s
            while k < e - 1:
                pair = cons[k - s + 1] + cons[k - s + 2] if k + 2 < e else ""
                insert_after.add(k)
                k += 2 if (k + 2 < e and pair in LEGAL_CODA) else 1
                if k == e - 1:
                    break
            if length >= 2 and (e - 2) not in insert_after and (cons[-2] + cons[-1]) not in LEGAL_CODA and length > 2:
                insert_after.add(e - 2)
            continue
        if at_start:
            if length == 1:
                continue  # C + vowel: nothing to insert
            # no initial clusters in Urdu: C1 ə, then treat the rest as a medial run
            insert_after.add(s)
            s2 = s + 1
            rem = e - s2
            if rem >= 3:
                if cons[1] + cons[2] in LEGAL_CODA:
                    insert_after.add(s2 + 1)
                else:
                    insert_after.add(s2)
            continue
        if at_end:
            if length == 1:
                continue
            prev_v = seq[s - 1].ph
            long_before = prev_v.endswith("ː")
            if length == 2:
                cluster = cons[0] + cons[1]
                if (cluster in LONG_OK_FINAL) if long_before else (cluster in LEGAL_CODA):
                    continue
                insert_after.add(s)
            elif length == 3:
                if cons[0] + cons[1] in LEGAL_CODA:
                    insert_after.add(s + 1)
                elif cons[1] + cons[2] in LEGAL_CODA:
                    insert_after.add(s)
                else:
                    insert_after.add(s)
                    insert_after.add(s + 1)
            else:
                insert_after.add(s)
                insert_after.add(s + 2)
            continue
        # medial run between vowels
        if length <= 2:
            continue
        if length == 3:
            if cons[0] + cons[1] in LEGAL_CODA:
                insert_after.add(s + 1)
            else:
                insert_after.add(s)
        else:
            insert_after.add(s + 1)
            if length >= 5:
                insert_after.add(s + 3)
    for k, u in enumerate(seq):
        out.append(u.ph)
        if k in insert_after and k not in no_vowel_after and u.kind == "C":
            out.append("ə")
    return out


def rules_phonemize_word(word: str) -> str:
    """Urdu word (with or without diacritics) -> canonical phoneme string."""
    units = _letters_to_units(word)
    phones = _insert_short_vowels(units)
    return to_lughaat_tts("".join(phones))


# ---------------------------------------------------------------------------
# espeak-ng candidate (optional; lexicon building only)
# ---------------------------------------------------------------------------
_ESPEAK = None


def _get_espeak():
    global _ESPEAK
    if _ESPEAK is None:
        try:
            from phonemizer.backend import EspeakBackend  # type: ignore
            try:
                from phonemizer.backend.espeak.wrapper import EspeakWrapper  # type: ignore
                import espeakng_loader  # type: ignore
                EspeakWrapper.set_library(espeakng_loader.get_library_path())
                EspeakWrapper.set_data_path(espeakng_loader.get_data_path())
            except Exception:
                pass
            _ESPEAK = EspeakBackend("ur", preserve_punctuation=False, with_stress=False)
        except Exception:
            _ESPEAK = False
    return _ESPEAK or None


def espeak_candidate(word: str) -> Optional[str]:
    """espeak-ng 'ur' + the plan 4.4.3 fixes. None if espeak-ng is unavailable."""
    be = _get_espeak()
    if be is None:
        return None
    try:
        raw = be.phonemize([word], strip=True)[0]
    except Exception:
        return None
    if not raw:
        return None
    s = raw.replace("r.h", "ɽʰ")
    s = re.sub(r"\.(?=\S)", "", s)
    # positional fix for ڑ: espeak collapses it into r
    if "ڑ" in word and "ɽ" not in s:
        n_r_letters = word.count("ر") + word.count("ڑ")
        rs = [m.start() for m in re.finditer("[rɾ]", s)]
        if len(rs) == n_r_letters:
            letters = [ch for ch in word if ch in "رڑ"]
            s_list = list(s)
            for pos, letter in zip(rs, letters):
                if letter == "ڑ":
                    s_list[pos] = "ɽ"
            s = "".join(s_list)
    return to_lughaat_tts(s)


__all__ = ["rules_phonemize_word", "espeak_candidate", "CONS", "LEGAL_CODA"]

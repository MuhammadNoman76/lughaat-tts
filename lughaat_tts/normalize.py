"""Urdu text normaliser (plan 4.2).

Steps
-----
1. Unicode NFC + character unification (Arabic -> Urdu code points), tatweel and
   zero-width removal (ZWNJ kept inside words).
2. Numbers -> Urdu words, context aware (years, times, decimals, percentages, ordinals,
   currency). Urdu-script and Arabic-Indic digits are converted first (pitfall 14).
3. Abbreviations and symbols.
4. Latin-script tokens are left untouched here: ``codeswitch`` tags them ``en`` and
   routes them to the English frontend. Digits inside an English span are expanded in
   English by misaki, so :func:`expand_numbers` only touches digits in Urdu context.
5. Sentence splitting on ۔ ؟ ! . ? and newlines.
"""
from __future__ import annotations

import re
import unicodedata

from . import numbers as num

# --- 1. character unification -------------------------------------------------
_CHAR_MAP = {
    "ي": "ی",  # ي -> ی
    "ى": "ی",  # ى -> ی
    "ې": "ی",  # ې -> ی
    "ك": "ک",  # ك -> ک
    "ڪ": "ک",  # ڪ -> ک
    "ه": "ہ",  # ه -> ہ  (ھ U+06BE is kept: it marks aspiration)
    "ۀ": "ۂ",  # ۀ -> ۂ
    "أ": "ا",  # أ -> ا
    "إ": "ا",  # إ -> ا
    "ٱ": "ا",  # ٱ -> ا
    "ة": "ۃ",  # ة -> ۃ
    "٬": ",",       # arabic thousands separator
    "٫": ".",       # arabic decimal separator
    "’": "'", "‘": "'", "“": '"', "”": '"',
    " ": " ", " ": " ", " ": " ",
}
_REMOVE = set("ـ‍‎‏‪‫‬﻿­")  # tatweel, ZWJ, bidi marks
_ZWNJ = "‌"

HARAKAT = set("ًٌٍَُِّْٰٕٖٓٔٗ٘")

_URDU_LETTER_RE = re.compile(r"[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]")
_LATIN_RE = re.compile(r"[A-Za-z]")


def normalize_chars(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    out = []
    for ch in text:
        if ch in _REMOVE:
            continue
        out.append(_CHAR_MAP.get(ch, ch))
    s = "".join(out)
    # ZWNJ only matters inside a word; strip it at boundaries
    s = re.sub("(^|\\s)" + _ZWNJ + "+|" + _ZWNJ + "+(\\s|$)", lambda m: (m.group(1) or "") + (m.group(2) or ""), s)
    s = s.replace("\t", " ")
    s = re.sub(r"[ ]{2,}", " ", s)
    return s.strip()


def strip_diacritics(word: str) -> str:
    return "".join(ch for ch in word if ch not in HARAKAT)


def has_arabic_script(s: str) -> bool:
    return bool(_URDU_LETTER_RE.search(s))


def has_latin(s: str) -> bool:
    return bool(_LATIN_RE.search(s))


# --- 2. numbers -----------------------------------------------------------------
_YEAR_AFTER = re.compile(r"^\s*(ء|ع|میں|کو|کا|کی|کے|سے|تک|والے|والا|والی)(?=\s|$|[۔،؟!.,?])")
_YEAR_BEFORE = re.compile(r"(سن|سال|جنوری|فروری|مارچ|اپریل|مئی|جون|جولائی|اگست|ستمبر|اکتوبر|نومبر|دسمبر)\s*$")
_ORDINAL_AFTER = re.compile(r"^\s*(واں|ویں|وان|وین)")
_TIME_RE = re.compile(r"(\d{1,2}):(\d{2})(?::\d{2})?")
_DATE_RE = re.compile(r"(?<!\d)(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4})(?!\d)")
_NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")
_PERCENT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(%|٪|فیصد)")
_MONTHS = ["جنوری", "فروری", "مارچ", "اپریل", "مئی", "جون", "جولائی", "اگست", "ستمبر", "اکتوبر", "نومبر", "دسمبر"]
_BREAK_AFTER = (" ", "۔", ".", ",", "،", "?", "؟", "!", ":", ";", "واں", "ویں", ")", "\n")


def _number_token_to_words(tok: str, before: str, after: str) -> str:
    """Verbalise one ASCII number *tok* given its left/right text context (Urdu)."""
    neg = tok.startswith("-")
    tok = tok.lstrip("+-")
    if "." in tok:
        words = num.decimal_words(tok)
    else:
        n = int(tok)
        is_year = (
            (len(tok) == 4 and 1000 <= n <= 2199)
            and (bool(_YEAR_AFTER.match(after)) or bool(_YEAR_BEFORE.search(before)))
        )
        if _ORDINAL_AFTER.match(after):
            words = num.cardinal(n)  # the suffix واں stays in the text and attaches: 'پانچواں'
        elif is_year:
            words = num.year_words(n)
        else:
            words = num.cardinal(n)
    return ("منفی " + words) if neg else words


def _latin_neighbour(side: str) -> bool:
    toks = side.split()
    return bool(toks) and bool(_LATIN_RE.search(toks[0]))


def expand_numbers(text: str) -> str:
    """Replace every digit sequence in Urdu context with Urdu words. Latin digits adjacent
    to Latin words (an English span) are left for the English frontend."""
    s = num.to_ascii_digits(text)
    s = num.strip_thousands_separators(s)

    # percentages
    s = _PERCENT_RE.sub(lambda m: num.decimal_words(m.group(1)) + " فیصد", s)

    # dates dd/mm/yyyy
    def _date(m: re.Match) -> str:
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        month = _MONTHS[mo - 1] if 1 <= mo <= 12 else num.cardinal(mo)
        return f"{num.cardinal(d)} {month} {num.year_words(y)}"
    s = _DATE_RE.sub(_date, s)

    # times hh:mm
    s = _TIME_RE.sub(lambda m: num.time_words(int(m.group(1)), int(m.group(2))), s)

    # ordinal suffix glued to digits: "5واں"
    s = re.sub(r"(\d+)(واں|ویں)", lambda m: num.cardinal(int(m.group(1))) + m.group(2), s)

    out, pos = [], 0
    for m in _NUM_RE.finditer(s):
        before, after = s[:m.start()], s[m.end():]
        out.append(s[pos:m.start()])
        left_tok = before.rstrip().split()[-1] if before.strip() else ""
        right_tok = after.lstrip().split()[0] if after.strip() else ""
        latin_ctx = bool(_LATIN_RE.search(left_tok)) or bool(_LATIN_RE.search(right_tok))
        arabic_ctx = has_arabic_script(left_tok) or has_arabic_script(right_tok)
        glued_latin = (before and _LATIN_RE.match(before[-1])) or (after and _LATIN_RE.match(after[0]))
        if (latin_ctx and not arabic_ctx) or glued_latin:
            out.append(m.group(0))  # English span: misaki verbalises it
        else:
            words = _number_token_to_words(m.group(0), before, after)
            if before and not before.endswith((" ", "(", "\n")):
                words = " " + words
            if after and not after.startswith(_BREAK_AFTER):
                words = words + " "
            out.append(words)
            # the Hijri/CE year marker ء after a year is written, not spoken: ۱۹۴۷ء -> انیس سو سینتالیس
            ym = re.match(r"\s*ء(?=\s|$|[۔،؟!.,?;:])", after)
            if ym:
                pos = m.end() + ym.end()
                continue
        pos = m.end()
    out.append(s[pos:])
    res = "".join(out)
    res = re.sub(r"[ ]{2,}", " ", res)
    res = re.sub(r" +([۔،؟!.,?;:])", r"\1", res)
    return res.strip()


# --- 3. symbols, URLs -------------------------------------------------------------
_SYMBOLS = {
    "%": " فیصد ", "٪": " فیصد ", "&": " اور ", "+": " جمع ", "=": " برابر ", "#": " نمبر ",
    "@": " ایٹ ", "₹": " روپے ", "₨": " روپے ", "$": " ڈالر ", "€": " یورو ", "£": " پاؤنڈ ",
    "°": " ڈگری ", "×": " ضرب ", "÷": " تقسیم ", "→": " ", "←": " ", "•": " ", "·": " ",
    "/": " ", "*": " ", "_": " ", "~": " ", "^": " ", "|": " ", "<": " ", ">": " ", "{": " ", "}": " ",
    "[": " ", "]": " ",
}
_URL_RE = re.compile(r"(https?://\S+|www\.\S+|[\w.+-]+@[\w-]+\.[\w.]+)")


def expand_symbols(text: str) -> str:
    def _url(m: re.Match) -> str:
        u = m.group(0)
        u = re.sub(r"^https?://", "", u).replace("www.", "")
        u = (u.replace("@", " at ").replace(".", " dot ").replace("/", " slash ")
              .replace("-", " dash ").replace("_", " underscore "))
        return " " + u + " "
    s = _URL_RE.sub(_url, text)
    # "Rs. 2500" -> "2500 روپے"   "$ 20" -> "20 ڈالر"
    s = re.sub(r"(Rs\.?|₨|PKR)\s*(\d[\d,]*(?:\.\d+)?)", r"\2 روپے", s, flags=re.IGNORECASE)
    s = re.sub(r"\$\s*(\d[\d,]*(?:\.\d+)?)", r"\1 ڈالر", s)
    # keep dd/mm/yyyy dates intact for expand_numbers (the "/" symbol is otherwise a separator)
    s = re.sub(r"(?<!\d)(\d{1,2})/(\d{1,2})/(\d{4})(?!\d)", r"\1-\2-\3", s)
    for k, v in _SYMBOLS.items():
        s = s.replace(k, v)
    s = re.sub(r"[ ]{2,}", " ", s)
    return s.strip()


# --- 5. sentence splitting -------------------------------------------------------
_SENT_END = re.compile(r"(?<=[۔؟!?.])\s+|\n+")
_PUNCT_MAP = {"۔": ".", "؟": "?", "،": ",", "؛": ";", "٭": "", "–": ",", "-": " ", "'": "", '"': ""}


def split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENT_END.split(text)]
    return [p for p in parts if p]


def normalize_punctuation(text: str) -> str:
    """Map Urdu punctuation to the ASCII forms Kokoro was trained with."""
    return "".join(_PUNCT_MAP.get(ch, ch) for ch in text)


def normalize(text: str, numbers: bool = True) -> str:
    """Full normaliser: characters -> symbols -> numbers. Latin spans are left untouched."""
    s = normalize_chars(text)
    s = expand_symbols(s)
    if numbers:
        s = expand_numbers(s)
    s = normalize_chars(s)
    return s


__all__ = [
    "normalize", "normalize_chars", "normalize_punctuation", "expand_numbers", "expand_symbols",
    "split_sentences", "strip_diacritics", "has_arabic_script", "has_latin", "HARAKAT",
]

"""Urdu number verbalisation (plan 4.2.2).

``num2words`` has no Urdu, so the 0-99 table is hand-written (Urdu numerals below 100 are
irregular). Larger numbers use سو / ہزار / لاکھ / کروڑ / ارب / کھرب.

Public helpers
--------------
cardinal(n)                      -> 'پچیس'
ordinal(n)                       -> 'پانچواں'
decimal_words('38.5')            -> 'اڑتیس اعشاریہ پانچ'
year_words(1947)                 -> 'انیس سو سینتالیس'
time_words(7, 45)                -> 'سات بج کر پینتالیس منٹ'
to_ascii_digits('۲۰۲۶')          -> '2026'
"""
from __future__ import annotations

import re

_UNITS = [
    "صفر", "ایک", "دو", "تین", "چار", "پانچ", "چھ", "سات", "آٹھ", "نو",
    "دس", "گیارہ", "بارہ", "تیرہ", "چودہ", "پندرہ", "سولہ", "سترہ", "اٹھارہ", "انیس",
    "بیس", "اکیس", "بائیس", "تیئیس", "چوبیس", "پچیس", "چھبیس", "ستائیس", "اٹھائیس", "انتیس",
    "تیس", "اکتیس", "بتیس", "تینتیس", "چونتیس", "پینتیس", "چھتیس", "سینتیس", "اڑتیس", "انتالیس",
    "چالیس", "اکتالیس", "بیالیس", "تینتالیس", "چوالیس", "پینتالیس", "چھیالیس", "سینتالیس", "اڑتالیس", "انچاس",
    "پچاس", "اکیاون", "باون", "ترپن", "چون", "پچپن", "چھپن", "ستاون", "اٹھاون", "انسٹھ",
    "ساٹھ", "اکسٹھ", "باسٹھ", "تریسٹھ", "چونسٹھ", "پینسٹھ", "چھیاسٹھ", "سڑسٹھ", "اڑسٹھ", "انہتر",
    "ستر", "اکہتر", "بہتر", "تہتر", "چوہتر", "پچہتر", "چھہتر", "ستتر", "اٹھتر", "اناسی",
    "اسی", "اکیاسی", "بیاسی", "تراسی", "چوراسی", "پچاسی", "چھیاسی", "ستاسی", "اٹھاسی", "نواسی",
    "نوے", "اکانوے", "بانوے", "ترانوے", "چورانوے", "پچانوے", "چھیانوے", "ستانوے", "اٹھانوے", "ننانوے",
]
assert len(_UNITS) == 100

HUNDRED, THOUSAND, LAKH, CRORE, ARAB, KHARAB = "سو", "ہزار", "لاکھ", "کروڑ", "ارب", "کھرب"

_ORDINALS = {1: "پہلا", 2: "دوسرا", 3: "تیسرا", 4: "چوتھا", 6: "چھٹا", 9: "نواں"}

URDU_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
ARABIC_INDIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"
_DIGIT_MAP = {c: str(i) for i, c in enumerate(URDU_DIGITS)}
_DIGIT_MAP.update({c: str(i) for i, c in enumerate(ARABIC_INDIC_DIGITS)})
_DIGIT_MAP["٫"] = "."   # Arabic decimal separator
_DIGIT_MAP["٬"] = ","   # Arabic thousands separator


def to_ascii_digits(s: str) -> str:
    return "".join(_DIGIT_MAP.get(c, c) for c in s)


def _below_thousand(n: int) -> str:
    assert 0 <= n < 1000
    if n < 100:
        return _UNITS[n]
    h, r = divmod(n, 100)
    out = (_UNITS[h] + " " + HUNDRED) if h > 1 else (_UNITS[1] + " " + HUNDRED)
    if r:
        out += " " + _UNITS[r]
    return out


def cardinal(n: int) -> str:
    """Integer -> Urdu words. Uses the South-Asian grouping (lakh, crore, arab, kharab)."""
    if n < 0:
        return "منفی " + cardinal(-n)
    if n < 1000:
        return _below_thousand(n)
    parts: list[str] = []
    kharab, n = divmod(n, 10**11)
    arab, n = divmod(n, 10**9)
    crore, n = divmod(n, 10**7)
    lakh, n = divmod(n, 10**5)
    thousand, rest = divmod(n, 1000)
    for value, name in ((kharab, KHARAB), (arab, ARAB), (crore, CRORE), (lakh, LAKH), (thousand, THOUSAND)):
        if value:
            parts.append(_below_thousand(value) + " " + name)
    if rest:
        parts.append(_below_thousand(rest))
    return " ".join(parts)


def ordinal(n: int) -> str:
    if n in _ORDINALS:
        return _ORDINALS[n]
    return cardinal(n) + "واں"


def digits_individually(digits: str) -> str:
    return " ".join(_UNITS[int(d)] for d in digits if d.isdigit())


def decimal_words(num: str) -> str:
    """'38.5' -> 'اڑتیس اعشاریہ پانچ'. Digits after the point are read one by one."""
    num = to_ascii_digits(num).replace(",", "")
    neg = num.startswith("-")
    num = num.lstrip("+-")
    if "." in num:
        whole, frac = num.split(".", 1)
        words = cardinal(int(whole or "0")) + " اعشاریہ " + digits_individually(frac)
    else:
        words = cardinal(int(num))
    return ("منفی " + words) if neg else words


def year_words(y: int) -> str:
    """Years 1100-1999 are read as 'X سو Y' (انیس سو سینتالیس); others as cardinals."""
    if 1100 <= y <= 1999:
        h, r = divmod(y, 100)
        return _UNITS[h] + " " + HUNDRED + ((" " + _UNITS[r]) if r else "")
    return cardinal(y)


def time_words(h: int, m: int) -> str:
    h = h % 24
    h12 = h % 12 or 12
    if m == 0:
        return _UNITS[h12] + " بجے"
    if m == 30:
        return ("ساڑھے " + _UNITS[h12]) if h12 not in (1, 2) else ("ڈیڑھ" if h12 == 1 else "ڈھائی")
    if m == 15:
        return "سوا " + _UNITS[h12]
    if m == 45:
        nxt = (h12 % 12) + 1
        return "پونے " + _UNITS[nxt]
    return _UNITS[h12] + " بج کر " + _UNITS[m] + " منٹ"


_SEP_RE = re.compile(r"(?<=\d)[,٬](?=\d{3}\b)")


def strip_thousands_separators(s: str) -> str:
    return _SEP_RE.sub("", s)


__all__ = [
    "cardinal", "ordinal", "decimal_words", "year_words", "time_words", "digits_individually",
    "to_ascii_digits", "strip_thousands_separators", "URDU_DIGITS",
]

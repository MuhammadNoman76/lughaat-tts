import pytest

from lughaat_tts import numbers as num


@pytest.mark.parametrize("n,words", [
    (0, "صفر"), (1, "ایک"), (7, "سات"), (11, "گیارہ"), (19, "انیس"), (20, "بیس"), (25, "پچیس"), (38, "اڑتیس"),
    (45, "پینتالیس"), (47, "سینتالیس"), (59, "انسٹھ"), (70, "ستر"), (89, "نواسی"), (99, "ننانوے"),
    (100, "ایک سو"), (101, "ایک سو ایک"), (250, "دو سو پچاس"), (999, "نو سو ننانوے"),
    (1000, "ایک ہزار"), (2500, "دو ہزار پانچ سو"), (2026, "دو ہزار چھبیس"),
    (100000, "ایک لاکھ"), (2500000, "پچیس لاکھ"), (10000000, "ایک کروڑ"), (1000000000, "ایک ارب"),
    (-5, "منفی پانچ"),
])
def test_cardinal(n, words):
    assert num.cardinal(n) == words


def test_years_and_decimals():
    assert num.year_words(1947) == "انیس سو سینتالیس"
    assert num.year_words(2026) == "دو ہزار چھبیس"
    assert num.year_words(1999) == "انیس سو ننانوے"
    assert num.decimal_words("38.5") == "اڑتیس اعشاریہ پانچ"
    assert num.decimal_words("3.14") == "تین اعشاریہ ایک چار"
    assert num.decimal_words("-2.5") == "منفی دو اعشاریہ پانچ"


def test_time_and_ordinals():
    assert num.time_words(7, 45) == "پونے آٹھ"
    assert num.time_words(3, 0) == "تین بجے"
    assert num.time_words(4, 30) == "ساڑھے چار"
    assert num.time_words(1, 30) == "ڈیڑھ"
    assert num.time_words(10, 7) == "دس بج کر سات منٹ"
    assert num.ordinal(1) == "پہلا" and num.ordinal(5) == "پانچواں" and num.ordinal(12) == "بارہواں"


def test_digit_conversion():
    assert num.to_ascii_digits("۲۰۲۶") == "2026"
    assert num.to_ascii_digits("٠١٢") == "012"
    assert num.strip_thousands_separators("2,500,000") == "2500000"

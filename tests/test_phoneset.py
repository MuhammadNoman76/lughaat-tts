import unicodedata

import pytest

from lughaat_tts.phoneset import to_lughaat_tts, assert_in_vocab, is_canonical, split_units
from lughaat_tts.vocab import VOCAB, oov_chars, N_TOKEN


def test_vocab_shape():
    assert N_TOKEN == 178
    assert len(VOCAB) == 114
    assert VOCAB["ʰ"] == 162 and VOCAB["ɽ"] == 129 and VOCAB["q"] == 59 and VOCAB["̃"] == 17
    for missing in ("ɦ", "ʱ", "ʐ", "g"):
        assert missing not in VOCAB


@pytest.mark.parametrize("raw,expected", [
    ("p ɑː k ɪ s t̪ ɑː n".replace(" ", ""), "pɑːkɪstɑːn"),
    ("ɡʱəɾ", "ɡʰər"),
    ("t͡ʃʰoːʈɑː", "ʧʰoːʈɑː"),
    ("d͡ʒɑːnɑː", "ʤɑːnɑː"),
    ("nəɦĩː", "nəhĩː"),
    ("xuːbʂuːrat", "xuːbsuːrət"),
    ("ʋəzireː aːʐəm", "ʋəzireː ɑːzəm"),
    ("pʌr.hi", "pərhi"),
    ("mẽː", "mẽː"),
    ("gʰər", "ɡʰər"),
    ("mːəhəbbət", "mmə" + "həbbət"),
])
def test_mapping(raw, expected):
    got = to_lughaat_tts(raw)
    assert got == unicodedata.normalize("NFD", expected)
    assert not oov_chars(got)


def test_idempotent_and_in_vocab():
    s = to_lughaat_tts("bʱɑːiː ɡʱər mẽ dʱoːp hɛ")
    assert to_lughaat_tts(s) == s
    assert_in_vocab(s)
    assert is_canonical(s)


def test_precomposed_nasal_vowels_become_combining():
    s = to_lughaat_tts("ĩː ẽː õː ũː ã")
    assert "̃" in s
    for ch in s:
        assert ch in VOCAB or ch == " ", repr(ch)


def test_split_units():
    assert split_units("bʰɑːiː") == ["bʰ", "ɑː", "iː"]
    assert split_units("nəhĩː") == ["n", "ə", "h", unicodedata.normalize("NFD", "ĩː")]
    assert split_units("ʧʰoːʈɑː") == ["ʧʰ", "oː", "ʈ", "ɑː"]


def test_oov_raises():
    with pytest.raises(ValueError):
        assert_in_vocab("ɦə")

@pytest.mark.parametrize("raw,expected", [
    ("ãː", "ɑ̃ː"), ("ãː", "ɑ̃ː"), ("aː̃", "ɑ̃ː"),
    ("ä", "ɑ"), ("ä", "ɑ"), ("ō", "oː"), ("ō", "oː"),
])
def test_unicode_vowels_preserve_quality_and_length(raw, expected):
    result = to_lughaat_tts(raw)
    assert result == unicodedata.normalize("NFD", expected)
    assert to_lughaat_tts(result) == result
    assert_in_vocab(result)

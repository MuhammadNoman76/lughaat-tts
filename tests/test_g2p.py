import unicodedata

import pytest

from eval.testsets import APPENDIX_B
from lughaat_tts.g2p import UrduG2P
from lughaat_tts.lexicon import default_lexicon
from lughaat_tts.rules import rules_phonemize_word
from lughaat_tts.skeleton import skeleton_ok
from lughaat_tts.vocab import oov_chars


@pytest.fixture(scope="module")
def g2p():
    return UrduG2P()


@pytest.mark.parametrize("word,expected", list(APPENDIX_B.items()))
def test_appendix_b_gold(g2p, word, expected):
    got = " ".join(g2p.word(w) for w in word.split())
    assert got == unicodedata.normalize("NFD", expected)


def test_lexicon_loaded():
    lex = default_lexicon()
    assert len(lex) > 5000
    assert lex.lookup("پاکستان") == "pɑːkɪstɑːn"
    assert lex.lookup("ہے") == "hɛː"
    # diacritised spelling resolves to the same entry
    assert lex.lookup("پَاکِستان") == "pɑːkɪstɑːn"


@pytest.mark.parametrize("word,pron,ok", [
    ("لڑکی", "ləɽkiː", True), ("لڑکی", "lərkiː", False), ("گھر", "ɡʰər", True), ("گھر", "ɡər", False),
    ("محبت", "mʊhəbbət", True), ("آنسو", "ɑ̃ːsuː", True), ("پاکستان", "pɑːkɪstɑːn", True), ("پاکستان", "bɑːkɪstɑːn", False),
])
def test_skeleton(word, pron, ok):
    assert skeleton_ok(word, pron) is ok


@pytest.mark.parametrize("word", ["پاکستان", "لڑکی", "کتابیں", "دوست", "ہاں", "سڑک", "ٹھیک", "خواب", "دیوار", "زندگی"])
def test_rules_in_vocab_and_skeleton_valid(word):
    p = rules_phonemize_word(word)
    assert p and not oov_chars(p)
    assert skeleton_ok(word, p)


def test_rules_known_words():
    assert rules_phonemize_word("دوست") == "doːst"
    assert rules_phonemize_word("لڑکی") == "ləɽkiː"
    assert rules_phonemize_word("ہاں") == unicodedata.normalize("NFD", "hɑ̃ː")
    assert rules_phonemize_word("دیوار") == "diːʋɑːr"
    assert rules_phonemize_word("لوگوں") == unicodedata.normalize("NFD", "loːɡõː")


def test_izafat_and_letters(g2p):
    assert g2p.word("وزیرِ") == "ʋəziːreː"
    assert g2p.word("ق") == "qɑːf"
    assert g2p.word("غ") == "ɣɛːn"


def test_neural_or_rules_fallback_is_deterministic(g2p):
    w = "گلدستوں"  # unlikely to be in the lexicon
    a = g2p.word_with_source(w)
    b = g2p.word_with_source(w)
    assert a == b and a[0] and not oov_chars(a[0])
    assert skeleton_ok(w, a[0])

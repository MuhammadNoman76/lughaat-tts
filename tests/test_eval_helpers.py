"""Scoring helpers that need no models: Urdu/English scoring normalisers, CER/WER and the phonetic CER."""
import math

from eval.asr import scoring_normalize_ur, scoring_normalize_en, cer, wer
from eval.pcer import pcer_detail, collapse


def test_scoring_normalizer_unifies_script_and_digits():
    assert scoring_normalize_ur("كتاب يه ۲۵") == scoring_normalize_ur("کتاب یہ پچیس")
    assert cer("اس کی قیمت 2500 روپے ہے۔", "اس کی قیمت دو ہزار پانچ سو روپے ہے") == 0.0
    assert 0.0 < cer("پاکستان ایک خوبصورت ملک ہے", "پاکستان ایک خوبسورت ملک ہے") < 0.1


def test_english_normalizer_and_wer():
    assert scoring_normalize_en("Thank you, Mr. Khan!").startswith("thank you")
    assert wer("good morning how are you", "good morning how are you", lang="en") == 0.0
    assert wer("good morning how are you", "good evening how are you", lang="en") == 0.2


def test_collapse_merges_classes():
    assert collapse("ʈɑːp") == collapse("tɑp")
    assert collapse("ʋɔːʈər") == collapse("wɔtəɹ")
    assert collapse("mˈiTɪŋ") == collapse("miːʈɪŋ")


def test_pcer_is_script_agnostic():
    ref = "آج کی meeting کینسل ہو گئی ہے۔"
    d = pcer_detail(ref, "آج کی میٹنگ کینسل ہو گئی ہے")      # ASR transliterated the English word
    assert d.pcer < 0.05 and not d.english_words_lost
    d2 = pcer_detail(ref, "آج کی meeting cancel ہو گئی ہے")   # ASR kept Latin script
    assert d2.pcer < 0.05
    d3 = pcer_detail("Office میں آج internet کا problem تھا۔", "میں آج کا تھا")
    assert d3.english_words_lost == ["Office", "internet", "problem"]
    assert d3.pcer_en > 0.8 and not math.isnan(d3.pcer_ur)

import unicodedata

import pytest

from eval.testsets import APPENDIX_A2, APPENDIX_A2_TAGS, PAKISTANI_ANCHORS
from lughaat_tts.codeswitch import MixedFrontend, tag_tokens
from lughaat_tts.english import to_pakistani, to_native, classify_latin, spell_letters
from lughaat_tts.vocab import oov_chars


@pytest.fixture(scope="module")
def fe():
    return MixedFrontend()


def test_tags_without_english_model():
    toks = tag_tokens("آج کی meeting کینسل ہو گئی ہے۔")
    assert [t.tag for t in toks] == ["ur", "ur", "en", "ur", "ur", "ur", "ur", "punct"]


@pytest.mark.parametrize("sentence,tags", list(APPENDIX_A2_TAGS.items()))
def test_appendix_a2_tags(fe, sentence, tags):
    toks = [t for t in tag_tokens(sentence, fe.english) if t.tag != "punct"]
    assert [t.tag for t in toks] == tags


def test_classify_latin():
    assert classify_latin("USB") == "en_acronym"
    assert classify_latin("NADRA") == "en_acronym_word"
    assert classify_latin("PTI") == "en_acronym"
    assert classify_latin("laptop") == "en"
    assert spell_letters("USB") == "ju ɛs bi"


def test_pakistani_mapping_table():
    assert to_pakistani("lˈæptˌɑp") == "lɛːpʈɑːp"
    assert to_pakistani("θˈɪn") == "tʰɪn"
    assert to_pakistani("ðˈɪs") == "dɪs"
    assert to_pakistani("wˈɔTəɹ") == "ʋɔːʈər"
    assert to_pakistani("lˈæptˌɑp", retroflex_td=False) == "lɛːptɑːp"
    assert to_native("ˈAʒə") == "ˈAʒə"


@pytest.mark.parametrize("word,expected", list(PAKISTANI_ANCHORS.items()))
def test_pakistani_anchors(fe, word, expected):
    assert fe.english.phonemize_span(word, accent="pakistani") == unicodedata.normalize("NFD", expected)


@pytest.mark.parametrize("sentence", APPENDIX_A2)
def test_a2_in_vocab_all_accents(fe, sentence):
    for acc in ("pakistani", "native", "auto"):
        ph = fe(sentence, english_accent=acc)
        assert ph and not oov_chars(ph), (acc, ph)
        assert len(ph) <= 510


def test_auto_accent_choice(fe):
    assert fe.phonemize("آج کی meeting کینسل ہو گئی ہے۔").accent == "pakistani"
    assert fe.phonemize("Good morning, how are you today?").accent == "native"


def test_punctuation_and_year_marker(fe):
    assert fe("پاکستان ایک خوبصورت ملک ہے۔").endswith("hɛː.")
    assert fe("کیا آپ میری بات سمجھ رہے ہیں؟").endswith("?")
    ph = fe("۱۹۴۷ء میں پاکستان آزاد ہوا۔")
    assert "həmzɑː" not in ph and ph.startswith("ʊnniːs sɔː")


def test_deterministic(fe):
    s = APPENDIX_A2[4]
    assert fe(s) == fe(s)

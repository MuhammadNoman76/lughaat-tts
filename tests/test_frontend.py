"""Plan 4.6: every Appendix sentence yields only in-vocabulary phones; output is deterministic;
chunking respects the limits."""
import pytest

from eval.testsets import APPENDIX_A, APPENDIX_A2, APPENDIX_A3
from lughaat_tts.chunker import chunk_text, HARD_LIMIT
from lughaat_tts.codeswitch import MixedFrontend
from lughaat_tts.vocab import oov_chars
from lughaat_tts import FRONTEND_VERSION, phonemize


@pytest.fixture(scope="module")
def fe():
    return MixedFrontend()


@pytest.mark.parametrize("sentence", APPENDIX_A + APPENDIX_A2 + APPENDIX_A3)
def test_in_vocab(fe, sentence):
    ph = fe(sentence)
    assert ph.strip(), sentence
    assert not oov_chars(ph), (sentence, ph, oov_chars(ph))
    assert len(ph) <= HARD_LIMIT


@pytest.mark.parametrize("sentence", APPENDIX_A[:6] + APPENDIX_A2[:3] + APPENDIX_A3[:2])
def test_deterministic(fe, sentence):
    assert fe(sentence) == fe(sentence)


def test_chunking(fe):
    long_text = " ".join(APPENDIX_A)
    chunks = chunk_text(long_text, lambda s: fe(s), max_tokens=400)
    assert len(chunks) >= 3
    for c in chunks:
        assert 0 < len(c.phonemes) <= 400
    short = chunk_text("ہاں۔ نہیں۔ شکریہ۔", lambda s: fe(s))
    assert len(short) == 1   # very short sentences are merged


def test_english_only_sentence_uses_native_phones(fe):
    r = fe.phonemize("The quick brown fox jumps over the lazy dog.")
    assert r.accent == "native"
    assert "ð" in r.phonemes and "ˈ" in r.phonemes


def test_module_level_phonemize():
    assert phonemize("پاکستان").startswith("pɑːkɪstɑːn")
    assert FRONTEND_VERSION.startswith("ur-frontend-")

"""Urdu word-level G2P with the runtime order of plan 4.5:

    lexicon lookup -> (if missing) neural G2P -> skeleton check -> (if it fails) rules

Every result passes through ``phoneset.to_lughaat_tts`` and must be in vocabulary.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

from .lexicon import Lexicon, default_lexicon
from .normalize import normalize_chars, strip_diacritics
from .phoneset import to_lughaat_tts
from .rules import rules_phonemize_word
from .skeleton import skeleton_ok
from .g2p_model import DEFAULT_MODEL_PATH

_IZAFAT_ZER = "ِ"

# names of single letters (e.g. "ق، خ اور غ کی آواز")
LETTER_NAMES = {
    "ا": "əlɪf", "آ": "əlɪf mədd", "ب": "beː", "پ": "peː", "ت": "teː", "ٹ": "ʈeː", "ث": "seː", "ج": "ʤiːm",
    "چ": "ʧeː", "ح": "heː", "خ": "xeː", "د": "dɑːl", "ڈ": "ɖɑːl", "ذ": "zɑːl", "ر": "reː", "ڑ": "ɽeː",
    "ز": "zeː", "ژ": "ʒeː", "س": "siːn", "ش": "ʃiːn", "ص": "sʋɑːd", "ض": "zʋɑːd", "ط": "toːeː", "ظ": "zoːeː",
    "ع": "ɛːn", "غ": "ɣɛːn", "ف": "feː", "ق": "qɑːf", "ک": "kɑːf", "گ": "ɡɑːf", "ل": "lɑːm", "م": "miːm",
    "ن": "nuːn", "ں": "nuːn ɣʊnnɑː", "و": "ʋɑːoː", "ہ": "heː", "ھ": "heː", "ء": "həmzɑː", "ی": "jeː", "ے": "jeː",
}


class UrduG2P:
    def __init__(self, lexicon: Optional[Lexicon] = None, neural_path: Optional[str] = DEFAULT_MODEL_PATH,
                 use_neural: bool = True, device: str = "cpu"):
        self.lexicon = lexicon if lexicon is not None else default_lexicon()
        self.neural_path = neural_path
        self.use_neural = use_neural
        self.device = device
        self._neural = None
        self._neural_failed = False
        self.stats = {"lexicon": 0, "neural": 0, "rules": 0, "neural_rejected": 0}

    # -- neural model -----------------------------------------------------------
    def _get_neural(self):
        if self._neural is not None or self._neural_failed or not self.use_neural:
            return self._neural
        path = self.neural_path
        # `pip install git+https://huggingface.co/...` leaves LFS files as small pointer files;
        # anything under 1 MB is not a real checkpoint -> fetch it from the model repo instead.
        if not path or not os.path.exists(path) or os.path.getsize(path) < 1_000_000:
            path = self._download_neural() or path
        if not path or not os.path.exists(path) or os.path.getsize(path) < 1_000_000:
            self._neural_failed = True
            return None
        try:
            from .g2p_model import NeuralG2P
            self._neural = NeuralG2P.load(path, device=self.device)
        except Exception:
            self._neural_failed = True
        return self._neural

    @staticmethod
    def _download_neural() -> Optional[str]:
        from . import _repo
        repo = os.environ.get("LUGHAAT_TTS_REPO") or _repo.DEFAULT_REPO_ID
        if not repo:
            return None
        try:
            from huggingface_hub import hf_hub_download
            return hf_hub_download(repo, _repo.G2P_FILENAME, token=os.environ.get("HF_TOKEN") or None)
        except Exception:
            return None

    @property
    def has_neural(self) -> bool:
        return self._get_neural() is not None

    # -- word level ---------------------------------------------------------------
    def word_with_source(self, word: str) -> tuple[str, str]:
        """Return (phonemes, source) for one Urdu word (no spaces)."""
        w = normalize_chars(word).strip()
        if not w:
            return "", "none"
        if len(strip_diacritics(w)) == 1 and strip_diacritics(w) in LETTER_NAMES:
            return LETTER_NAMES[strip_diacritics(w)], "letter"
        izafat = w.endswith(_IZAFAT_ZER) or w.endswith("ۂ")
        base = w[:-1] if w.endswith(_IZAFAT_ZER) else w
        pron = self.lexicon.lookup(w)
        if pron is None and izafat and w.endswith(_IZAFAT_ZER):
            p = self.lexicon.lookup(base)
            if p is not None:
                pron = p + "eː" if not p.endswith("eː") else p
        if pron is not None:
            self.stats["lexicon"] += 1
            return pron, "lexicon"
        neural = self._get_neural()
        rules = rules_phonemize_word(w)
        # plan 4.5 order: neural -> skeleton check -> rules, implemented as a rules-anchored rerank of
        # the neural n-best (DECISIONS.md): skeleton-valid candidates compete with the rule output.
        if neural is not None:
            try:
                from .g2p_model import rerank_select
                cand, src = rerank_select(w, neural.predict_nbest(w, beam=5, n=5), rules)
                if src == "neural":
                    self.stats["neural"] += 1
                    return cand, "neural"
                self.stats["neural_rejected"] += 1
            except Exception:
                pass
        self.stats["rules"] += 1
        return rules, "rules"

    @lru_cache(maxsize=65536)
    def word(self, word: str) -> str:
        return self.word_with_source(word)[0]

    def __call__(self, text: str) -> str:
        """Phonemize an Urdu-only span (already normalised; numbers expanded)."""
        out = []
        for tok in text.split():
            p = self.word(tok)
            if p:
                out.append(p)
        return to_lughaat_tts(" ".join(out))


@lru_cache(maxsize=1)
def default_urdu_g2p() -> UrduG2P:
    return UrduG2P()


__all__ = ["UrduG2P", "default_urdu_g2p"]

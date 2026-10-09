"""ASR round-trip scoring (plan 3.2.6, 9.2).

* :class:`WhisperASR` wraps a transformers Whisper pipeline (large-v3 for evaluation,
  large-v3-turbo for data filtering on Kaggle).
* :func:`scoring_normalize_ur` is the scoring normaliser shared by filtering and evaluation:
  strip diacritics and punctuation, unify ی/ي, ک/ك, ہ/ه, ے->ی, convert digits to words.
* :func:`cer` / :func:`wer` are plain edit-distance rates.
"""
from __future__ import annotations

import os
import re
import sys
import unicodedata
from typing import Iterable, Optional, Sequence, Union

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from lughaat_tts.normalize import HARAKAT, normalize_chars, expand_numbers  # noqa: E402

_PUNCT_RE = re.compile(r"[\s،؛؟۔\.,;:!?\"'`()\[\]{}\-—–…٭٪%«»]+")
_UNIFY = {"ي": "ی", "ى": "ی", "ك": "ک", "ه": "ہ", "ے": "ی", "ۂ": "ہ", "ۃ": "ہ", "ة": "ہ", "أ": "ا", "إ": "ا", "آ": "ا", "ئ": "ی", "ؤ": "و", "ء": ""}


def scoring_normalize_ur(text: str) -> str:
    s = normalize_chars(text or "")
    s = expand_numbers(s)
    s = unicodedata.normalize("NFC", s)
    s = "".join(ch for ch in s if ch not in HARAKAT)
    s = "".join(_UNIFY.get(ch, ch) for ch in s)
    s = _PUNCT_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


_EN_PUNCT = re.compile(r"[^a-z0-9' ]+")


def scoring_normalize_en(text: str) -> str:
    """Whisper-style basic English normaliser (lowercase, strip punctuation, spell small numbers)."""
    try:
        from transformers.models.whisper.english_normalizer import EnglishTextNormalizer  # type: ignore
        norm = EnglishTextNormalizer({})
        return norm(text or "").strip()
    except Exception:
        s = (text or "").lower()
        s = _EN_PUNCT.sub(" ", s)
        return re.sub(r"\s+", " ", s).strip()


def edit_distance(a: Sequence, b: Sequence) -> int:
    if a == b:
        return 0
    if not a or not b:
        return max(len(a), len(b))
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(ref: str, hyp: str, lang: str = "ur") -> float:
    r = scoring_normalize_ur(ref) if lang == "ur" else scoring_normalize_en(ref)
    h = scoring_normalize_ur(hyp) if lang == "ur" else scoring_normalize_en(hyp)
    r, h = r.replace(" ", ""), h.replace(" ", "")
    return edit_distance(r, h) / len(r) if r else float("nan")


def wer(ref: str, hyp: str, lang: str = "en") -> float:
    r = (scoring_normalize_ur(ref) if lang == "ur" else scoring_normalize_en(ref)).split()
    h = (scoring_normalize_ur(hyp) if lang == "ur" else scoring_normalize_en(hyp)).split()
    return edit_distance(r, h) / len(r) if r else float("nan")


def _load_audio16(x: Union[str, np.ndarray], sr: Optional[int]) -> np.ndarray:
    import librosa
    if isinstance(x, str):
        y, s = librosa.load(x, sr=None, mono=True)
    else:
        y, s = np.asarray(x, dtype=np.float32), sr or 24000
    if s != 16000:
        y = librosa.resample(y, orig_sr=s, target_sr=16000)
    return y.astype(np.float32)


class WhisperASR:
    def __init__(self, model_id: str = "openai/whisper-large-v3", device: Optional[str] = None, batch_size: int = 8, num_beams: int = 1):
        import torch
        from transformers import pipeline  # type: ignore
        self.model_id = model_id
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.pipe = pipeline("automatic-speech-recognition", model=model_id, torch_dtype=dtype, device=device)
        self.batch_size = batch_size
        self.num_beams = num_beams

    def transcribe(self, audios: Iterable[Union[str, np.ndarray]], language: str = "ur", sr: Optional[int] = None) -> list[str]:
        items = [{"raw": _load_audio16(a, sr), "sampling_rate": 16000} for a in audios]
        if not items:
            return []
        gen = {"language": language, "task": "transcribe", "num_beams": self.num_beams}
        outs = self.pipe(items, batch_size=self.batch_size, generate_kwargs=gen, chunk_length_s=30, return_timestamps=False)
        if isinstance(outs, dict):
            outs = [outs]
        return [o["text"].strip() for o in outs]

    def transcribe_one(self, audio: Union[str, np.ndarray], language: str = "ur", sr: Optional[int] = None) -> str:
        return self.transcribe([audio], language, sr)[0]


__all__ = ["WhisperASR", "scoring_normalize_ur", "scoring_normalize_en", "cer", "wer", "edit_distance"]

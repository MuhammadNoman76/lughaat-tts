"""lughaat_tts: Urdu (+ English, + code-switching) frontend and pipeline for Kokoro-82M-Urdu.

    from lughaat_tts import UrduPipeline, phonemize
    UrduPipeline().save("پاکستان ایک خوبصورت ملک ہے۔", "out.wav")
    print(phonemize("آج کی meeting کینسل ہو گئی ہے۔"))

Only the frontend is imported eagerly; ``UrduPipeline`` (which needs torch + kokoro) is
loaded on first access so the frontend stays usable in light environments.
"""
from __future__ import annotations

from .version import __version__, FRONTEND_VERSION
from .codeswitch import MixedFrontend, FrontendResult, RomanUrduHook, phonemize, default_frontend
from .g2p import UrduG2P
from .normalize import normalize
from .phoneset import to_lughaat_tts, assert_in_vocab
from .vocab import VOCAB

__all__ = [
    "UrduPipeline", "MixedFrontend", "FrontendResult", "RomanUrduHook", "UrduG2P", "phonemize",
    "default_frontend", "normalize", "to_lughaat_tts", "assert_in_vocab", "VOCAB", "__version__", "FRONTEND_VERSION",
]


def __getattr__(name: str):
    if name == "UrduPipeline":
        from .pipeline import UrduPipeline
        return UrduPipeline
    raise AttributeError(name)

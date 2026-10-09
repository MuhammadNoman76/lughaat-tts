"""UrduPipeline: text -> 24 kHz audio with the fine-tuned Kokoro-Urdu model (plan 11.1).

    from lughaat_tts import UrduPipeline
    tts = UrduPipeline(voice="uf_rasa", device="cpu")      # downloads weights from HF on first use
    audio = tts("آج موسم بہت اچھا ہے۔", speed=1.0)          # np.float32 @ 24 kHz
    tts.save("آج کی meeting کینسل ہو گئی ہے۔", "mixed.wav")
    tts.save("Good morning, this is an English sentence.", "en.wav", english_accent="native")
    print(tts.phonemize("آج کی meeting کینسل ہو گئی ہے۔"))

Internally: normaliser -> sentence chunker -> token language tagging -> Urdu G2P /
misaki English G2P + accent mapping -> vocabulary mapping -> KModel(repo_id, config,
model=path) -> voice[len(tokens)-1] -> audio, chunks joined with 120 ms of silence.
"""
from __future__ import annotations

import json
import os
from typing import Optional, Union

import numpy as np

from . import _repo
from .chunker import chunk_text, MAX_TOKENS
from .codeswitch import MixedFrontend, FrontendResult
from .version import FRONTEND_VERSION

SAMPLE_RATE = 24000
PAUSE_SECONDS = 0.12


def _hf_download(repo_id: str, filename: str, token: Optional[str], revision: Optional[str]) -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=repo_id, filename=filename, token=token, revision=revision)


class UrduPipeline:
    def __init__(
        self,
        voice: str = "uf_rasa",
        device: Optional[str] = None,
        repo_id: Optional[str] = None,
        model_path: Optional[str] = None,
        config_path: Optional[str] = None,
        voices_dir: Optional[str] = None,
        g2p_model_path: Optional[str] = None,
        token: Optional[str] = None,
        revision: Optional[str] = None,
        english_accent: str = "auto",
        retroflex_td: bool = True,
        max_tokens: int = MAX_TOKENS,
        british_base: bool = False,
    ):
        import torch
        self.repo_id = repo_id or os.environ.get("LUGHAAT_TTS_REPO") or _repo.DEFAULT_REPO_ID
        self.token = token or os.environ.get("HF_TOKEN") or None
        self.revision = revision
        self.default_voice = voice
        self.english_accent = english_accent
        self.retroflex_td = retroflex_td
        self.max_tokens = max_tokens
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # -- files ---------------------------------------------------------------
        if model_path is None or config_path is None:
            if not self.repo_id:
                raise ValueError("No repo_id: pass repo_id='<user>/lughaat-tts-82m', set LUGHAAT_TTS_REPO, "
                                 "or pass model_path= and config_path= explicitly.")
        self.config_path = config_path or _hf_download(self.repo_id, _repo.CONFIG_FILENAME, self.token, revision)
        self.model_path = model_path or _hf_download(self.repo_id, _repo.MODEL_FILENAME, self.token, revision)
        self.voices_dir = voices_dir
        with open(self.config_path, encoding="utf-8") as f:
            self.config = json.load(f)
        trained_with = self.config.get("urdu_frontend_version")
        if trained_with and trained_with != FRONTEND_VERSION:
            import warnings
            warnings.warn(f"model was trained with frontend {trained_with}, this package is {FRONTEND_VERSION}")

        # -- frontend -------------------------------------------------------------
        from .g2p import UrduG2P
        from .g2p_model import DEFAULT_MODEL_PATH
        g2p_path = g2p_model_path or DEFAULT_MODEL_PATH
        if not os.path.exists(g2p_path) and self.repo_id:
            try:
                g2p_path = _hf_download(self.repo_id, _repo.G2P_FILENAME, self.token, revision)
            except Exception:
                g2p_path = None
        self.frontend = MixedFrontend(urdu=UrduG2P(neural_path=g2p_path), british_base=british_base)

        # -- model ------------------------------------------------------------------
        from kokoro import KModel
        self.model = KModel(repo_id=self.repo_id or "hexgrad/Kokoro-82M", config=self.config_path, model=self.model_path)
        self.model = self.model.to(device).eval()
        self._voices: dict[str, "torch.Tensor"] = {}

    # ---------------------------------------------------------------------------
    def load_voice(self, voice: Union[str, "np.ndarray"]):
        import torch
        if not isinstance(voice, str):
            return torch.as_tensor(voice).float()
        if voice in self._voices:
            return self._voices[voice]
        if voice.endswith(".pt") and os.path.exists(voice):
            path = voice
        elif self.voices_dir and os.path.exists(os.path.join(self.voices_dir, f"{voice}.pt")):
            path = os.path.join(self.voices_dir, f"{voice}.pt")
        else:
            path = _hf_download(self.repo_id, f"voices/{voice}.pt", self.token, self.revision)
        pack = torch.load(path, map_location="cpu", weights_only=True).float()
        assert tuple(pack.shape) == (510, 1, 256), f"voicepack {voice} has shape {tuple(pack.shape)}, expected (510, 1, 256)"
        self._voices[voice] = pack
        return pack

    def phonemize(self, text: str, english_accent: Optional[str] = None, retroflex_td: Optional[bool] = None) -> FrontendResult:
        """Return phonemes plus per-token language tags (for debugging)."""
        return self.frontend.phonemize(
            text,
            english_accent=english_accent or self.english_accent,
            retroflex_td=self.retroflex_td if retroflex_td is None else retroflex_td,
        )

    def _ph(self, text: str, english_accent: str, retroflex_td: bool) -> str:
        return self.frontend(text, english_accent=english_accent, retroflex_td=retroflex_td)

    def synthesize_phonemes(self, phonemes: str, voice: Optional[str] = None, speed: float = 1.0) -> np.ndarray:
        import torch
        pack = self.load_voice(voice or self.default_voice).to(self.device)
        ps = phonemes.strip()
        if not ps:
            return np.zeros(0, dtype=np.float32)
        if len(ps) > 510:
            ps = ps[:510]
        ref_s = pack[len(ps) - 1]
        with torch.no_grad():
            audio = self.model(ps, ref_s, speed)
        return audio.detach().cpu().numpy().astype(np.float32)

    def __call__(self, text: str, speed: float = 1.0, voice: Optional[str] = None,
                 english_accent: Optional[str] = None, retroflex_td: Optional[bool] = None,
                 pause: float = PAUSE_SECONDS) -> np.ndarray:
        accent = english_accent or self.english_accent
        rtd = self.retroflex_td if retroflex_td is None else retroflex_td
        chunks = chunk_text(text, lambda s: self._ph(s, accent, rtd), max_tokens=self.max_tokens)
        pieces: list[np.ndarray] = []
        gap = np.zeros(int(SAMPLE_RATE * pause), dtype=np.float32)
        for i, ch in enumerate(chunks):
            audio = self.synthesize_phonemes(ch.phonemes, voice=voice, speed=speed)
            if audio.size:
                if pieces:
                    pieces.append(gap)
                pieces.append(audio)
        if not pieces:
            return np.zeros(0, dtype=np.float32)
        out = np.concatenate(pieces)
        peak = float(np.max(np.abs(out))) if out.size else 0.0
        if peak > 0.99:
            out = out / peak * 0.99
        return out

    def save(self, text: str, path: str, **kwargs) -> str:
        import soundfile as sf
        audio = self(text, **kwargs)
        sf.write(path, audio, SAMPLE_RATE)
        return path

    @property
    def sample_rate(self) -> int:
        return SAMPLE_RATE


__all__ = ["UrduPipeline", "SAMPLE_RATE"]

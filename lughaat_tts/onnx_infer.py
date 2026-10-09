"""ONNX inference with onnxruntime + numpy only (plan 11.1).

    python -m lughaat_tts.onnx_infer --onnx onnx/lughaat-tts-82m.onnx --voice voices/uf_rasa.pt \
        --text "پاکستان ایک خوبصورت ملک ہے۔" --out out.wav

The text frontend is pure Python (lexicon + rules). The neural G2P fallback needs torch;
when torch is missing, out-of-lexicon words fall back to the rules (slightly lower accuracy).
Voicepacks are ``.pt`` files; without torch they are read with ``safetensors``/numpy if a
``.npy`` copy exists next to them (``scripts/09_onnx.py`` writes ``voices/<name>.npy``).
"""
from __future__ import annotations

import argparse
import os
from typing import Optional

import numpy as np

from .vocab import VOCAB
from .chunker import chunk_text

SAMPLE_RATE = 24000


def load_voice_np(path: str) -> np.ndarray:
    if path.endswith(".npy"):
        return np.load(path).astype(np.float32)
    npy = os.path.splitext(path)[0] + ".npy"
    if os.path.exists(npy):
        return np.load(npy).astype(np.float32)
    import torch  # type: ignore
    return torch.load(path, map_location="cpu", weights_only=True).float().numpy()


class OnnxUrduTTS:
    def __init__(self, onnx_path: str, voice_path: str, g2p_model_path: Optional[str] = None, threads: int = 4):
        import onnxruntime as ort  # type: ignore
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
        self.voice = load_voice_np(voice_path)
        assert self.voice.shape == (510, 1, 256), self.voice.shape
        from .codeswitch import MixedFrontend
        from .g2p import UrduG2P
        try:
            import torch  # noqa: F401
            use_neural = True
        except Exception:
            use_neural = False
        self.frontend = MixedFrontend(urdu=UrduG2P(neural_path=g2p_model_path, use_neural=use_neural) if g2p_model_path else UrduG2P(use_neural=use_neural))

    def synthesize_phonemes(self, ps: str, speed: float = 1.0) -> np.ndarray:
        ids = [VOCAB[c] for c in ps if c in VOCAB][:510]
        if not ids:
            return np.zeros(0, dtype=np.float32)
        input_ids = np.array([[0, *ids, 0]], dtype=np.int64)
        style = self.voice[len(ids) - 1]                 # [1, 256]
        out = self.sess.run(None, {"input_ids": input_ids, "style": style.astype(np.float32), "speed": np.array([speed], dtype=np.float32)})
        return out[0].astype(np.float32).reshape(-1)

    def __call__(self, text: str, speed: float = 1.0, english_accent: str = "auto", pause: float = 0.12) -> np.ndarray:
        chunks = chunk_text(text, lambda s: self.frontend(s, english_accent=english_accent))
        pieces: list[np.ndarray] = []
        gap = np.zeros(int(SAMPLE_RATE * pause), dtype=np.float32)
        for ch in chunks:
            a = self.synthesize_phonemes(ch.phonemes, speed)
            if a.size:
                if pieces:
                    pieces.append(gap)
                pieces.append(a)
        return np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Lughaat-TTS ONNX inference")
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--voice", required=True, help="voices/uf_rasa.pt or .npy")
    ap.add_argument("--text", required=True)
    ap.add_argument("--out", default="out.wav")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--accent", default="auto", choices=["auto", "pakistani", "native"])
    ap.add_argument("--g2p-model", default=None)
    a = ap.parse_args(argv)
    tts = OnnxUrduTTS(a.onnx, a.voice, a.g2p_model)
    audio = tts(a.text, speed=a.speed, english_accent=a.accent)
    import soundfile as sf  # type: ignore
    sf.write(a.out, audio, SAMPLE_RATE)
    print(f"wrote {a.out} ({len(audio)/SAMPLE_RATE:.2f}s)")


if __name__ == "__main__":
    main()

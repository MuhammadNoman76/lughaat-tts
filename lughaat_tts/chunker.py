"""Sentence chunking (plan 4.2.5).

Kokoro rushes long inputs and is weak on very short ones, so text is split into
sentences, sentences shorter than ``min_tokens`` phoneme tokens are merged with their
neighbours, and chunks are capped at ``max_tokens`` phoneme characters (default 400,
hard limit 510 from PL-BERT's 512 positions).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .normalize import split_sentences

MAX_TOKENS = 400
HARD_LIMIT = 510
MIN_TOKENS = 10


@dataclass
class Chunk:
    text: str
    phonemes: str


def _split_long(ph: str, limit: int) -> list[str]:
    """Split an over-long phoneme string at punctuation, then at spaces."""
    if len(ph) <= limit:
        return [ph]
    out: list[str] = []
    cur = ""
    for piece in _pieces(ph):
        if len(cur) + len(piece) + (1 if cur else 0) > limit:
            if cur:
                out.append(cur)
            cur = piece if len(piece) <= limit else piece[:limit]
            if len(piece) > limit:
                rest = piece[limit:]
                while rest:
                    out.append(cur)
                    cur, rest = rest[:limit], rest[limit:]
        else:
            cur = (cur + " " + piece) if cur else piece
    if cur:
        out.append(cur)
    return out


def _pieces(ph: str) -> list[str]:
    pieces: list[str] = []
    buf: list[str] = []
    for w in ph.split(" "):
        buf.append(w)
        if w and w[-1] in ",.!?;:…":
            pieces.append(" ".join(buf))
            buf = []
    if buf:
        pieces.append(" ".join(buf))
    return pieces


def chunk_text(text: str, phonemize: Callable[[str], str], max_tokens: int = MAX_TOKENS,
               min_tokens: int = MIN_TOKENS) -> list[Chunk]:
    sentences = split_sentences(text) or [text]
    phs = [(s, phonemize(s)) for s in sentences]
    phs = [(s, p) for s, p in phs if p.strip()]
    # merge very short sentences with the following one (or the previous at the end)
    merged: list[tuple[str, str]] = []
    for s, p in phs:
        if merged and (len(merged[-1][1]) < min_tokens or len(p) < min_tokens) and len(merged[-1][1]) + len(p) + 1 <= max_tokens:
            ps, pp = merged[-1]
            merged[-1] = (ps + " " + s, pp + " " + p)
        else:
            merged.append((s, p))
    chunks: list[Chunk] = []
    for s, p in merged:
        for part in _split_long(p, min(max_tokens, HARD_LIMIT)):
            chunks.append(Chunk(text=s, phonemes=part))
    return chunks


__all__ = ["chunk_text", "Chunk", "MAX_TOKENS", "HARD_LIMIT", "MIN_TOKENS"]

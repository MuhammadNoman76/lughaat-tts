"""Pronunciation lexicon (plan 4.4).

File format (``lughaat_tts/data/lexicon.tsv``, UTF-8, tab separated, ``#`` comments)::

    word<TAB>pronunciation<TAB>source<TAB>variant1;variant2

``source`` is one of ``gold`` (hand-checked, Appendix B / function words), ``wikipron``
(Wiktionary via WikiPron, CC-BY-SA), ``agree`` (espeak + LLM agreed), ``llm``, ``espeak``,
``rules``. Lookups try the exact word, the word without diacritics, and common spelling
variants.

The lexicon file derived from WikiPron is licensed CC-BY-SA-4.0 (see data/LICENSE.lexicon).
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Iterable, Optional

from .normalize import normalize_chars, strip_diacritics
from .phoneset import to_lughaat_tts, is_canonical, LONG_VOWELS
from .skeleton import skeleton_ok

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
LEXICON_PATH = os.path.join(DATA_DIR, "lexicon.tsv")
GOLD_PATH = os.path.join(DATA_DIR, "gold_overrides.tsv")
WIKIPRON_TEST_PATH = os.path.join(DATA_DIR, "wikipron_test.tsv")

SOURCE_RANK = {"gold": 0, "wikipron": 1, "agree": 2, "llm": 3, "espeak": 4, "rules": 5, "neural": 6}


@dataclass
class LexEntry:
    word: str
    pron: str
    source: str = "wikipron"
    variants: list[str] = field(default_factory=list)


class Lexicon:
    def __init__(self, entries: Optional[dict[str, LexEntry]] = None):
        self.entries: dict[str, LexEntry] = entries or {}
        self._bare: dict[str, str] = {}
        for w in self.entries:
            self._bare.setdefault(strip_diacritics(w), w)

    # -- construction ---------------------------------------------------------
    @classmethod
    def load(cls, path: str = LEXICON_PATH, extra: Iterable[str] = ()) -> "Lexicon":
        lex = cls()
        for p in [path, *extra]:
            if p and os.path.exists(p):
                lex.update_from_file(p)
        return lex

    def update_from_file(self, path: str) -> int:
        n = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) < 2:
                    continue
                word = normalize_chars(parts[0].strip())
                pron = to_lughaat_tts(parts[1].strip())
                source = parts[2].strip() if len(parts) > 2 and parts[2].strip() else "wikipron"
                variants = [to_lughaat_tts(v) for v in parts[3].split(";")] if len(parts) > 3 and parts[3].strip() else []
                self.add(word, pron, source, variants)
                n += 1
        return n

    def add(self, word: str, pron: str, source: str = "wikipron", variants: Optional[list[str]] = None, override: bool = False) -> None:
        cur = self.entries.get(word)
        if cur is None or override or SOURCE_RANK.get(source, 9) < SOURCE_RANK.get(cur.source, 9):
            self.entries[word] = LexEntry(word, pron, source, list(variants or []))
            self._bare.setdefault(strip_diacritics(word), word)
        elif cur is not None and pron not in cur.variants and pron != cur.pron:
            cur.variants.append(pron)

    def save(self, path: str, header: str = "") -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            if header:
                for line in header.rstrip("\n").split("\n"):
                    f.write("# " + line + "\n")
            for w in sorted(self.entries):
                e = self.entries[w]
                f.write(f"{w}\t{e.pron}\t{e.source}\t{';'.join(e.variants)}\n")

    # -- lookup -----------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.entries)

    def __contains__(self, word: str) -> bool:
        return self.get(word) is not None

    def get(self, word: str) -> Optional[LexEntry]:
        w = normalize_chars(word)
        e = self.entries.get(w)
        if e is not None:
            return e
        bare = strip_diacritics(w)
        if bare in self.entries:
            return self.entries[bare]
        key = self._bare.get(bare)
        if key is not None:
            return self.entries[key]
        # spelling variants: final ہ/ۂ, ی/ے, ئ/ی
        for a, b in (("ۂ", "ہ"), ("ئ", "ی"), ("ے", "ی"), ("ہ", "ا")):
            if a in bare:
                alt = bare.replace(a, b)
                if alt in self.entries:
                    return self.entries[alt]
        return None

    def lookup(self, word: str) -> Optional[str]:
        e = self.get(word)
        return e.pron if e else None


# ---------------------------------------------------------------------------
# WikiPron seed handling
# ---------------------------------------------------------------------------
def _variant_score(word: str, pron: str) -> tuple:
    """Prefer skeleton-valid, canonical, more detailed (length-marked) transcriptions."""
    ok = skeleton_ok(word, pron)
    canonical = is_canonical(pron)
    n_long = sum(pron.count(v) for v in LONG_VOWELS)
    return (ok, canonical, n_long, len(pron))


def choose_variant(word: str, prons: list[str]) -> tuple[str, list[str]]:
    mapped = []
    for p in prons:
        m = to_lughaat_tts(p.replace(" ", ""))
        if m and m not in mapped:
            mapped.append(m)
    if not mapped:
        return "", []
    mapped.sort(key=lambda p: _variant_score(word, p), reverse=True)
    return mapped[0], mapped[1:]


def read_wikipron(path: str) -> dict[str, list[str]]:
    words: dict[str, list[str]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            w, p = line.split("\t")[:2]
            w = normalize_chars(w)
            if not w or len(w) < 2 and w not in ("ہے", "نے"):
                if len(w) < 2:
                    continue
            words.setdefault(w, []).append(p)
    return words


def build_seed_lexicon(
    broad_tsv: str,
    narrow_tsv: Optional[str],
    gold_tsv: Optional[str],
    out_path: str,
    test_out_path: str,
    holdout: float = 0.10,
    seed: int = 42,
) -> tuple["Lexicon", list[tuple[str, str]]]:
    """Map WikiPron into the canonical set, hold out 10 % for G2P evaluation, add gold."""
    wp = read_wikipron(broad_tsv)
    if narrow_tsv and os.path.exists(narrow_tsv):
        for w, ps in read_wikipron(narrow_tsv).items():
            wp.setdefault(w, []).extend(ps)
    words = sorted(wp)
    rng = random.Random(seed)
    rng.shuffle(words)
    n_test = int(len(words) * holdout)
    test_words = set(words[:n_test])
    lex = Lexicon()
    test_rows: list[tuple[str, str]] = []
    for w in sorted(wp):
        pron, variants = choose_variant(w, wp[w])
        if not pron:
            continue
        if w in test_words:
            test_rows.append((w, pron))
        else:
            lex.add(w, pron, "wikipron", variants)
    if gold_tsv and os.path.exists(gold_tsv):
        gold = Lexicon()
        gold.update_from_file(gold_tsv)
        for w, e in gold.entries.items():
            lex.add(w, e.pron, "gold", e.variants, override=True)
    lex.save(out_path, header=(
        "lughaat-tts pronunciation lexicon. word<TAB>pron<TAB>source<TAB>variants\n"
        "Derived in part from WikiPron (CUNY-CL/wikipron, data scraped from Wiktionary).\n"
        "Licence of this file: CC-BY-SA-4.0. Model weights are unaffected."))
    with open(test_out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("# held-out WikiPron words (never used for training the neural G2P)\n")
        for w, p in test_rows:
            f.write(f"{w}\t{p}\n")
    return lex, test_rows


@lru_cache(maxsize=1)
def default_lexicon() -> Lexicon:
    return Lexicon.load(LEXICON_PATH, extra=[GOLD_PATH])


__all__ = ["Lexicon", "LexEntry", "default_lexicon", "build_seed_lexicon", "choose_variant", "read_wikipron",
           "LEXICON_PATH", "GOLD_PATH", "WIKIPRON_TEST_PATH", "DATA_DIR"]

#!/usr/bin/env python3
"""Build / expand the pronunciation lexicon (plan 4.4).

    python scripts/02_build_lexicon.py --words data/rasa_words.txt --wiki-top 50000 \
        --out-dir lughaat_tts/data [--llm-max-words 60000]

1. Seed: WikiPron broad+narrow mapped into the canonical set; 10 % held out to
   lughaat_tts/data/wikipron_test.tsv (done once; reused if present so the test set is stable).
2. Word list: every word type in the Rasa transcripts (+ the top-N Urdu Wikipedia words).
3. Candidate A: espeak-ng `ur` + fixes (rules.espeak_candidate); falls back to the pure-Python
   rules when espeak-ng is missing. Candidate B (optional): an OpenAI-Responses-compatible LLM
   (the user's Azure deployment), batches of 100 words, cached in reports/llm_cache.jsonl, capped
   by --llm-max-words. Environment: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_MODEL.
4. Consonant-skeleton validation and the selection rule of plan 4.4.6.
5. Writes lexicon.tsv (CC-BY-SA-4.0) and lexicon_review.tsv (flagged words).
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from lughaat_tts.lexicon import Lexicon, build_seed_lexicon, LEXICON_PATH, GOLD_PATH, WIKIPRON_TEST_PATH, DATA_DIR  # noqa: E402
from lughaat_tts.normalize import normalize_chars, strip_diacritics  # noqa: E402
from lughaat_tts.phoneset import to_lughaat_tts, is_canonical, CONSONANTS, VOWEL_CHARS  # noqa: E402
from lughaat_tts.rules import rules_phonemize_word, espeak_candidate  # noqa: E402
from lughaat_tts.skeleton import skeleton_ok  # noqa: E402

ASSETS = os.path.join(ROOT, "assets")
URDU_WORD_RE = re.compile(r"^[ؠ-ٟٮ-ۓەۡ-ۯۺ-ۿ]+$")


def log(m: str) -> None:
    print(f"[lexicon] {m}", flush=True)


def read_words(path: str, top: int = 0) -> list[str]:
    words = []
    with open(path, encoding="utf-8") as f:
        for ln in f:
            w = normalize_chars(ln.split("\t")[0].strip())
            if w and URDU_WORD_RE.match(strip_diacritics(w)) and len(strip_diacritics(w)) >= 2:
                words.append(w)
            if top and len(words) >= top:
                break
    return words


def wikipedia_words(top: int, token: str | None) -> list[str]:
    """Top-N word types from Urdu Wikipedia (streaming; stops after ~40k articles)."""
    if top <= 0:
        return []
    cnt: collections.Counter = collections.Counter()
    try:
        from datasets import load_dataset  # type: ignore
        ds = load_dataset("wikimedia/wikipedia", "20231101.ur", split="train", streaming=True, token=token)
        for i, ex in enumerate(ds):
            for w in re.findall(r"[ؠ-ٟٮ-ۓەۡ-ۯۺ-ۿ]+", ex.get("text", "")):
                if len(w) >= 2:
                    cnt[normalize_chars(w)] += 1
            if i >= 40000:
                break
    except Exception as e:
        log(f"Wikipedia word list unavailable: {e!r}")
    return [w for w, _ in cnt.most_common(top)]


# ---------------------------------------------------------------------------
# Candidate B: LLM pronunciations (optional, OpenAI Responses API compatible)
# ---------------------------------------------------------------------------
PROMPT = (
    "You are an expert Urdu phonetician. Transcribe each Urdu word into IPA using ONLY these symbols:\n"
    "consonants p b t d ʈ ɖ k ɡ q ʔ f ʋ s z ʃ ʒ x ɣ h m n ŋ l r ɽ j ʧ ʤ, aspiration marked with ʰ after the consonant "
    "(pʰ bʰ tʰ dʰ ʈʰ ɖʰ kʰ ɡʰ ʧʰ ʤʰ ɽʰ),\n"
    "vowels ə ɪ ʊ ɑː iː uː eː oː ɛː ɔː, nasalisation with a combining tilde after the vowel (ɑ̃ː ẽː ĩː ũː õː), "
    "gemination by doubling the consonant.\n"
    "No stress marks, no syllable dots, no spaces inside a word. Standard Pakistani Urdu pronunciation.\n"
    "Examples:\n{examples}\n"
    "Now transcribe these words. Answer with one line per word in the form `word<TAB>ipa` and nothing else:\n{words}"
)


class LLMCandidates:
    """Pronunciation candidates from an OpenAI-Responses-compatible endpoint (e.g. Azure).

    Reads AZURE_OPENAI_ENDPOINT (full URL ending in /responses or /chat/completions),
    AZURE_OPENAI_API_KEY and AZURE_OPENAI_MODEL. No SDK dependency; results are cached so a
    re-run never pays twice. ``max_words`` caps how many new words are sent."""

    def __init__(self, max_words: int, cache_path: str, examples: list[tuple[str, str]]):
        self.max_words = max_words
        self.sent_words = 0
        self.tokens = {"input": 0, "output": 0}
        self.cache_path = cache_path
        self.cache: dict[str, str] = {}
        if os.path.exists(cache_path):
            for ln in open(cache_path, encoding="utf-8"):
                try:
                    d = json.loads(ln)
                    self.cache[d["w"]] = d["p"]
                except Exception:
                    pass
        self.examples = examples
        self.endpoint = (os.environ.get("AZURE_OPENAI_ENDPOINT") or os.environ.get("OPENAI_BASE_URL") or "").strip()
        self.api_key = os.environ.get("AZURE_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
        self.model = os.environ.get("AZURE_OPENAI_MODEL") or os.environ.get("OPENAI_MODEL") or ""
        self.enabled = bool(max_words > 0 and self.endpoint and self.api_key and self.model)
        if max_words > 0 and not self.enabled:
            log("LLM candidates disabled: set AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY (secret) and AZURE_OPENAI_MODEL")

    # -- transport -----------------------------------------------------------------
    def _call(self, prompt: str) -> str:
        chat = self.endpoint.rstrip("/").endswith("chat/completions")
        body = ({"model": self.model, "messages": [{"role": "user", "content": prompt}]} if chat
                else {"model": self.model, "input": prompt})
        headers = {"Content-Type": "application/json", "api-key": self.api_key, "Authorization": f"Bearer {self.api_key}"}
        req = urllib.request.Request(self.endpoint, data=json.dumps(body).encode("utf-8"), headers=headers)
        with urllib.request.urlopen(req, timeout=180) as r:
            d = json.load(r)
        usage = d.get("usage") or {}
        self.tokens["input"] += int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
        self.tokens["output"] += int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
        if chat:
            return d["choices"][0]["message"]["content"]
        text = d.get("output_text")
        if not text:
            text = "".join(c.get("text", "") for o in d.get("output", []) for c in (o.get("content") or [])
                           if c.get("type") in ("output_text", "text"))
        return text or ""

    def get(self, words: list[str]) -> dict[str, str]:
        out = {w: self.cache[w] for w in words if w in self.cache}
        todo = [w for w in words if w not in self.cache]
        if not self.enabled or not todo:
            return out
        todo = todo[: max(0, self.max_words)]
        log(f"LLM candidates: {len(out)} cached, requesting {len(todo)} words from {self.model}")
        for i in range(0, len(todo), 100):
            batch = todo[i:i + 100]
            prompt = PROMPT.format(examples="\n".join(f"{w}\t{p}" for w, p in self.examples[:40]), words="\n".join(batch))
            text = ""
            for attempt in range(4):
                try:
                    text = self._call(prompt)
                    break
                except urllib.error.HTTPError as e:
                    msg = e.read().decode("utf-8", "replace")[:300]
                    log(f"LLM HTTP {e.code} (attempt {attempt + 1}): {msg}")
                    if e.code in (400, 401, 403, 404):
                        self.enabled = False
                        return out
                    time.sleep(5 * (attempt + 1))
                except Exception as e:
                    log(f"LLM call failed (attempt {attempt + 1}): {e!r}")
                    time.sleep(5 * (attempt + 1))
            self.sent_words += len(batch)
            got = 0
            with open(self.cache_path, "a", encoding="utf-8") as f:
                for ln in text.splitlines():
                    parts = ln.replace("<TAB>", "\t").split("\t")
                    if len(parts) < 2:
                        parts = ln.split()
                    if len(parts) < 2:
                        continue
                    w, p = normalize_chars(parts[0].strip().strip("`")), to_lughaat_tts(parts[1].strip().strip("`"))
                    if w in batch and p and is_canonical(p):
                        out[w] = p
                        self.cache[w] = p
                        f.write(json.dumps({"w": w, "p": p}, ensure_ascii=False) + "\n")
                        got += 1
            if (i // 100) % 10 == 0:
                log(f"  LLM batch {i // 100 + 1}: {got}/{len(batch)} usable; tokens so far {self.tokens}")
        return out


def select(word: str, a: str | None, b: str | None, review: list) -> tuple[str, str]:
    """Plan 4.4.6 selection rule. Returns (pron, source)."""
    a_ok = bool(a) and skeleton_ok(word, a)
    b_ok = bool(b) and skeleton_ok(word, b)
    if a and b and a == b:
        return a, "agree"
    if a_ok and b_ok:
        # prefer B's vowels, A's consonants: take B if its consonant skeleton equals A's
        ca = [c for c in a if c in CONSONANTS]
        cb = [c for c in b if c in CONSONANTS]
        return (b, "llm") if ca == cb else (a, "espeak")
    if b_ok:
        return b, "llm"
    if a_ok:
        return a, "espeak"
    rules = rules_phonemize_word(word)
    review.append((word, a or "", b or "", rules))
    if a:
        return a, "espeak"
    return rules, "rules"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--words", default=None, help="word list (one word per line, optional count after a tab)")
    ap.add_argument("--wiki-top", type=int, default=0)
    ap.add_argument("--out-dir", default=DATA_DIR)
    ap.add_argument("--llm-max-words", type=int, default=int(os.environ.get("LLM_LEXICON_MAX_WORDS", 0)),
                    help="send up to this many new words to the LLM (0 = off; needs AZURE_OPENAI_* env)")
    ap.add_argument("--max-new", type=int, default=120000)
    ap.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    a = ap.parse_args()

    out_lex = os.path.join(a.out_dir, "lexicon.tsv")
    test_path = os.path.join(a.out_dir, "wikipron_test.tsv")
    reports = os.path.join(ROOT, "reports")
    os.makedirs(reports, exist_ok=True)

    # 1. seed (keep an existing held-out split stable)
    if not os.path.exists(test_path) or not os.path.exists(out_lex):
        lex, test_rows = build_seed_lexicon(os.path.join(ASSETS, "urd_arab_broad.tsv"), os.path.join(ASSETS, "urd_arab_narrow.tsv"),
                                            GOLD_PATH, out_lex, test_path)
        log(f"seed lexicon: {len(lex)} entries, {len(test_rows)} held out")
    lex = Lexicon.load(out_lex, extra=[GOLD_PATH])
    test_words = set()
    for ln in open(test_path, encoding="utf-8"):
        if ln.strip() and not ln.startswith("#"):
            test_words.add(ln.split("\t")[0])

    # 2. word list
    words: list[str] = []
    if a.words and os.path.exists(a.words):
        words += read_words(a.words)
    words += wikipedia_words(a.wiki_top, a.hf_token)
    seen = set()
    new_words = []
    for w in words:
        key = strip_diacritics(w)
        if key in seen or key in test_words or lex.get(w) is not None:
            continue
        seen.add(key)
        new_words.append(w)
    new_words = new_words[: a.max_new]
    log(f"{len(new_words)} new word types to pronounce")

    # 3. candidates
    examples = [(w, e.pron) for w, e in list(lex.entries.items()) if e.source in ("gold", "wikipron")][:40]
    llm = LLMCandidates(a.llm_max_words, os.path.join(reports, "llm_cache.jsonl"), examples)
    b_map = llm.get(new_words) if (llm.enabled or llm.cache) else {}
    espeak_available = espeak_candidate("پاکستان") is not None
    log(f"espeak-ng available: {espeak_available}; LLM candidates: {len(b_map)}")

    review: list = []
    counts = collections.Counter()
    for i, w in enumerate(new_words):
        cand_a = espeak_candidate(w) if espeak_available else None
        if cand_a is not None and not is_canonical(cand_a):
            cand_a = None
        if cand_a is None:
            cand_a = rules_phonemize_word(w)
            src_a_is_rules = True
        else:
            src_a_is_rules = False
        pron, source = select(w, cand_a, b_map.get(w), review)
        if source == "espeak" and src_a_is_rules:
            source = "rules"
        if pron and is_canonical(pron):
            lex.add(w, pron, source)
            counts[source] += 1
        if (i + 1) % 5000 == 0:
            log(f"  {i+1}/{len(new_words)} {dict(counts)}")
    lex.save(out_lex, header=("lughaat-tts pronunciation lexicon. word<TAB>pron<TAB>source<TAB>variants\n"
                              "Derived in part from WikiPron (CUNY-CL/wikipron; Wiktionary data).\n"
                              "Licence of this file: CC-BY-SA-4.0. Model weights are unaffected."))
    with open(os.path.join(reports, "lexicon_review.tsv"), "w", encoding="utf-8") as f:
        f.write("# word\tespeak\tllm\trules  (no candidate passed the consonant-skeleton check)\n")
        for row in review:
            f.write("\t".join(row) + "\n")
    summary = {"entries": len(lex), "new_words": len(new_words), "by_source": dict(counts),
               "sources_total": dict(collections.Counter(e.source for e in lex.entries.values())),
               "review": len(review), "llm_words_sent": llm.sent_words, "llm_tokens": llm.tokens,
               "llm_model": llm.model if llm.enabled or llm.cache else None, "espeak_available": espeak_available}
    with open(os.path.join(reports, "lexicon_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log(f"saved {out_lex}: {summary}")


if __name__ == "__main__":
    main()

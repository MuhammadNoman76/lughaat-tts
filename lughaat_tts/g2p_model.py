"""Neural grapheme-to-phoneme model for out-of-lexicon Urdu words (plan 4.5).

A small character-to-phone Transformer (3+3 layers, d_model 256, ~3.5M parameters).
Input: Urdu letters (with diacritics when present). Output: phone units of the canonical
set (see ``phoneset.split_units``). Decoding is beam search (beam 4) with a maximum
length of 2.5x the input.

Only ``torch`` is required. The trained weights live in ``lughaat_tts/data/g2p_model.pt``
(a dict with ``state_dict``, ``src_vocab``, ``tgt_vocab`` and ``hparams``) and are
downloaded from the model repo when missing.
"""
from __future__ import annotations

import math
import os
import random
import time
from dataclasses import dataclass, asdict
from typing import Iterable, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .phoneset import split_units, to_lughaat_tts

PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "g2p_model.pt")


@dataclass
class HParams:
    d_model: int = 256
    nhead: int = 4
    num_encoder_layers: int = 3
    num_decoder_layers: int = 3
    dim_feedforward: int = 1024
    dropout: float = 0.1
    max_len: int = 96
    # rules-conditioned: the encoder input is  letters <sep> rule-based phones ; the model learns
    # to correct the deterministic rules (plan 4.5 "iterate"), which is far more data-efficient
    # than learning Urdu orthography from ~6k words alone.
    rules_conditioned: bool = True


SEP = "<sep>"


def source_tokens(word: str, hp: "HParams") -> list[str]:
    toks = list(word)
    if hp.rules_conditioned:
        from .rules import rules_phonemize_word
        toks = toks + [SEP] + split_units(rules_phonemize_word(word))
    return toks


class Vocab:
    def __init__(self, tokens: Sequence[str]):
        self.itos = [PAD, BOS, EOS, UNK] + [t for t in tokens if t not in (PAD, BOS, EOS, UNK)]
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, toks: Iterable[str]) -> list[int]:
        return [self.stoi.get(t, self.stoi[UNK]) for t in toks]

    def decode(self, ids: Iterable[int]) -> list[str]:
        out = []
        for i in ids:
            t = self.itos[i]
            if t == EOS:
                break
            if t in (PAD, BOS):
                continue
            out.append(t)
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class G2PTransformer(nn.Module):
    def __init__(self, n_src: int, n_tgt: int, hp: HParams):
        super().__init__()
        self.hp = hp
        self.src_emb = nn.Embedding(n_src, hp.d_model, padding_idx=0)
        self.tgt_emb = nn.Embedding(n_tgt, hp.d_model, padding_idx=0)
        self.pos = PositionalEncoding(hp.d_model, max_len=max(512, hp.max_len * 4))
        self.transformer = nn.Transformer(
            d_model=hp.d_model, nhead=hp.nhead, num_encoder_layers=hp.num_encoder_layers,
            num_decoder_layers=hp.num_decoder_layers, dim_feedforward=hp.dim_feedforward,
            dropout=hp.dropout, batch_first=True, norm_first=True,
        )
        self.out = nn.Linear(hp.d_model, n_tgt)
        self.scale = math.sqrt(hp.d_model)

    def encode(self, src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        src_pad = src.eq(0)
        mem = self.transformer.encoder(self.pos(self.src_emb(src) * self.scale), src_key_padding_mask=src_pad)
        return mem, src_pad

    def decode_step(self, tgt: torch.Tensor, mem: torch.Tensor, src_pad: torch.Tensor) -> torch.Tensor:
        T = tgt.size(1)
        causal = torch.triu(torch.full((T, T), float("-inf"), device=tgt.device), diagonal=1)
        h = self.transformer.decoder(
            self.pos(self.tgt_emb(tgt) * self.scale), mem, tgt_mask=causal,
            tgt_key_padding_mask=tgt.eq(0), memory_key_padding_mask=src_pad,
        )
        return self.out(h)

    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor) -> torch.Tensor:
        mem, src_pad = self.encode(src)
        return self.decode_step(tgt_in, mem, src_pad)


class NeuralG2P:
    """Inference wrapper: ``NeuralG2P.load(path)(word) -> phoneme string``."""

    def __init__(self, model: G2PTransformer, src_vocab: Vocab, tgt_vocab: Vocab, device: str = "cpu", meta: Optional[dict] = None):
        self.model = model.to(device).eval()
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.device = device
        self.meta: dict = dict(meta or {})   # dev_per / dev_wer / rules_per recorded at training time
        self._cache: dict[str, str] = {}

    @classmethod
    def load(cls, path: str = DEFAULT_MODEL_PATH, device: str = "cpu") -> "NeuralG2P":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        hparams = dict(ckpt["hparams"])
        hparams.setdefault("rules_conditioned", False)   # models trained before the corrector design
        hp = HParams(**hparams)
        src_vocab = Vocab(ckpt["src_vocab"][4:])
        tgt_vocab = Vocab(ckpt["tgt_vocab"][4:])
        model = G2PTransformer(len(src_vocab), len(tgt_vocab), hp)
        model.load_state_dict(ckpt["state_dict"])
        return cls(model, src_vocab, tgt_vocab, device, meta=ckpt.get("meta"))

    @property
    def beats_rules(self) -> bool:
        """True unless the checkpoint records a held-out PER (rerank selection if available, else
        top-1) that is worse than the rules baseline."""
        per = self.meta.get("dev_per_rerank", self.meta.get("dev_per"))
        rules = self.meta.get("rules_per")
        if per is None or rules is None:
            return True
        return per <= rules

    @torch.no_grad()
    def predict_nbest(self, word: str, beam: int = 4, n: int = 3) -> list[tuple[str, float]]:
        src_ids = [1] + self.src_vocab.encode(source_tokens(word, self.model.hp))[: self.model.hp.max_len - 2] + [2]
        src = torch.tensor([src_ids], device=self.device)
        mem, src_pad = self.model.encode(src)
        max_out = min(int(len(word) * 2.5) + 4, self.model.hp.max_len * 2)
        beams: list[tuple[list[int], float, bool]] = [([1], 0.0, False)]
        for _ in range(max_out):
            cand: list[tuple[list[int], float, bool]] = []
            alive = [b for b in beams if not b[2]]
            if not alive:
                break
            tgt = torch.tensor([b[0] for b in alive], device=self.device)
            logits = self.model.decode_step(tgt, mem.expand(len(alive), -1, -1), src_pad.expand(len(alive), -1))
            logp = F.log_softmax(logits[:, -1, :], dim=-1)
            topv, topi = logp.topk(beam, dim=-1)
            for bi, b in enumerate(alive):
                for v, i in zip(topv[bi].tolist(), topi[bi].tolist()):
                    cand.append((b[0] + [i], b[1] + v, i == 2))
            cand += [b for b in beams if b[2]]
            cand.sort(key=lambda x: x[1] / (len(x[0]) ** 0.7), reverse=True)
            beams = cand[:beam]
            if all(b[2] for b in beams):
                break
        out: list[tuple[str, float]] = []
        for ids, score, _ in beams:
            ph = to_lughaat_tts("".join(self.tgt_vocab.decode(ids[1:])))
            if ph and ph not in [o[0] for o in out]:
                out.append((ph, score))
        return out[:n]

    def __call__(self, word: str) -> str:
        if word in self._cache:
            return self._cache[word]
        nb = self.predict_nbest(word, beam=4, n=1)
        res = nb[0][0] if nb else ""
        self._cache[word] = res
        return res


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
def _batches(pairs: list[tuple[list[int], list[int]]], bs: int, shuffle: bool):
    idx = list(range(len(pairs)))
    if shuffle:
        random.shuffle(idx)
    for i in range(0, len(idx), bs):
        chunk = [pairs[j] for j in idx[i: i + bs]]
        S = max(len(s) for s, _ in chunk)
        T = max(len(t) for _, t in chunk)
        src = torch.zeros(len(chunk), S, dtype=torch.long)
        tgt = torch.zeros(len(chunk), T, dtype=torch.long)
        for k, (s, t) in enumerate(chunk):
            src[k, : len(s)] = torch.tensor(s)
            tgt[k, : len(t)] = torch.tensor(t)
        yield src, tgt


def _augment(word: str, rng: random.Random) -> str:
    """Randomly strip diacritics so the model works on undiacritised text too."""
    from .normalize import HARAKAT
    if any(c in HARAKAT for c in word) and rng.random() < 0.5:
        return "".join(c for c in word if c not in HARAKAT)
    return word


def train_g2p(
    train_pairs: list[tuple[str, str]],
    out_path: str,
    dev_pairs: Optional[list[tuple[str, str]]] = None,
    epochs: int = 40,
    batch_size: int = 128,
    lr: float = 7e-4,
    hp: Optional[HParams] = None,
    device: Optional[str] = None,
    seed: int = 0,
    log=print,
    meta: Optional[dict] = None,
) -> dict:
    """Train on (word, canonical pron) pairs; returns metrics dict. Deterministic given seed.
    *meta* (e.g. the rules-only baseline PER) is stored in the checkpoint for the runtime ordering."""
    torch.manual_seed(seed)
    rng = random.Random(seed)
    hp = hp or HParams()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    src_chars = sorted({c for w, _ in train_pairs for c in source_tokens(w, hp)})
    tgt_units = sorted({u for _, p in train_pairs for u in split_units(p)})
    src_vocab, tgt_vocab = Vocab(src_chars), Vocab(tgt_units)
    model = G2PTransformer(len(src_vocab), len(tgt_vocab), hp).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"[g2p] {len(train_pairs)} train pairs, {len(src_vocab)} src symbols, {len(tgt_vocab)} phone units, {n_params/1e6:.2f}M params, device={device}")

    def enc(w: str, p: str) -> tuple[list[int], list[int]]:
        return ([1] + src_vocab.encode(source_tokens(w, hp))[: hp.max_len - 2] + [2],
                [1] + tgt_vocab.encode(split_units(p))[: hp.max_len * 2 - 2] + [2])

    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98), weight_decay=0.01)
    steps_per_epoch = math.ceil(len(train_pairs) / batch_size)
    total = steps_per_epoch * epochs
    warm = max(100, total // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * max(0.05, 0.5 * (1 + math.cos(math.pi * min(1.0, s / total)))))
    best = (float("inf"), None)
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        pairs = [enc(_augment(w, rng), p) for w, p in train_pairs]
        tot, n = 0.0, 0
        for src, tgt in _batches(pairs, batch_size, shuffle=True):
            src, tgt = src.to(device), tgt.to(device)
            logits = model(src, tgt[:, :-1])
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1), ignore_index=0, label_smoothing=0.1)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item() * src.size(0)
            n += src.size(0)
        msg = f"[g2p] epoch {ep}/{epochs} loss {tot/n:.4f} ({time.time()-t0:.0f}s)"
        if dev_pairs and (ep % 5 == 0 or ep == epochs):
            g = NeuralG2P(model, src_vocab, tgt_vocab, device)
            m = evaluate_g2p(g, dev_pairs)
            msg += f" dev PER {m['per']:.3f} WER {m['wer']:.3f}"
            if m["per"] < best[0]:
                best = (m["per"], {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
            model.train()
        log(msg)
    state = best[1] if best[1] is not None else {k: v.detach().cpu() for k, v in model.state_dict().items()}
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    ckpt = {"state_dict": state, "src_vocab": src_vocab.itos, "tgt_vocab": tgt_vocab.itos, "hparams": asdict(hp), "meta": dict(meta or {})}
    torch.save(ckpt, out_path)
    g = NeuralG2P.load(out_path, device)
    metrics = evaluate_g2p(g, dev_pairs) if dev_pairs else {}
    metrics.update({"n_train": len(train_pairs), "n_params": n_params, "epochs": epochs})
    ckpt["meta"].update({"dev_per": metrics.get("per"), "dev_wer": metrics.get("wer"), "n_train": len(train_pairs)})
    torch.save(ckpt, out_path)
    log(f"[g2p] saved {out_path} ({os.path.getsize(out_path)/1e6:.1f} MB) metrics={metrics}")
    return metrics


def edit_distance(a: Sequence, b: Sequence) -> int:
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


RERANK_LAMBDA = 0.5      # penalty per phone-unit edit between a candidate and the rule output
RERANK_RULES_PRIOR = -2.0  # length-normalised log-prob assigned to the rule output itself


def rerank_select(word: str, nbest: list[tuple[str, float]], rules_pron: str,
                  lam: float = RERANK_LAMBDA, rules_prior: float = RERANK_RULES_PRIOR) -> tuple[str, str]:
    """Rules-anchored rerank of neural n-best candidates (plan 4.5 "iterate").

    Candidates must pass the consonant-skeleton check; each is scored by its length-normalised
    log-probability minus ``lam`` x (edit distance to the deterministic rule output). The rule
    output competes with a fixed prior, so a confident neural candidate close to the rules wins,
    while far-fetched candidates lose to the rules. Returns (pronunciation, source)."""
    from .skeleton import skeleton_ok
    ru = split_units(rules_pron)
    best, best_score, src = rules_pron, rules_prior, "rules"
    for p, logp in nbest:
        if not p or not skeleton_ok(word, p):
            continue
        pu = split_units(p)
        score = logp / (len(pu) ** 0.7) - lam * edit_distance(pu, ru)
        if score > best_score:
            best, best_score, src = p, score, "neural"
    return best, src


def evaluate_g2p(g2p, pairs: list[tuple[str, str]], rerank: bool = False) -> dict:
    """Phone error rate (unit level) and word error rate on (word, gold) pairs.
    With ``rerank=True`` (needs a NeuralG2P), candidates are selected by :func:`rerank_select`."""
    errs, total, wrong = 0, 0, 0
    if rerank:
        from .rules import rules_phonemize_word
    for w, gold in pairs:
        if rerank:
            hyp, _ = rerank_select(w, g2p.predict_nbest(w, beam=5, n=5), rules_phonemize_word(w))
        else:
            hyp = g2p(w)
        gu, hu = split_units(to_lughaat_tts(gold)), split_units(hyp)
        errs += edit_distance(gu, hu)
        total += len(gu)
        wrong += int(gu != hu)
    return {"per": errs / max(total, 1), "wer": wrong / max(len(pairs), 1), "n": len(pairs)}


__all__ = ["NeuralG2P", "train_g2p", "evaluate_g2p", "rerank_select", "edit_distance", "HParams", "DEFAULT_MODEL_PATH",
           "source_tokens", "RERANK_LAMBDA", "RERANK_RULES_PRIOR"]

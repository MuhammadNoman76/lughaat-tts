#!/usr/bin/env python3
"""Train the neural G2P on the lexicon and report PER/WER on the held-out WikiPron words.

    python scripts/03_train_g2p.py --epochs 40 --out lughaat_tts/data/g2p_model.pt

Targets (plan 4.5): PER <= 8 %, WER <= 25 % on wikipron_test.tsv. The script prints the
rules-only baseline for comparison and writes metrics to reports/g2p_metrics.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from lughaat_tts.lexicon import Lexicon, LEXICON_PATH, GOLD_PATH, WIKIPRON_TEST_PATH  # noqa: E402
from lughaat_tts.g2p_model import train_g2p, evaluate_g2p, HParams, DEFAULT_MODEL_PATH  # noqa: E402
from lughaat_tts.rules import rules_phonemize_word  # noqa: E402
from lughaat_tts.phoneset import to_lughaat_tts  # noqa: E402


def read_pairs(path: str) -> list[tuple[str, str]]:
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            w, p = line.rstrip("\n").split("\t")[:2]
            pairs.append((w, to_lughaat_tts(p)))
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lexicon", default=LEXICON_PATH)
    ap.add_argument("--gold", default=GOLD_PATH)
    ap.add_argument("--test", default=WIKIPRON_TEST_PATH)
    ap.add_argument("--out", default=DEFAULT_MODEL_PATH)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=7e-4)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--device", default=None)
    ap.add_argument("--report", default=os.path.join(ROOT, "reports", "g2p_metrics.json"))
    a = ap.parse_args()

    lex = Lexicon.load(a.lexicon, extra=[a.gold])
    test_pairs = read_pairs(a.test)
    test_words = {w for w, _ in test_pairs}
    train_pairs = [(w, e.pron) for w, e in lex.entries.items() if w not in test_words and " " not in e.pron]
    # variants are extra supervision
    for w, e in lex.entries.items():
        if w in test_words:
            continue
        for v in e.variants[:1]:
            if v and " " not in v:
                train_pairs.append((w, v))
    print(f"train pairs: {len(train_pairs)}  test pairs: {len(test_pairs)}")

    base = evaluate_g2p(rules_phonemize_word, test_pairs)
    print(f"rules-only baseline on wikipron_test: PER {base['per']:.3f} WER {base['wer']:.3f}")

    hp = HParams(d_model=a.d_model, num_encoder_layers=a.layers, num_decoder_layers=a.layers, dim_feedforward=a.d_model * 4)
    metrics = train_g2p(train_pairs, a.out, dev_pairs=test_pairs, epochs=a.epochs, batch_size=a.batch_size, lr=a.lr, hp=hp, device=a.device,
                        meta={"rules_per": base["per"], "rules_wer": base["wer"], "rules_conditioned": hp.rules_conditioned})
    metrics["rules_baseline"] = base
    # runtime selection = rules-anchored rerank of the n-best; record its held-out accuracy in the checkpoint
    import torch
    from lughaat_tts.g2p_model import NeuralG2P
    g = NeuralG2P.load(a.out)
    rr = evaluate_g2p(g, test_pairs, rerank=True)
    metrics["rerank_per"], metrics["rerank_wer"] = rr["per"], rr["wer"]
    ckpt = torch.load(a.out, map_location="cpu", weights_only=False)
    ckpt.setdefault("meta", {}).update({"dev_per_rerank": rr["per"], "dev_wer_rerank": rr["wer"], "rules_per": base["per"], "rules_wer": base["wer"]})
    torch.save(ckpt, a.out)
    print(f"rerank (runtime selection) on wikipron_test: PER {rr['per']:.3f} WER {rr['wer']:.3f}  (rules {base['per']:.3f} / top-1 neural {metrics.get('per', float('nan')):.3f})")
    metrics["targets_met"] = bool(rr["per"] <= 0.08 and rr["wer"] <= 0.25)
    os.makedirs(os.path.dirname(a.report), exist_ok=True)
    with open(a.report, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if not metrics["targets_met"]:
        print("WARNING: G2P targets (PER<=8%, WER<=25%) not met; see plan 4.5 for iteration ideas.")


if __name__ == "__main__":
    main()

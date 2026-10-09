# Lughaat-TTS-82M training report (template)

This file is rewritten by `kaggle/train_session.py` as the run progresses (data hours, G2P accuracy,
per-epoch metrics, the chosen checkpoints, time projections, measured GPU hours and the final evaluation).
Until the first Kaggle session has run it only documents what was verified locally.

## Local verification (before any GPU time)

- Frontend package built and tested on Windows with Python 3.12 (`pytest tests`): Appendix B gold words,
  Appendix A2 language tags, the seven Pakistani-English anchor words, vocabulary coverage of every
  Appendix sentence, determinism.
- Lexicon: WikiPron `urd_arab_broad` + `urd_arab_narrow` (6,296 word types) mapped to the canonical set,
  10 % (626 words) held out in `wikipron_test.tsv`; 394 gold overrides (5,790 entries shipped).
- Neural G2P on the 626 held-out WikiPron words (`reports/g2p_metrics.json`), CPU training on the seed lexicon:

  | selector | PER | WER |
  |---|---|---|
  | rules only | 0.193 | 0.633 |
  | chars-only Transformer, top-1 | 0.213 | 0.588 |
  | chars-only + rules-anchored rerank | 0.171 | 0.582 |
  | rules-conditioned Transformer, top-1 | 0.217 | 0.575 |
  | **rules-conditioned + rerank (shipped, `lughaat_tts/data/g2p_model.pt`)** | **0.163** | **0.545** |

  The plan's PER ≤ 8 % target is not reachable from 5.8k seed words (and WikiPron variants make the
  WER floor high); the Kaggle `lexicon` step retrains on the expanded lexicon and re-measures.
- ONNX export of the base weights (`scripts/09_onnx.py`): fp32 graph vs the exported PyTorch model on 20
  Appendix sentences: min waveform correlation 0.996, mean 0.998, log-mel correlation 0.999, identical
  lengths (gate ≥ 0.99 passed). Against Kokoro's default complex-STFT path the correlation is 0.95 (an
  inherent difference between Kokoro's two iSTFT implementations, reported as information).
  fp16: the conversion needs ONNX shape inference and onnxruntime emulates fp16 on CPU, so the step runs in
  a 25-minute bounded subprocess (`--fp16-timeout-min`) and the fp16 file is shipped only if it passes the
  same parity gate; otherwise the fp32 graph is shipped alone (plan 10.5 allows this). On this PC the
  bounded local check did not complete in time (see below for the final status if it did).
- Trainer patches applied cleanly and idempotently to the pinned kikiri-tts checkout; all patched files compile.
- misaki American base reproduces all plan 4.7 anchors after the Pakistani mapping.
- Shard runner (`scripts/06_train_shard.py`) integration-tested with a stub trainer: fresh Stage-2 shard
  from `first_stage.pth`, resume at a later epoch/shard with the logical epoch rewritten, and a memory
  probe all produce the expected config, checkpoint hand-off and result JSON.
- Export path tested on the base `hexgrad/Kokoro-82M` weights: `convert_base.py` -> `07_export.py`
  -> strict KModel verification (bert 25/25, bert_encoder 2/2, predictor 122/122, text_encoder 24/24,
  decoder 375/375 tensors matched; 81.81 M params; forward pass OK). The only tensors KModel has that
  no checkpoint provides are the AdaIN InstanceNorm affine parameters, exactly as in upstream Kokoro.
- Phase-3 Hindi-voice baseline samples synthesised locally on CPU (`scripts/04_baseline.py --skip-asr`):
  156 WAVs (Appendix A + A2 with hm_psi / hf_alpha / hf_beta / hm_omega, Appendix A3 with af_heart) in
  `baseline_samples/`; the ASR scores are computed on Kaggle (whisper-large-v3) in the `baseline` phase.
- Evaluation harness (`eval/evaluate.py --light`) run end to end on CPU against the re-exported base weights
  with Hindi voicepacks as stand-ins and `openai/whisper-tiny`: 40 sentences, CER / PCER / WER / signal
  checks / gates / per-sentence JSONL all produced (numbers are not meaningful for quality; the run
  surfaced and fixed the roll-off percentile and comb-gate issues recorded in DECISIONS.md 29–30).

## Compute

Kaggle free tier, GPU T4 x2 (2 x 16 GB), ~30 GPU-hours/week; cost $0. Projections are appended below
after the first calibration shard of each stage.

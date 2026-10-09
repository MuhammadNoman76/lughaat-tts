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

## Code and baseline pronunciation audit — 2026-10-09

Audited all 156 baseline WAVs (594.625 seconds). No original or regenerated file
triggered the empty/nonfinite audio, wrong sample rate, clipping, or long internal
silence checks. These checks do not measure pronunciation accuracy. ASR and human
listening validation were not performed in this audit.

The saved baseline phonemes were partly stale relative to the shipped neural G2P.
Confirmed issues include جیسے being saved as /ʤiːseː/, بھیج as /bʰiːʤ/, and the
verb کیا being given the question pronunciation in خطاب کیا and آن کیا.
The verb/question distinction is documented in
[Wiktionary's Urdu entry](https://en.wiktionary.org/wiki/کیا);
بھیج follows the بھیجنا stem in the bundled WikiPron data. The pinned جیسے
reading follows Hindustani jaise, also documented in
[the cognate dictionary entry](https://en.wiktionary.org/wiki/जैसे).

Changes in frontend ur-frontend-1.0.1:
- Correct Unicode handling of long nasal /a/, ä, and ō.
- Apply productive izafat before diacritic-insensitive lexicon lookup.
- Add explicit marked homographs and pin جیسے / بھیج.
- Raise on unsupported output phones instead of silently deleting them.
- Provide explicit sample-text edits for ambiguous words; these are annotations,
  not an automatic context-disambiguation model.

Generated 20 candidate WAVs covering five sample identifiers across four voices
(A_04, A_15, A_16, A2_02, A2_04). The remaining recordings are included in the
review page. A_06's explicit جَلْد annotation prevents a current neural fallback
regression but reproduces its original saved phonemes, so it needs no new candidate.

Open [the comparison page](pronunciation_review/listening_test.html) or inspect
[the per-file audit](pronunciation_review/audit.json). All 156 original hashes and
all 176 audio links were verified. All 211 tests passed; every shipped lexicon
entry has in-vocabulary phones. The acoustic model and voicepacks were not trained
or changed. Audible improvement and complete pronunciation correctness remain unverified.

### Unresolved issues before claiming high accuracy

1. **Quality failures do not block upload.** In
   `kaggle/train_session.py:621-627`, failed final gates are logged and the session
   still advances. Separately, `eval/evaluate.py:225` ignores unavailable gates
   when computing `gates_passed`. Required full-evaluation measurements should
   be present and passing before a release is described as quality-approved.
2. **The reported G2P test is also used for checkpoint selection.**
   `scripts/03_train_g2p.py:68` passes the test pairs as development pairs, and
   `lughaat_tts/g2p_model.py:281-289` chooses weights using their error rate.
   The recorded 16.3% PER is development-selected performance, not an untouched
   final-test estimate. Use separate training, development, and final test sets.
3. **Unmarked Urdu words are still pronounced without sentence meaning.**
   Exact marked entries provide manual control, but unmarked homographs still
   use a single lexicon/default reading. More GPU training alone does not fix
   wrong input phonemes. The baseline voices are unmodified Hindi Kokoro voices,
   not evidence of a completed Urdu acoustic fine-tune.

For the accuracy goal, first establish an independent Urdu pronunciation test set
and native-speaker listening results. Correct the training labels before acoustic
fine-tuning, then compare systems on identical held-out text, pronunciation errors,
naturalness, omissions, and latency. No world-leading ranking has been established.

# Decisions

Each non-obvious choice and the reason (plan rule 2.8). The pipeline appends run-time decisions
(variant probes, checkpoint selection) to `REPORT.md`; design-time decisions are here.

## Naming

0. **The model is called `lughaat-tts-82m`** (user decision, 2026-10-09: "lughaat-tts-<number of
   parameters>"; لغات = lexicons; 82 M parameters, unchanged by fine-tuning). Repos:
   `<user>/lughaat-tts-82m` (model), `<user>/lughaat-tts-work` (private working state),
   `<user>/lughaat-tts-demo` (Space); weights `lughaat-tts-82m.pth`, ONNX `lughaat-tts-82m.onnx`;
   Kaggle dataset `lughaat-tts-code`, kernel `lughaat-tts-train`; pip distribution `lughaat-tts` with
   CLI `lughaat-tts`. The Python package is `lughaat_tts` as well (second user decision, 2026-10-09: no other package name
   anywhere). The plan file itself still says `kokoro-82m-urdu`; this decision overrides it.

## Frontend

1. **ڈاکٹر is `ɖɑːkʈər`, not the plan's `ɖɑːktər`.** The letter ٹ is retroflex and WikiPron agrees
   (`ɖ ɑː k ʈ ə ɾ`); the plan's table entry is treated as a typo. Appendix B consonants are otherwise
   authoritative; `eval/testsets.py` carries the corrected table.
2. **Gold overrides sit above WikiPron.** WikiPron lists several variants for many function words
   (ہے: eː / ɛː / ɦeː / ɦɛː). A hand-checked table of ~400 high-frequency words and Appendix-B words
   (`gold_overrides.tsv`) wins; for the rest, the variant that passes the consonant-skeleton check and
   has the most explicit length marks is chosen (`lexicon.choose_variant`).
3. **ʱ → ʰ, ɦ → h, dental diacritics dropped, ɾ → r, w → ʋ, a → ə, ʕ → ʔ** in `phoneset.to_lughaat_tts`.
   WikiPron's inventory (137 symbols) is mapped onto the canonical set; `d͡ʒ`/`t͡ʃ` and `dʤ`/`tʧ`
   (geminates) become `ʤ`/`ʧ`/`ʤʤ`/`ʧʧ`. NFD is applied so nasal vowels use the combining tilde (id 17).
4. **Rule-based G2P is pure Python** (no espeak). espeak-ng is only an optional candidate during
   lexicon building. The rules reach ~19 % PER on held-out WikiPron words (rules alone); they are the
   third fallback after the lexicon and the neural model and they feed the skeleton validator.
5. **Short-vowel insertion** uses a syllable heuristic with legal coda clusters (`rules.LEGAL_CODA`,
   `LONG_OK_FINAL`): no initial clusters, at most coda+onset medially, final CC kept after a long vowel
   only for st/ʃt/nd/... (دوست doːst) and broken otherwise (موسم mɔːsəm).
6. **Neural G2P**: 3+3-layer Transformer, d_model 256 (≈5.6 M params), label smoothing 0.1,
   random diacritic stripping as augmentation, beam 5. Trained first on the WikiPron seed (shipped),
   retrained on Kaggle on the expanded lexicon. Two findings from the seed training (626 held-out words):
   - a chars-only model reached PER 0.213, *worse* than the pure rules (0.193); the oracle of
     neural-or-rules was 0.137 and the gold answer was usually in the n-best list;
   - therefore runtime selection is a **rules-anchored rerank** (`g2p_model.rerank_select`): skeleton-valid
     n-best candidates are scored by length-normalised log-prob minus 0.5 x edit distance to the rule
     output, and the rule output itself competes with a prior of −2. Measured PER 0.171 on the held-out
     words (rules 0.193, neural top-1 0.214). The **rules-conditioned variant** (encoder sees the letters
     plus the rule phones, `HParams.rules_conditioned`) with the same rerank reached **PER 0.163 / WER 0.545**
     on the held-out words and is the shipped seed model; the checkpoint stores these numbers in
     `meta.dev_per_rerank` / `meta.rules_per`.
   The plan's PER ≤ 8 % target is not reachable from 5.8k WikiPron words; it is re-measured on Kaggle
   after the lexicon is expanded with the Rasa and Wikipedia vocabularies (every Rasa word type is in the
   lexicon, so the neural model only matters for out-of-vocabulary words at inference).
7. **Pakistani-English mapping uses the American misaki base** (`british=False`). All seven anchor
   words of plan 4.7 come out exactly as specified (`tests/test_codeswitch.py`). misaki writes the flap
   as `T`, so both `T` and `ɾ` map to `ʈ`. Retroflexion of t/d is applied *before* θ/ð → tʰ/d so
   "thank" keeps a dental tʰ. The British base is implemented (`british_base=True`) for the Phase-7
   A/B; the default stays American until the listening test says otherwise.
8. **Spelled acronyms bypass misaki** (`english.spell_letters`): misaki reads a lone "A" as the
   article (ə). Pronounceable all-caps words of ≥ 4 letters (NADRA, WAPDA) are spoken as words;
   a small override list decides borderline cases (PTI, USB, SIM).
9. **Urdu punctuation is excluded from the word tokenizer** (۔ ؟ ، ؛ live inside the Arabic Unicode
   block) so sentence-final ۔ becomes "." and is attached to the previous word, like Kokoro's English pipeline.
10. **Year marker ء after a number is silent** (۱۹۴۷ء → انیس سو سینتالیس); single letters are read by
    their names (ق → qɑːf) so sentences about letters work.
11. **Numbers**: years 1100–1999 are read as "X سو Y"; other 4-digit numbers as thousands unless a year
    context word follows (میں, کو, ء ...). Times use پونے/سوا/ساڑھے/ڈیڑھ/ڈھائی. Digits glued to
    Latin words stay for misaki.

## Training

12. **Pinned upstream commits** (kikiri-tts a12d041, StyleTTS2 fork b1956da, kokoro fork b96fef9).
    All trainer patches in `training/patches.py` are exact-anchor replacements that fail loudly if the
    file changed; applying them twice is a no-op. Verified locally against the pinned checkout.
13. **Trainer patches beyond the Kannada set**: `TB-SKIP` (the kikiri trainers extract a 200-clip
    voicepack and synthesise German test sentences every epoch; skipped), `ANOMALY-OFF`
    (`set_detect_anomaly(True)` in train_second.py costs a lot of speed), `SLM-OFF` (Stage-2-lite A
    never loads WavLM), `SLMADV-NONE` restructured so a `None` adversary result no longer skips the
    logging/`iters` counters, `RESUME-FIX-B` guarded by `config.init_predictor_encoder` instead of
    `start_epoch == 0` because with shards a resume inside epoch 0 is normal.
14. **The LR schedulers are never stepped in either upstream trainer**, so the learning rate is a
    constant `max_lr`. Shards therefore do not disturb any schedule; `SCHED-SAVE` is kept for completeness.
15. **Shards instead of epochs**: one logical epoch = K shards of `train_list.txt` (deterministic
    per-epoch shuffle). Each shard is a separate trainer process that resumes weights + optimizer and
    rewrites the checkpoint's `epoch` field to the logical epoch, so TMA/diff/joint gating sees the right
    epoch. K is calibrated from a 600-clip calibration shard; a killed session loses ≤ one shard.
16. **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments` is deliberately NOT set** during Stage 2: with it a
    too-small GPU deadlocks silently instead of raising OOM; without it the memory probe gets a loud
    error and falls back to the next variant. A 25-minute no-progress watchdog catches the remaining hangs.
17. **Stage-2 variants are probed in order** (L4 full → T4x2 full b1/GPU ml300 → lite A without the
    WavLM adversary → lite B without GANs); the first that survives a 30-step probe is used; the chosen
    variant and peak memory are recorded in `state.json` / `REPORT.md`. Stage 1 probes fp16 b8 → fp16 b4
    → fp32 b4 → fp32 b4 ml300.
18. **Pinned torch 2.4.1 venv for the trainers** (the Kannada-proven stack) inside `/kaggle/tmp`, while the
    session itself uses Kaggle's Python for data prep / evaluation. Trade-off: ~5 extra minutes per
    session for reproducibility.
19. **`multispeaker: false`** as in the kikiri/Kannada recipes (style from the clip itself); speaker ids
    in the lists are still real so the reference sampler works and the voicepacks are per speaker.
20. **Data lives only in the HF work repo** (`audio.tar` ≈ 2.6 GB, downloaded once per session, ~1 min).
    A Kaggle Dataset for the audio would need Kaggle API credentials inside the kernel; not worth it.
21. **Rasa selection**: neutral styles first up to the per-speaker cap, emotional styles kept as a
    0.5× reserve and used only to fill the cap (plan 14.4); all selected styles are trained on, only
    neutral clips feed the voicepacks (plan 3.3). Style names are classified by keyword
    (HAPP/SAD/ANG/FEAR/DISG/SURP); the real distribution is logged in `data_report.json`.
22. **Whisper large-v3-turbo for filtering, large-v3 for evaluation** (plan 14.4 / 9); beam 1 for
    filtering to save quota. The FLEURS ur_pk floor is measured with the same model and settings.
23. **Clips > 20 s are split only when the silence count matches the sentence count**, otherwise dropped
    (plan 3.2.4); Rasa clips are short so this rarely triggers.
24. **Voicepacks** average the acoustic/prosodic vectors over the top-200 neutral clips by DNSMOS; the
    Stage-1 style encoder and the Stage-2 predictor encoder are used (kikiri two-checkpoint mode). A
    band-limited speaker (95 % roll-off < 8 kHz) is refused as a reference (`08_voicepack.py` exits).
25. **Export saves the weights without `module.` prefixes** so KModel's strict load path is used and
    verified per module (matched == total); the base Kokoro file uses prefixes but relies on a
    silent `strict=False` fallback that we do not want.
26. **Checkpoint selection score** = 3·CER(Appendix A) + PCER(mixed) + WER(English) + val mel, over the
    light per-epoch evaluation; early stop after 2 epochs without CER improvement.
27. **English WER gate failure handling**: the session records the failure, ships `auto` (Pakistani
    fallback) and notes it in `REPORT.md`; retraining Stage 2 with a larger English share is left to a
    manual `FORCE_PHASE=stage2` with `ENGLISH_REPLAY_HOURS` raised (quota-bound decision for the user).
28. **No INT8 ONNX** (slower for this vocoder, per the Swedish fine-tune); fp16 kept only if parity ≥ 0.99.

## Evaluation details (found by running the harness locally on the base weights)

29. **Bandwidth audit uses the 99 % roll-off** (Kannada audit), not the plan's 95 %: the 95 % roll-off of
    normal 24 kHz speech is only ~4–6 kHz, so a 95 % / 8 kHz rule would flag every speaker as band-limited
    and `08_voicepack.py` would refuse every reference. Threshold: 99 % roll-off < 8 kHz or little energy
    above 7 kHz = band-limited.
30. **Comb/echo detector window is 12.5–30 ms and the gate is relative** to real recordings of the same
    voices (synth ≤ real + 3 MAD units). A 5–30 ms window overlaps the pitch period (4–12 ms), so base
    Kokoro output already "failed" an absolute threshold.
31. **ONNX parity is gated against the exported model path** (`disable_complex=True`, the real-valued
    iSTFT Kokoro ships for ONNX); the correlation against the default complex-STFT KModel path (~0.95 on
    the base weights) and a log-mel correlation are reported as information. fp16 conversion keeps Cast /
    index-arithmetic ops in fp32 (`op_block_list`), otherwise onnxruntime refuses to load the graph.
32. **KModel strict verification tolerates exactly the AdaIN InstanceNorm affine parameters** (`*.norm.weight`,
    `*.norm.bias`): neither hexgrad's own checkpoint nor StyleTTS2 (affine=False) has them.

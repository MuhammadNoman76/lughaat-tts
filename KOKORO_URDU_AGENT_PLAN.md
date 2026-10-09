# Kokoro-82M Urdu (+ English and code-switching) — End-to-End Build Plan for a Coding Agent

> **How to use this file:** Give this whole file to the coding agent (e.g. Claude Code) and fill in Section 1 first. With `COMPUTE_MODE=kaggle` (the default), the agent runs on **any ordinary PC** and all training happens free on Kaggle (Section 14). With `COMPUTE_MODE=gpu`, it runs on a rented 80 GB GPU machine. Everything else — downloading datasets, building the Urdu text frontend and G2P, training, evaluation, export and the Hugging Face upload — is the agent's job.

---

## 0. Mission and definition of done

You are building **Kokoro-Urdu**: a fine-tune of [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) (82M parameters, StyleTTS2 + ISTFTNet, Apache-2.0) that loads with the standard Kokoro inference code (`kokoro.KModel`) and speaks three kinds of input, all in the same Urdu voices:

- **Urdu**, with accurate pronunciation.
- **English**.
- **Urdu–English mixed sentences** (code-switching), e.g. «میں نے اپنا laptop آن کیا اور email چیک کی۔».

Kokoro itself knows nothing about languages; it turns a phoneme string into audio. So "supporting English" means two things:

1. **The frontend** must detect the English words in the input and phonemize them correctly (Section 4.7).
2. **The training data** must keep English sounds in the model, so that fine-tuning on Urdu doesn't erase them. This is the "English replay" in Section 3.1.

Fine-tuning does **not** change the parameter count. The final model stays at about 82M parameters:
- FP32: 82M × 4 bytes = 328 MB (the base `kokoro-v1_0.pth` is 327 MB)
- FP16: 82M × 2 bytes = 164 MB

**Done means all of the following are true:**

1. A Hugging Face model repo `${HF_USERNAME}/kokoro-82m-urdu` exists and contains:
   - `kokoro-urdu-v1_0.pth`: KModel-format weights with exactly the 5 modules `bert`, `bert_encoder`, `predictor`, `text_encoder`, `decoder`.
   - `config.json`: the Kokoro config with the 178-token vocabulary.
   - `voices/uf_rasa.pt` and `voices/um_rasa.pt`: voicepacks, each of shape `[510, 1, 256]` float32.
   - `onnx/kokoro-urdu.onnx` (fp32), plus `onnx/kokoro-urdu-fp16.onnx` if fp16 passes the parity check in Phase 8.
   - `kokoro_urdu/`: a pip-installable package containing the Urdu normalizer, the G2P (lexicon + rules + neural fallback) and the `UrduPipeline` class.
   - `diffusion/`: the full StyleTTS2 Stage-2 checkpoint and `synthesize_diffusion.py`. This is the optional highest-naturalness inference path.
   - `samples/`: WAV files for every test sentence in Appendices A (Urdu), A2 (mixed) and A3 (English), for both voices.
   - `README.md`: a model card with usage, metrics, limitations, licences and attributions.
2. In a **fresh** virtualenv, these two commands work and produce intelligible Urdu:
   - `pip install git+https://huggingface.co/${HF_USERNAME}/kokoro-82m-urdu`
   - `python -c "from kokoro_urdu import UrduPipeline; UrduPipeline().save('پاکستان ایک خوبصورت ملک ہے۔', 'out.wav')"`
   The same call must also work for English text (`'Hello, how are you today?'`) and mixed text (`'آج کی meeting کینسل ہو گئی ہے۔'`).
3. The quality gates in Phase 7 pass for Urdu, English and code-switched test sets, and the numbers are written into the model card.
4. A Gradio Space `${HF_USERNAME}/kokoro-urdu-demo` runs on free CPU hardware.
5. A final `REPORT.md` is uploaded to the model repo. It records the data used, hours, the G2P accuracy, metrics for every epoch, the chosen checkpoint, total GPU hours and cost, and the known limitations.

---

## 1. Inputs from the user (the ONLY things the user must provide)

Before starting, the agent checks that every REQUIRED item below is present. If one is missing, the agent asks the user for it **once**, listing all the missing items together, and then never stops to ask again unless a STOP condition in Section 12 is hit.

```bash
# ===== COMPUTE MODE =====
export COMPUTE_MODE="kaggle"        # kaggle = FREE dual-T4 training on Kaggle (Section 14) | gpu = one rented 80 GB GPU (the original plan)

# ===== REQUIRED =====
export HF_TOKEN="hf_xxx"            # Hugging Face token with WRITE scope (Settings → Access Tokens → "Write")
export HF_USERNAME="your-hf-username"  # account or org that will own the repos

# One-time manual click (cannot be automated: gated datasets need the account holder to accept terms):
#   Open https://huggingface.co/datasets/ai4bharat/Rasa while logged in as HF_USERNAME → "Agree and access".
#   (Optional, only if USE_INDICVOICES_R=1) same for https://huggingface.co/datasets/ai4bharat/indicvoices_r

# --- Required only when COMPUTE_MODE=kaggle (see Section 14.6 for the one-time clicks) ---
export KAGGLE_USERNAME="your-kaggle-username"
export KAGGLE_KEY="xxxxxxxx"         # from kaggle.com → Settings → API → "Create New Token" (the kaggle.json file)
# GitHub CLI logged in (`gh auth login`), so the agent can create the private orchestrator repo and set its secrets.

# --- Required only when COMPUTE_MODE=gpu ---
# Hardware: run this agent ON a Linux machine with ONE 80 GB NVIDIA GPU (A100-80GB or H100-80GB),
#   ≥ 32 vCPU, ≥ 64 GB RAM, ≥ 400 GB free disk. (RunPod / Lambda / Vast / Modal / GCP all fine.)
#   A 40 GB GPU is NOT acceptable for Stage 2 (it silently deadlocks — see Pitfalls).

export MAX_BUDGET_USD="400"         # agent stops and reports if projected GPU cost exceeds this
#   Rough sizing: the Kannada fine-tune used ~11 h of audio for < $100. If cost scales with data:
#   all Rasa Urdu + 10 h English = 52.26 + 10 = 62.26 h → 62.26 / 11 = 5.66× → up to 5.66 × $100 = $566;
#   Rasa capped at 15 h per speaker + 10 h English = 30 + 10 = 40 h → 40 / 11 = 3.64× → up to 3.64 × $100 = $364.
#   The default of 400 covers the capped run; raise it to ~600 to allow training on all Rasa Urdu.
#   The agent measures real epoch times after Stage-1 epoch 1 and decides with actual numbers.
export GPU_HOURLY_USD="1.80"        # what the user pays per GPU-hour, for cost projection

# ===== OPTIONAL =====
export ANTHROPIC_API_KEY=""         # enables the LLM-assisted pronunciation-lexicon step (Phase 2.4). Strongly recommended for accuracy; ~$5–20 of API usage.
export WANDB_API_KEY=""             # optional experiment tracking; otherwise TensorBoard logs are uploaded to HF
export USE_INDICVOICES_R="0"        # 1 = add IndicVoices-R Urdu (multi-speaker) to Stage-1 data
export USE_ASLP_URDUSPEECH="0"      # 1 = add Pakistani-accent ASLP UrduSpeech clips: US-Std (Urdu) + US-CS (real Urdu–English code-switched speech).
                                    #     Best data for code-switching quality, BUT the audio comes from YouTube/PTV, so possible third-party rights.
                                    #     Leave 0 for a commercial release; the plan works without it (Section 4.7).
export ENGLISH_REPLAY_HOURS="10"    # hours of LibriTTS-R English mixed into training, so the model doesn't forget English sounds
export DEFAULT_ENGLISH_ACCENT="auto" # auto | pakistani | native (see Section 4.7)
export REPO_PRIVATE="1"             # 1 = create HF repos as private (user can flip to public later)
```

**Repos the agent creates** (all private unless `REPO_PRIVATE=0`):
- `${HF_USERNAME}/kokoro-82m-urdu`: the final model (model repo)
- `${HF_USERNAME}/kokoro-urdu-work`: processed data, lexicons and per-epoch checkpoints, used as backup so that losing a spot instance loses nothing (dataset repo, always private)
- `${HF_USERNAME}/kokoro-urdu-demo`: the Gradio demo (Space)

---

## 2. Non-negotiable rules for the agent

1. **Never print, log, commit or upload secrets.** Read `HF_TOKEN` and `ANTHROPIC_API_KEY` from the environment only.
2. **Training phonemes and inference phonemes must come from the exact same frontend code.** Ship the frontend inside `kokoro_urdu/`, version it (`FRONTEND_VERSION`), and write that version into `config.json` under a new top-level key, `"urdu_frontend_version"`.
3. **Never change the 178-token vocabulary size or reorder indices.** Every phoneme string that enters the model must contain only symbols from Kokoro's vocabulary. Use the mapping in Section 4.3, and assert this in code.
4. **Back up every Stage-1 and Stage-2 epoch checkpoint** to `${HF_USERNAME}/kokoro-urdu-work` immediately after it is saved.
5. **Score every epoch** (Phase 7), not just the last one. The best epoch is usually not the last.
6. Run long jobs under `tmux` or `nohup`, write logs to files, and poll them. Never block on a foreground training process.
7. **In `kaggle` mode**, after the first Stage-1 session, measure the time per epoch, project the total *weeks* (Section 14.5), and write the projection to `REPORT.md`. There is no dollar budget to enforce. **In `gpu` mode**, after Stage-1 epoch 1, measure the time per epoch and project the total cost:

   `projected_cost = (stage1_epochs × t1 + stage2_epochs × t2) × GPU_HOURLY_USD`

   If `projected_cost > MAX_BUDGET_USD`, first reduce the data (Section 3.4 explains how). If the projection is still over budget after reducing data, STOP and report.
8. Keep a running `REPORT.md` and a `DECISIONS.md` that records each non-obvious choice and the reason for it.

---

## 3. Phase 1 — Data

### 3.1 Sources (already researched; verify each one still exists before use)

| Use | Dataset | What it is | Licence | Notes |
|---|---|---|---|---|
| **PRIMARY training data** | [`ai4bharat/Rasa`](https://huggingface.co/datasets/ai4bharat/Rasa), Urdu subset | Expressive studio TTS corpus. Urdu has **2 speakers: female 26.23 h + male 26.03 h = 52.26 h**, 48 kHz mono | CC-BY-4.0 (gated: accept terms) | Same family of data the Kannada Kokoro fine-tune used. Discover the Urdu config/split name and schema with `datasets.get_dataset_config_names` and inspect the columns (text, audio, speaker/gender, style). |
| Optional Stage-1 robustness | [`ai4bharat/indicvoices_r`](https://huggingface.co/datasets/ai4bharat/indicvoices_r), Urdu subset | Multi-speaker TTS corpus enhanced from ASR data (1,704 h total across 22 languages) | CC-BY-4.0 (gated) | Only if `USE_INDICVOICES_R=1`. Measure Urdu hours and sample rate yourself, and run the bandwidth audit. |
| **English replay** (always on) | [`mythicinfinity/libritts_r`](https://huggingface.co/datasets/mythicinfinity/libritts_r), config `clean`, split `train.clean.100` | Sound-quality-restored LibriTTS: read English, **24 kHz**, multi-speaker | CC-BY-4.0 | Take `ENGLISH_REPLAY_HOURS` (default 10 h): about 20 speakers × 30 min, balanced male/female, clips 2–15 s. Phonemize with **misaki[en]** (American), *not* the Urdu frontend. This keeps English-only sounds such as ɹ θ ð æ ɜ in the model. |
| Optional Pakistani accent + **real code-switching** | [`ASLP-lab/UrduSpeech`](https://huggingface.co/datasets/ASLP-lab/UrduSpeech). Files are under `corpus/US-Std/{short,long}/<category>/` and **`corpus/US-CS/...`** (each category folder has `audio/` and `<category>_final_transcription.jsonl`) | US-Std: 59.2 h of Standard Pakistani Urdu. **US-CS: 89.4 h of Urdu–English code-switched speech, with English words written in Latin script**, e.g. «میں conclude کرتے ہوئے…». 1,000+ speakers. | Card says CC-BY-4.0, **but the audio was sourced from YouTube and PTV broadcasts** | Only if `USE_ASLP_URDUSPEECH=1`. **The transcripts are noisy:** I found lines containing **Devanagari (Hindi-script) fragments** («हो सकती», «घंटों») and Roman Urdu («چobees»). Filtering rules: drop any line with Devanagari (U+0900–U+097F); drop any line whose Latin tokens are not English dictionary words (they are Roman Urdu); keep `Confidence_score ≥ 0.97`; prefer the `news`, `interview`, `podcast` and `proses` categories; then apply the phonetic-CER gate (3.2 step 6) and the bandwidth audit. Cap US-CS at 20 h and US-Std at 15 h. Use this data in **Stage 1 only**, and never as a voice reference. Add a provenance note to the model card, and do not use it for a commercial release without the user's explicit OK. |
| Eval only | [`google/fleurs`](https://huggingface.co/datasets/google/fleurs) `ur_pk` | Read speech, 16 kHz | CC-BY-4.0 | Use its sentences as held-out **test text** and to measure the ASR's own baseline CER. Never use it for training. |
| OOD text for Stage 2 | [`wikimedia/wikipedia`](https://huggingface.co/datasets/wikimedia/wikipedia), Urdu config (e.g. `20231101.ur`) | Urdu Wikipedia text | CC-BY-SA | Take about 10,000 clean sentences of 20–200 characters for `OOD_texts.txt`. |
| G2P gold seed | [WikiPron](https://github.com/CUNY-CL/wikipron) `data/scrape/tsv/urd_arab_broad.tsv` (+ `urd_arab_narrow.tsv`) | **7,709** Urdu word→IPA pairs (broad) and 309 (narrow), scraped from Wiktionary | CC-BY-SA (Wiktionary) | Raw URL: `https://raw.githubusercontent.com/CUNY-CL/wikipron/master/data/scrape/tsv/urd_arab_broad.tsv`. Any lexicon file **derived** from it must be released as CC-BY-SA; the model weights are unaffected. |

Do **not** use 16 kHz-only datasets (for example `codewithdark/urdu-tts`, which is 16 kHz) as training audio for the voice. They contain nothing above 8 kHz and give the voice a dull, metallic timbre.

### 3.2 Processing pipeline (`scripts/01_prepare_data.py`)

1. Download the Urdu subset of Rasa with `datasets` (streaming if needed) into `data/raw/`.
2. Write a manifest of the raw data: speaker, style label, duration and text for every clip.
3. **Resample to 24 kHz mono, 16-bit WAV.** Use `soxr`/`librosa` high-quality resampling, and peak-normalise to −1 dBFS. A wrong sample rate is the classic cause of chipmunk-sounding output.
4. Trim leading and trailing silence, keeping about 100 ms of padding. Keep clips **1.5–20 s** long. If a clip is longer than 20 s, split it at a silence of 300 ms or more, re-align the text by sentence punctuation, and drop the clip if it can't be split cleanly.
5. **Text cleaning**: run the Urdu normalizer from Phase 2.2 on every transcript and store both the raw and the normalized text.
6. **ASR consistency filter**: transcribe every Urdu clip with `openai/whisper-large-v3` (`language="ur"`, `task="transcribe"`, beam size 5). For LibriTTS-R, use `language="en"`; for code-switched clips, see the phonetic CER in Phase 7. Compute the CER between the ASR output and the normalized text, after both are passed through a scoring normalizer (strip diacritics and punctuation, unify ی/ي, ک/ك, ہ/ه, and convert digits to words).
   - First measure the ASR's own error rate on **FLEURS `ur_pk` test**. That number is the ASR floor.
   - Drop a clip if its CER is more than `max(0.25, ASR_floor + 0.15)`.
   - Log how many clips were dropped per speaker.
7. **Bandwidth audit** (port `eval/audit_data.py` from [Kokoro-Indic-Fine-Tuning](https://github.com/sammy4321/Kokoro-Indic-Fine-Tuning)). For each speaker, measure the 95% spectral rolloff and the ratio of energy above 8 kHz. Flag any speaker whose rolloff is below 8 kHz: that audio may still be used for training but **must not** be used as the voicepack reference.
8. Optional quality check: compute DNSMOS or UTMOS per clip and drop clips scoring below 2.8.
9. Phonemize every clip with the frontend from Phase 2, then validate it: every symbol must be in Kokoro's vocabulary and the length must be at most 510 tokens. Drop failures and log them.
10. Write the StyleTTS2 lists, in the format `relative/path.wav|<phonemes>|<speaker_id>`:
    - `train_list.txt` (about 98%) and `val_list.txt` (about 2%, stratified by speaker and by language, at least 100 lines).
    - Speaker IDs: `0` = Rasa female, `1` = Rasa male, `2…` = LibriTTS-R English speakers, then ASLP speakers if they are enabled.
    - **Target mix by hours:** about 75% Urdu (Rasa), about 15% English (LibriTTS-R), and the remaining ~10% code-switched (ASLP US-CS, if enabled; otherwise give that share to Rasa).
    - If the English share drops below 10%, English sounds start to degrade. If it rises above 25%, Urdu quality drops. Record the final mix in `data_report.md`.
    - English clips are phonemized with **misaki[en]** (American). Rasa clips go through the Urdu frontend. ASLP US-CS clips go through the full mixed frontend (Section 4.7) with `english_accent="pakistani"`, because that is how those speakers actually pronounce English.
11. Write `OOD_texts.txt`, one phonemized sentence per line, with none of them taken from the training texts. **It must not be empty**, because Stage 2 silently disables half of adversarial training when it is. Contents:
    - about 8,000 Urdu Wikipedia sentences
    - about 1,000 English sentences, phonemized with misaki
    - about 1,000 **synthetic code-switched sentences**: take Urdu Wikipedia sentences, swap 1–3 nouns or verbs for common English equivalents (e.g. meeting, office, phone, cancel, problem, update), and phonemize the result with the mixed frontend.
12. Pack `audio/` into `audio.tar` and upload it, together with the lists and `OOD_texts.txt`, to `${HF_USERNAME}/kokoro-urdu-work`.
13. Write `data_report.md`: hours per speaker and per style before and after each filter, the bandwidth table, a CER histogram and the phoneme frequency table.

### 3.3 Style labels

Rasa contains several styles, such as neutral and conversational reading as well as emotional styles. Train on **all** of them, because this gives more robust prosody. When building the voicepacks, however, use only the neutral reading styles (e.g. BOOK/WIKI/CONV, or whatever names the schema actually uses) as reference clips.

### 3.4 If the budget forces a cut

Cut the data in this order:
1. Drop the emotional styles from Stage 2 only, keeping them in Stage 1.
2. Cap each Rasa speaker at 15 h.
3. Reduce the English replay to 6 h (never below that).

The Kannada fine-tune reached an ASR round-trip CER of 0.00 with about 11 h of audio, so 2 × 15 h = 30 h of Urdu is still plenty.

---

## 4. Phase 2 — Urdu text frontend (normalizer + G2P)

This phase determines whether the Urdu sounds accurate. Build it as the package `kokoro_urdu/` with unit tests.

### 4.1 What we already know about espeak-ng Urdu (measured; do not re-investigate, but do write regression tests for it)

Using `phonemizer` with the espeak-ng `ur` voice, `with_stress=False`:

| Input | espeak-ng output | Problem |
|---|---|---|
| پاکستان ایک خوبصورت ملک ہے۔ | `paːkɪstaːn eːk xuːbʂuːrat mʊlk hɛ` | ص comes out as **ʂ** (should be s) |
| بھائی گھر میں دھوپ ہے | `bʰaːi ɡʰʌr mẽ dʰoːp hɛ` | OK. Aspirates use ʰ, which is in the vocab. |
| میں نے کتاب پڑھی | `mẽ neː kɪtaːb pʌr.hi` | **ڑھ → `r.h`** (should be ɽʰ); a stray `.`; ڑ collapsed into r |
| قلم خط غلط | `qʌləm xʌt ɣʌlət` | OK |
| وزیرِ اعظم نے کہا | `ʋəzireː aːʐəm neː kʌhaː` | ظ comes out as **ʐ** (should be z); izafat handled only roughly |
| ہم ہاں نہیں | `ham hãː nahiːn` | نہیں should be nəhĩː (final nasal) |
| ۲۰۲۶ میں 25 لوگ | `mẽ pacciːs loːɡ` | **Urdu-script digits are silently dropped.** Latin digits get Hindi-style reading. |

**Conclusion:** espeak-ng `ur` is usable only as a **bootstrap and fallback**. It must not be the main G2P.

A related subtlety: if the G2P makes a *consistent, deterministic* mistake (for example, always writing ʂ for ص), training partly absorbs it, because the model simply learns that symbol → that sound. *Inconsistent* or *ambiguous* output is what destroys accuracy: missing short vowels, wrong nasalisation, and ر/ڑ collapsing into one symbol. That is why the lexicon-first design below matters.

### 4.2 Normalizer (`kokoro_urdu/normalize.py`)

1. Unicode NFC normalisation. Unify Arabic and Persian code points: ي→ی, ك→ک, ه→ہ (and ھ only where it marks aspiration), ى→ی, ۀ→ۂ. Remove tatweel (ـ) and zero-width characters, but keep ZWNJ where it is meaningful.
   - You may reuse `LughaatNLP` (`pip install LughaatNLP`), which provides `normalize_characters`, `normalize_combine_characters` and `replace_digits`. Wrap it; do not depend on its large models.
2. **Numbers → Urdu words.** Python's `num2words` has **no Urdu**, so build this yourself:
   - Urdu numbers 0–99 are irregular and need a full table (e.g. پچیس for 25, پینتالیس for 45). First check whether ICU RBNF covers Urdu spell-out (`PyICU`, `RuleBasedNumberFormat(URBNFRuleSetTag.SPELLOUT, Locale("ur"))`). If it does, use it as the reference and cross-check it against a hand-written table. If it doesn't, write the table yourself and verify every entry 0–99 against Wiktionary's Urdu numerals pages.
   - Then: سو (100), ہزار (1,000), لاکھ (100,000), کروڑ (10,000,000), ارب (1,000,000,000).
   - Handle Urdu-script digits (۰–۹), Latin digits, decimals (اعشاریہ), negatives (منفی), ordinals (پہلا/دوسرا/تیسرا/چوتھا, then the -واں suffix), years (read ۲۰۲۶ as دو ہزار چھبیس), times (۷:۴۵ → سات بج کر پینتالیس منٹ), dates, percentages (فیصد) and currency (روپے, ڈالر).
3. Common abbreviations (ڈاکٹر, etc.), symbols (% & @ + =), URLs and emails: spell them out or drop them.
4. **Latin-script tokens** are *not* handled here. Tag them with `lang="en"` and pass them to the code-switching frontend (Section 4.7). Normalize English numbers, dates and abbreviations inside English spans **in English** (e.g. "3 pm" → "three p m"). Normalize digits inside Urdu spans in Urdu. When a digit sits between scripts, use the language of the majority of the sentence.
5. Split text into sentences on ۔ ؟ ! . ? and on newlines, and chunk to at most 400 phoneme tokens per call. Kokoro rushes long inputs and is weak on very short ones, so merge sentences shorter than 10 tokens with their neighbours.

### 4.3 Target phone set and the Kokoro vocabulary mapping (`kokoro_urdu/phoneset.py`)

Kokoro's vocabulary has 178 index slots, of which **115 are used**; the 63 unused ones hold placeholder characters. I verified every Urdu phone against it. All of them are present **except ɦ, ʱ and ʐ**.

**Canonical Urdu output phone set** (use only these symbols):

- Consonants: `p b t d ʈ ɖ k ɡ q ʔ f v ʋ s z ʃ ʒ x ɣ h m n ŋ l r ɽ j ʧ ʤ`
- Aspiration: a following `ʰ` (pʰ bʰ tʰ dʰ ʈʰ ɖʰ kʰ ɡʰ ʧʰ ʤʰ ɽʰ, also lʰ mʰ nʰ rʰ when spelled with ھ)
- Vowels: `ə ɪ ʊ ɑː iː uː eː oː ɛː ɔː`, plus the short peripheral forms `i u e o` only where the lexicon specifies them
- Nasalisation: combining tilde `̃` (U+0303) after the vowel, e.g. `ɑ̃ː`, `ẽː`, `ĩː`, `ũː`
- Gemination (tashdīd): write the consonant twice, e.g. `bb`
- Word separator: a space. Punctuation: `, . ! ? ;` Stress marks: **none** (`with_stress=False`).

**Mandatory mapping applied as the last step of the G2P:**

| From | To | Why |
|---|---|---|
| `ɦ` | `h` | ɦ is not in the vocab; Urdu has no h/ɦ contrast |
| `ʱ` | `ʰ` | ʱ is not in the vocab; Urdu has no bʰ/bʱ contrast, so `bʰ` is unambiguous |
| `ʐ` | `z` | ʐ is not in the vocab; it is an espeak error for ظ/ض/ذ/ز |
| `ʂ` | `s` | ص/س/ث are all s in Urdu |
| `.` *inside a word* | (delete) | espeak syllable separator |
| `tʃ`, `dʒ` | `ʧ`, `ʤ` | single-token affricates, which Kokoro's vocab and Hindi data use |
| `ɾ` | `r` | one rhotic symbol for ر, to keep the output consistent |
| `g` (U+0067) | `ɡ` (U+0261) | the vocab uses IPA ɡ |
| precomposed nasal vowels (`ĩ` U+0129, `ẽ` U+1EBD, `õ`, `ũ`, `ã`) | **Unicode NFD** → base vowel + U+0303 | the vocab contains only the *combining* tilde (id 17). Precomposed characters are silently dropped by `KModel`, which **filters out unknown symbols without any error**. |

After mapping, assert that every character is in `config.json["vocab"]`. If anything is out of vocabulary, raise an error in tests and log a warning at runtime.

### 4.4 Pronunciation lexicon (`kokoro_urdu/data/lexicon.tsv`)

1. **Word list**: every word type in the normalized Rasa transcripts + the 50,000 most frequent words in Urdu Wikipedia + every WikiPron headword.
2. **Gold seed**: WikiPron `urd_arab_broad.tsv`, mapped into the canonical phone set. Hold out a random 10% of it (`wikipron_test.tsv`) **before** doing anything else. It is used only for evaluating G2P accuracy.
3. **Candidate A**: espeak-ng `ur` + the mapping in 4.3 + rule fixes:
   - ص → s
   - ظ/ض/ذ → z
   - ڑ → ɽ (never r)
   - ڑھ → ɽʰ
   - word-final ں → nasalise the preceding vowel
   - izafat (zer-e-izafat ِ or final ۂ / ئے joining two words) → append `eː` to the first word
   - ع → handled per the lexicon (usually lengthens or colours the vowel; do not insert ʔ by default)
   - final ہ after a consonant → `ɑː` or `ə` per the lexicon
   - و/ی as consonant vs vowel → rely on the lexicon, else the neural model
4. **Candidate B** (only if `ANTHROPIC_API_KEY` is set): ask Claude to give the IPA for each word, in batches of 100.
   - In the prompt, include the strict canonical phone set, 40 few-shot examples from the WikiPron training split, and the instruction to output only phones from the set.
   - Use temperature 0, cache all responses, and cap spending at about $20.
5. **Consonant-skeleton validator** (deterministic, and the key to accuracy). Urdu consonant letters are written explicitly, so build a letter → allowed-consonant map (e.g. ق→{q}, ک→{k}, کھ→{kʰ}, ط/ت→{t}, ث/س/ص→{s}, ح/ہ→{h}, ع→{∅, ʔ}, و→{ʋ or a vowel}, ی→{j or a vowel}, ا→{vowel, ʔ}, ں→{nasalisation}). Reject any candidate whose consonant sequence does not match the letters.
6. **Selection rule:**
   - WikiPron gold first.
   - Otherwise, if A and B agree after normalisation, use them.
   - Otherwise, use the candidate that passes the skeleton check. If both pass, prefer B for vowels but take A's consonants.
   - Otherwise, use A and flag the word in `lexicon_review.tsv`.
7. Write the final lexicon (word → one canonical pronunciation; also keep variants if there are several). Licence the file CC-BY-SA-4.0 because of the WikiPron/Wiktionary seed.

### 4.5 Neural G2P for out-of-vocabulary words (`kokoro_urdu/g2p_model.pt`)

- A small character-to-phone transformer: 3 encoder + 3 decoder layers, d_model 256, about 3–5M parameters. Alternatively, an LSTM seq2seq with attention.
- Train it on the final lexicon, excluding `wikipron_test.tsv`.
- Report the word error rate (WER) and phone error rate (PER) on `wikipron_test.tsv`.
  - Target: **PER ≤ 8%, WER ≤ 25%**.
  - If those aren't met, iterate: add data augmentation, apply the skeleton check to the outputs, and make sure the model uses the diacritics (zer/zabar/pesh) when the input text contains them.
- Runtime order: lexicon lookup → (if not found) neural G2P → skeleton check → (if the check fails) espeak + rules.

### 4.6 Frontend tests (`tests/test_frontend.py`)

- Unit tests for numbers, dates, times and currency, in both Urdu and English spans.
- Every gold word in Appendix B must map exactly to its expected phones.
- Every sentence in Appendices A and A2 must produce only in-vocabulary phones.
- Language tagging of every token in Appendix A2 must match the expected tags.
- Deterministic output: the same input always gives the same phonemes.

### 4.7 English and Urdu–English code-switching (`kokoro_urdu/codeswitch.py`)

**Step 1 — Tag each token's language.**

| Token | Tag | Handled by |
|---|---|---|
| Arabic-script word (including English loanwords already written in Urdu script, e.g. کمپیوٹر, ڈاکٹر) | `ur` | Urdu lexicon / neural G2P (Sections 4.4–4.5). These are ordinary Urdu words. |
| Latin-script word that is in the English dictionary (misaki's gold/silver lexicon, or CMUdict as a backup) | `en` | English G2P (step 2) |
| Latin-script word that is *not* an English word | `roman_ur?` | **Roman Urdu is out of scope for v1.** Spell it as English (step 2) and log it. Add a `RomanUrduHook` interface so transliteration to Urdu script can be plugged in later. |
| Acronyms (USB, NADRA, PTI) | `en_acronym` | Spell letter by letter with English letter names, e.g. USB → "yoo es bee" |
| Digits | language of the surrounding span | Section 4.2 |

**Step 2 — English G2P.** Use **`misaki[en]`** (`misaki.en.G2P(trf=False, british=False)`), Kokoro's own English G2P, with espeak `en-us` as its fallback for unknown words. Its output symbols are all already in Kokoro's vocabulary: the 22 IPA consonants plus ɹ θ ð, the vowels ə i u ɑ ɔ ɛ ɜ ɪ ʊ ʌ æ ᵻ ᵊ, the special diphthong letters **A=eɪ, I=aɪ, W=aʊ, Y=ɔɪ, O=oʊ**, flap ɾ, and the stress marks ˈ ˌ (see misaki's `EN_PHONES.md`).

**Step 3 — Choose the accent** with `english_accent`:

- **`"pakistani"`**: English words are re-mapped onto the Urdu phone set. This is how Urdu speakers actually say English words inside Urdu sentences. It is also the **most robust** option, because every resulting sound is one the model has heard thousands of times in the Urdu data.
- **`"native"`**: keep misaki's American phonemes unchanged. This works only because of the English replay data.
- **`"auto"`** (default): use `pakistani` for English words inside a sentence that is mostly Urdu, and `native` for sentences where at least 80% of the tokens are English.

**Pakistani-English mapping** (applied to misaki output). This is a starting table; tune it in Phase 7 by phonetic CER and the listening page.

| misaki | → | Note |
|---|---|---|
| `ˈ` `ˌ` | (delete) | the Urdu data is trained without stress marks |
| `t` / `d` | `ʈ` / `ɖ` | English alveolar t/d are realised as retroflex by Urdu speakers. Make this switchable with `retroflex_td=True`. |
| `θ` / `ð` | `tʰ` / `d` | thin → tʰɪn, this → dɪs |
| `ɹ` | `r` | |
| `ɾ` | `ʈ` | water → ʋɔːʈər |
| `w`, `v` | `ʋ` | the v/w merger |
| `æ` | `ɛː` | |
| `ɛ` | `ɛ` | |
| `ʌ`, `ə`, `ᵊ`, `ᵻ` | `ə` | |
| `ɜ` (`ɜɹ`) | `ə` (`ər`) | |
| `ɑ` | `ɑː` | |
| `ɔ` | `ɔː` | |
| `i` | `iː` | |
| `u` | `uː` | |
| `ɪ`, `ʊ` | `ɪ`, `ʊ` | |
| `A` | `eː` | |
| `O` | `oː` | |
| `I` | `ɑɪ` | |
| `W` | `ɑʊ` | |
| `Y` | `ɔɪ` | |
| `ʤ`, `ʧ`, `ʃ`, `ʒ`, `z`, `s`, `ŋ`, `j` | unchanged | |

**Step 4 — Join the spans.** Keep the spaces and punctuation between words. Apply NFD, then assert that every symbol is in the vocabulary.

**Why this works even without code-switched training audio:** in `pakistani` mode, an English word becomes a sequence of Urdu sounds, which the fine-tuned model already produces fluently in the Urdu voice. Real code-switched audio (ASLP US-CS) improves the *prosody* at language boundaries, but it isn't required for intelligibility.

**Tests:** add 30 English words with expected Pakistani-mode outputs. A few anchors, using an American misaki base:
- laptop → `lɛːpʈɑːp`
- meeting → `miːʈɪŋ`
- email → `iːmeːl`
- thank you → `tʰɛːŋk juː`
- problem → `prɑːbləm`
- cancel → `kɛːnsəl`
- update → `əpɖeːʈ`

Pakistani English is closer to British English for "o" words (laptop → lɛpʈɔp). So also try **misaki British (`british=True`) as the base for Pakistani mode**, with `ɒ → ɔ` and the British length mark `ː` kept, but **re-insert r** after vowels where the spelling has "r", because Pakistani English is rhotic. Pick whichever base gives the lower phonetic CER and the better listening results. Regenerate the anchors from the chosen base rather than special-casing individual words, and record the choice in `DECISIONS.md`.

---

## 5. Phase 3 — Zero-training baseline (about 30 minutes; for comparison only)

1. Install Kokoro: `pip install "kokoro>=0.9.2" soundfile`, plus `apt-get install espeak-ng`.
2. Load `KModel(repo_id='hexgrad/Kokoro-82M')` and the Hindi voices `hf_alpha`, `hf_beta`, `hm_omega` and `hm_psi`.
3. For each sentence in Appendix A: run the new Urdu frontend to get phonemes, then call `KPipeline(lang_code='h', repo_id='hexgrad/Kokoro-82M', model=kmodel).generate_from_tokens(tokens=phonemes, voice='hm_psi')`. The `voice` argument takes a voice name such as `hm_psi`; repeat for the other Hindi voices.
4. Also synthesize Appendix A2 (mixed) and A3 (English) with **base** Kokoro: the mixed sentences with a Hindi voice, and the English sentences with `af_heart`. Compute English WER for the English sentences.
   - This English WER is the "no-forgetting" reference: after fine-tuning, English WER in the Urdu voices should stay close to it (Phase 7 gate).
5. Save the outputs to `baseline_samples/` and compute ASR CER for them. This is the bar the fine-tune must clearly beat.

---

## 6. Phase 4 — Training environment

Base it on the German recipe [semidark/kikiri-tts](https://github.com/semidark/kikiri-tts), plus the patches from the Kannada recipe [sammy4321/Kokoro-Indic-Fine-Tuning](https://github.com/sammy4321/Kokoro-Indic-Fine-Tuning). Read both repos' `README.md`, `docs/TRAINING_GUIDE.md`, `docs/TROUBLESHOOTING.md`, `docs/JOURNEY.md` and `training/run_kannada.py` before starting.

1. Install system packages: `apt-get install -y espeak-ng libespeak-ng1 libsndfile1 tmux`.
2. Clone the training code:
   ```bash
   git clone --recurse-submodules https://github.com/semidark/kikiri-tts
   ```
   The `StyleTTS2/` and `kokoro/` submodules must not be empty.
3. Create a Python 3.12 venv with `uv`. Pin the versions the Kannada run proved:
   ```
   torch==2.4.1 torchaudio==2.4.1 torchvision==0.19.1 'numpy<2' accelerate==0.34.2
   transformers==4.49.0 phonemizer==3.3.0 munch einops einops-exts pyyaml soundfile librosa
   scipy tqdm loguru tensorboard Cython nltk pydub matplotlib==3.7.5 pandas huggingface_hub
   'git+https://github.com/resemble-ai/monotonic_align.git'
   ```
   Install a CUDA build of torch that matches the driver.
4. **Symbol map:** copy `kikiri-tts/training/kokoro_symbols.py` into `StyleTTS2/` and make sure `text_utils.py` and `meldataset.py` import from it. Check it:
   ```python
   assert len(symbols) == 178
   assert dicts['ʰ'] == 162
   assert dicts['ɽ'] == 129
   assert dicts['q'] == 59
   assert dicts['̃'] == 17
   ```
   Without this, training appears to run but every token embedding is silently wrong.
5. **Convert the base weights:** download `kokoro-v1_0.pth` from `hexgrad/Kokoro-82M`. Strip the `module.` prefixes and save `{'net': {bert, bert_encoder, predictor, text_encoder, decoder}}` as `kokoro_base.pth` (the code is in kikiri's `TRAINING_GUIDE.md` §3). Set `load_only_params: true`.
6. **Utility models**: make sure `Utils/JDC/bst.t7`, `Utils/ASR/config.yml`, `Utils/ASR/epoch_00080.pth` and `Utils/PLBERT/` exist in `StyleTTS2/`, fetching them from the upstream `yl4579/StyleTTS2` if they are missing.
   - Pre-download `microsoft/wavlm-base-plus` to a local directory and point the `slm.model` config at it, so training never fetches it mid-run.
7. **Apply the Kannada trainer patches** (port them exactly from `run_kannada.py`, step 4):
   - `RESUME-FIX-A`: strip `module.` prefixes when loading a checkpoint, and print matched/total tensors per module.
   - `RESUME-FIX-B`: run `predictor_encoder = deepcopy(style_encoder)` only when `start_epoch == 0`.
   - `FT-LR`: scale the learning rates after the optimizer state is loaded.
   - `SCHED-SAVE`: save scheduler state in every checkpoint.
   - Verbose exceptions (replace the bare `except: continue`), print when `slmadv` returns None, gradient clipping at 100, and NaN-skip.
8. Build `monotonic_align`: `cd StyleTTS2/monotonic_align && python setup.py build_ext --inplace`.
9. Generate the config programmatically, using `build_config()` from `Kokoro-Indic-Fine-Tuning/training/kokoro_config.py`. Keep all critical keys at the **top level** of the YAML; anything nested under `training:` is silently ignored.

### 6.1 Smoke test (must pass before any long run)

1. The model loads. Missing keys for diffusion/SLM are expected; size-mismatch errors are not.
2. Run 2 training steps with finite losses. Expected first-step ranges: Mel 0.8–1.5, Gen 3–6, Disc 4–6, Mono 0.01–0.1, S2S 1–6, SLM 1–3.
3. Any NaN in the Mel loss means the symbol map is wrong. Fix that before doing anything else.

---

## 7. Phase 5 — Stage 1 (acoustic: learning to pronounce Urdu)

Config: `batch_size` 8–12 (whatever fits; start at 12 on 80 GB), `max_len` 400, `epochs_1st` 10, `save_freq` 1, `lr` 1e-4, `bert_lr` 1e-5, `ft_lr` 1e-4, `TMA_epoch` 0, `lambda_slm` as in the kikiri config, `multispeaker: false`, data from `train_list.txt`.

```bash
cd StyleTTS2 && tmux new -d -s s1 "accelerate launch train_first.py --config_path ../configs/config_urdu_s1.yml 2>&1 | tee ../logs/stage1.log"
```

- **Healthy run:** the Mel loss falls from about 0.8 towards about 0.25–0.3 over roughly 10 epochs, and the Mono loss stays below 0.05. If Mel is still above 0.4 after 4 epochs, check the data, the phonemes and the learning rate.
- After every epoch: back the checkpoint up to HF, extract a quick voicepack, then synthesize 5 Urdu, 3 mixed and 3 English test sentences. Compute Urdu CER, PCER and English WER (Phase 7). This catches failures, including English forgetting, early.
- Select the best Stage-1 epoch by validation Mel loss plus CER, and copy it to `first_stage.pth`.

---

## 8. Phase 6 — Stage 2 (prosody, adversarial training, diffusion; where naturalness comes from)

Use the **winning Kannada recipe**:

```
batch_size 4    max_len 400     lr 5e-5      joint_epoch 1
diff_epoch 1    lambda_diff 1.0 lambda_sty 1.0  lambda_slm 0.2
epochs_2nd 6–8  save_freq 1     slmadv_params: min_len 100, max_len 200
second_stage_load_pretrained: false   # load from first_stage.pth
```

- **80 GB GPU required.** At `joint_epoch`, the discriminators and the WavLM adversary switch on together. On 40 GB this hangs silently instead of throwing an out-of-memory error.
- **Key indicator:** the Stage-2 Mel loss should start around 0.4, not 7–8. A start of 7–8 means the weights did not load (see RESUME-FIX-A).
- Watch for "ALIGNER-EXC", "SLMADV-NONE" and "STEP-EXC" in the log. If adversarial steps are being skipped every time, stop and fix that before continuing.
- After every epoch: back it up to HF and run the full Phase 7 evaluation.
- If the adversarial/SLM epoch adds a comb or echo artifact (seen in the Swedish fine-tune), prefer the best checkpoint from **before** the adversarial phase, and record this in `DECISIONS.md`.

---

## 9. Phase 7 — Evaluation and checkpoint selection

Write `eval/evaluate.py` and run it for every candidate checkpoint.

1. **Test sets:**
   - `val_list` texts (in-domain)
   - 300 FLEURS `ur_pk` test sentences (out-of-domain)
   - Appendix A (hard Urdu cases: numbers, ھ, ں, ڑ, ق/خ/غ, izafat, Arabic-origin words)
   - **Appendix A2** (code-switched) plus 200 synthetic code-switched sentences, made the same way as the OOD ones but not overlapping them
   - **Appendix A3** (English-only) plus 200 LibriTTS-R `test.clean` sentences
2. **Intelligibility:** always report the ASR floor (its error rate on *real* audio of the same kind) next to every result.
   - **Urdu:** ASR round-trip CER/WER with `openai/whisper-large-v3` (`language="ur"`), using the same scoring normalizer as in 3.2.
   - **English:** WER with whisper-large-v3 (`language="en"`) and the Whisper English text normalizer. Floor: real LibriTTS-R `test.clean` audio.
   - **Code-switched: phonetic CER (PCER).** Whisper writes English words in mixed speech inconsistently, sometimes in Latin script and sometimes transliterated into Urdu script, so plain CER is meaningless. Instead:
     1. Transcribe with whisper-large-v3 using `language="ur"`.
     2. Pass both the reference text and the ASR text through **our own frontend** (Section 4.7) to get phonemes.
     3. Collapse both phoneme strings to coarse classes: t/ʈ, d/ɖ, v/w/ʋ, e/ɛ, o/ɔ and ə/ʌ are merged, and length marks, aspiration and stress are ignored.
     4. Compute the phone error rate between the two collapsed strings.

     PCER does not care which script the ASR chose. Report PCER separately for the Urdu part and the English part of each sentence.
3. **Naturalness:** UTMOS and/or DNSMOS. DNSMOS evaluates at 16 kHz, so it cannot hear high-frequency problems; also report the spectral rolloff of the generated audio.
4. **Speaker similarity** to the reference voice: ECAPA cosine similarity, using `speechbrain/spkrec-ecapa-voxceleb`.
5. **Speed:** real-time factor on GPU and on CPU (4 threads).

**Quality gates** (all must pass for the release candidate):
- Out-of-domain CER ≤ ASR_floor + 0.03, **and** clearly better than the Phase-3 Hindi-voice baseline.
- Appendix-A CER ≤ ASR_floor + 0.06.
- **Code-switched PCER ≤ 0.10** overall, and English words inside mixed sentences ≤ 0.15. Do this for `english_accent="pakistani"`, and also report `"native"` and `"auto"`.
- **English-only WER (Urdu voices, `native` accent) ≤ base-Kokoro `af_heart` WER + 0.05.** This is the no-forgetting check. If it fails, raise the English share (up to 20%) and retrain Stage 2. If it still fails after that, ship `auto` with `pakistani` as the fallback for English, and document this.
- No sentence may lose an English word entirely. Check this with the word-level alignment from PCER: every reference English word must align to some output.
- UTMOS ≥ 3.5, or within 0.3 of the real Rasa recordings.
- No clipping, no silence longer than 1.5 s, no audible comb/echo. Check the comb with a cepstral-peak detector between 5 and 30 ms.

Generate `eval/listening_test.html`: for each test sentence, real reference audio (where one exists) alongside each top-3 checkpoint, plus a simple rating form. Upload it to the work repo. The user *may* listen to it, but selection must not wait on that: pick the checkpoint automatically by the gates, then the highest UTMOS.

---

## 10. Phase 8 — Export (Kokoro-compatible) + voicepacks + ONNX

1. **KModel weights:** from the chosen Stage-2 checkpoint, take `bert`, `bert_encoder`, `predictor`, `text_encoder` and `decoder`. Strip `module.`. Use the new `torch.nn.utils.parametrizations.weight_norm` key layout (see kikiri `ARCHITECTURE.md`). Save the result with `torch.save` as `kokoro-urdu-v1_0.pth`.
2. **Verify** it loads strictly: `KModel(repo_id=<local>, config='config.json', model='kokoro-urdu-v1_0.pth')`. Then for **every** module, assert `matched == total` tensors.
   - Note: `KModel` only auto-downloads weights for `hexgrad/Kokoro-82M`. For this repo, the `model=` path must always be passed explicitly; `UrduPipeline` does this.
3. **config.json:** copy the base config unchanged (same 178-token vocab), and add `"urdu_frontend_version"` and `"languages": ["ur", "en"]`.
4. **Voicepacks:** use kikiri's `scripts/extract_voicepack.py`, with the Stage-1 style encoder and the Stage-2 predictor encoder (the two-checkpoint mode). Use only that speaker's neutral-style, full-bandwidth clips (top 200 by UTMOS).
   - Output: `voices/uf_rasa.pt` (female) and `voices/um_rasa.pt` (male), each `[510,1,256]` float32.
   - Optional experiment: build length-dependent voicepack rows. For each token length L, average style vectors from reference clips whose phoneme length is close to L. Keep this only if it wins the A/B on UTMOS and CER.
5. **ONNX:** use `kokoro/examples/export.py` (`KModelForONNX(KModel(..., disable_complex=True))`, opset 17).
   - Check parity: compare PyTorch and onnxruntime outputs on 20 sentences; the waveform correlation must be at least 0.99.
   - Export fp16 too, and keep it only if it passes the same parity check.
   - Do **not** ship dynamic INT8. The Swedish Kokoro fine-tune found that it *slows down* this convolution-heavy vocoder. Note it in the card as an experiment that didn't help.
6. **Diffusion path** (optional, highest naturalness): upload the full Stage-2 checkpoint, the config and a port of the Kannada `inference/synthesize.py` (diffusion style sampling with alpha 0.3, beta 0.7, 10 steps, embedding scale 1.5, best-of-3 by predicted MOS) to `diffusion/`.

---

## 11. Phase 9 — Package, upload, verify

### 11.1 `kokoro_urdu` package (lives in the model repo root, with a `pyproject.toml`)

```python
from kokoro_urdu import UrduPipeline
tts = UrduPipeline(voice="uf_rasa", device="cpu")   # downloads weights from HF on first use
audio = tts("آج موسم بہت اچھا ہے۔", speed=1.0)      # np.float32 @ 24 kHz
tts.save("آج موسم بہت اچھا ہے۔", "out.wav")
tts.save("آج کی meeting کینسل ہو گئی ہے، please email check کریں۔", "mixed.wav")   # Urdu + English
tts.save("Good morning, this is an English sentence.", "en.wav", english_accent="native")
phonemes = tts.phonemize("آج کی meeting کینسل ہو گئی ہے۔")   # for debugging; shows per-token language tags too
```

- `english_accent` is `"auto"` (default), `"pakistani"` or `"native"`; `retroflex_td` defaults to `True`.
- Internally: normalizer → sentence chunker → token language tagging → Urdu G2P / misaki English G2P + accent mapping → vocabulary mapping → `KModel(repo_id, config, model=path)` → `voice[len(tokens)-1]` → audio, concatenating chunks with 120 ms of silence.
- Dependencies: `kokoro>=0.9.2`, `misaki[en]`, `torch`, `numpy`, `soundfile`, `huggingface_hub`. espeak-ng is only an optional fallback; Urdu must work without it, using the lexicon plus the neural G2P. For English, misaki's own lexicon covers common words, and espeak is used only for rare words it doesn't know.
- Include `onnx_infer.py`, which runs with `onnxruntime` + `numpy` only.

### 11.2 Upload (`scripts/99_upload.py`, using `huggingface_hub.HfApi`)

1. `create_repo(f"{HF_USERNAME}/kokoro-82m-urdu", repo_type="model", private=REPO_PRIVATE, exist_ok=True)`
2. Upload everything listed in Section 0. Use `upload_large_folder` for any large files.
3. **Model card** (`README.md`) must contain:
   - The YAML header: `license: apache-2.0`, `language: [ur, en]`, `base_model: hexgrad/Kokoro-82M`, `pipeline_tag: text-to-speech`, `tags: [kokoro, styletts2, urdu, tts]`.
   - Usage (pip, Python, ONNX).
   - Sample audio embeds.
   - The metrics table with the ASR floor.
   - Training data and hours.
   - **Attributions:**
     - AI4Bharat Rasa (CC-BY-4.0): cite it.
     - LibriTTS-R (CC-BY-4.0): cite it.
     - misaki (Apache-2.0)
     - Kokoro-82M (Apache-2.0)
     - StyleTTS2 (MIT)
     - kikiri-tts and Kokoro-Indic-Fine-Tuning recipes (Apache-2.0)
     - WikiPron/Wiktionary: the lexicon file is CC-BY-SA-4.0.
   - Limitations: the accent is that of the Rasa Urdu speakers; known G2P weak spots; behaviour on Roman Urdu input; code-switching quality.
   - A prohibition on impersonation and voice misuse.
4. **Space:** a Gradio app with a text box, voice dropdown and speed slider, a "show phonemes" toggle and a CPU-basic SDK. It loads the model from the model repo; if that repo is private, set `HF_TOKEN` as a Space secret.
5. **Final verification:**
   - In a brand-new venv on CPU, `pip install git+https://huggingface.co/${HF_USERNAME}/kokoro-82m-urdu`, synthesize all of Appendices A, A2 and A3, and recompute CER, PCER and WER. It must match the Phase-7 numbers within ±0.01.
   - Confirm the Space builds and returns audio.
6. Upload `REPORT.md`. Print a final summary with the repo URLs, the metrics and the total cost.

---

## 12. STOP conditions (the only times to interrupt the user)

- A required input from Section 1 is missing, or the HF token lacks write scope.
- Access to the gated Rasa dataset is denied, meaning the user hasn't clicked "Agree". Give the exact URL.
- The projected cost exceeds `MAX_BUDGET_USD` even after the data reductions in 3.4.
- (`gpu` mode only) The GPU has less than 80 GB of memory. In `kaggle` mode, follow Section 14.3 instead.
- (`kaggle` mode) The account cannot enable internet in kernels (phone not verified), or the `HF_TOKEN` Kaggle Secret is not attached to the kernel (Section 14.6).
- The quality gates still fail after two extra Stage-2 attempts. Report what was tried and the best result.

---

## 13. Known pitfalls (each one has cost someone days)

1. **Stage-2 deadlocks silently on 40 GB GPUs.** Use 80 GB.
2. A **bare `except: continue`** in the trainer can swallow every adversarial step. Patch in verbose logging first.
3. **`max_len` 200** (about 2.5 s crops) quietly destroys prosody learning. Use 400.
4. **Untrained diffusion sampler = robotic prosody.** Use `diff_epoch 1`, `lambda_diff 1.0`, `lambda_sty 1.0`.
5. **Broken resume:** checkpoints saved with DataParallel `module.` prefixes, combined with `strict=False` loading, load *zero* tensors and fail silently. Also, `predictor_encoder` gets clobbered on resume. The RESUME-FIX patches handle both.
6. **Metallic timbre = band-limited audio** (16 kHz audio shipped as 24/48 kHz). Audit bandwidth, and never use band-limited audio as the voicepack reference. Note that the Kannada team found such audio could still *help* pronunciation when used as training data.
7. **An empty `OOD_texts.txt`** silently disables half of the SLM adversarial training.
8. **Wrong symbol map:** training appears to run but the embeddings are wrong. Watch for NaN Mel loss or Mel stuck high.
9. Config keys nested under `training:` are ignored by `train_first.py`. Put them at the top level.
10. **PyTorch 2.6+** changed the default of `torch.load` to `weights_only=True`. Pass `weights_only=False` for training checkpoints.
11. **PL-BERT maximum is 512 positions.** Filter samples above 510 tokens.
12. Mixing `weight_norm` APIs between training and inference breaks loading; use the parametrizations API.
13. On some cloud VMs, HF downloads stall on IPv6 or Xet. The Kannada driver forces IPv4 and uses chunked range downloads. Port that helper if downloads hang.
14. **espeak-ng drops Urdu-script digits silently.** Always normalize numbers *before* G2P.
15. **`KModel.forward` silently filters out phonemes that aren't in its vocab** (`self.vocab.get(p)` → None → dropped). An out-of-vocabulary symbol therefore never raises an error; that sound just disappears from the speech. Assert vocab membership in the frontend, and apply NFD to the phoneme string.
16. **Catastrophic forgetting:** fine-tuning on Urdu alone erases English-only sounds (ɹ θ ð æ ɜ). Never train without the English replay, and check English WER after every epoch, not just at the end.
17. **Symbol collisions between misaki and the Urdu phone set:** misaki uses capital `A I O W Y` as diphthongs and plain `i u` for long vowels. The Urdu set uses `iː uː eː oː`. Never run Urdu text through misaki, or English text through the Urdu G2P. Keep the spans separate until the final join.

---

## 14. Kaggle free mode (`COMPUTE_MODE=kaggle`)

The whole pipeline runs for free on Kaggle's **GPU T4 ×2** (two 16 GB T4s). It resumes automatically after every 12-hour session limit and every weekly quota reset, with no PC needing to stay on. Phases 1–9 keep the same *logic*; this section overrides **where** they run and **how big** they are.

### 14.1 Facts this design relies on (checked in the Kaggle API docs, Sep 2026)

- `kaggle kernels push -p <dir> --accelerator NvidiaTeslaT4 -t <seconds>` uploads a script and **runs it in the background**; no browser is needed. `NvidiaTeslaT4` = GPU T4 ×2, which is Kaggle's default GPU. The P100 is retired.
- `kernel-metadata.json` supports `enable_gpu`, `enable_internet`, `machine_shape`, `dataset_sources` and `kernel_sources`.
- **`kaggle quota`** prints the used, remaining and total weekly GPU hours and **`refreshAt`**, the time the quota resets. The orchestrator uses this to know when to resume.
- `kaggle kernels status` and `kaggle kernels output` let the orchestrator poll a run and fetch its logs.
- **Kaggle Secrets have no CLI support.** The user must add `HF_TOKEN` once in the web UI (Section 14.6). Code reads it with `from kaggle_secrets import UserSecretsClient; UserSecretsClient().get_secret("HF_TOKEN")`.
- The documented accelerator list also includes `NvidiaL4` (24 GB), but some accelerators are restricted to certain users. **Probe it once** (Section 14.3); if the account can use it, prefer it for Stage 2.
- Sessions last at most about 12 h. The weekly GPU quota is shown by `kaggle quota`; it is often reported as about 30 h, but always read the real number.
- Never store the HF token in a Kaggle Dataset or in code.
- **One Kaggle account only.** Using multiple accounts to get more quota violates Kaggle's rules.

### 14.2 Architecture

```
GitHub Actions (free cron, every 60 min)          Hugging Face (state + checkpoints)
  orchestrator.py ──kaggle CLI──► Kaggle kernel  ◄──► ${HF_USERNAME}/kokoro-urdu-work
        │  reads state.json / quota                 (state.json, ckpts/, logs/)
        └─ decides: wait | push next session | export | finished
Kaggle private Dataset: ${KAGGLE_USERNAME}/kokoro-urdu-data  (audio.tar + lists, mounted read-only at /kaggle/input)
```

1. **The local Claude Code session** builds the frontend and the data scripts, creates the repos, does the first pushes, then hands control to the orchestrator. After that, the user's PC can be off.
2. **Data lives in a private Kaggle Dataset** (`kaggle datasets create/version -p data/ --dir-mode tar`). It is mounted at `/kaggle/input/...` in every session, so there is no re-download each session.
   - Size check for the free-mode data (Section 14.4): 15 h of audio × 3,600 s × 24,000 samples × 2 bytes = 15 × 3,600 = 54,000 s; 54,000 × 24,000 = 1,296,000,000 samples; × 2 bytes = **about 2.6 GB**. That fits easily.
3. **One idempotent kernel script, `kaggle/train_session.py`** (kernel type `script`, GPU, internet on, data dataset attached). On every start it:
   1. Records `t0 = time.time()`, reads `HF_TOKEN` from Kaggle Secrets, and installs pinned dependencies. Cache the wheels in a second Kaggle Dataset to save about 5–10 minutes per session.
   2. Downloads `state.json` and the latest checkpoint from the HF work repo. `state.json` holds `{phase, stage, epoch, shard, best_ckpt, done}`.
   3. Runs **the next unit of work** for that phase: data prep, Stage-1 epochs, Stage-2 epochs, eval or export.
   4. **Time guard:** it never starts a unit it cannot finish before `t0 + 11 h 15 min`. Training saves at shard boundaries (see "shards" below), uploads the checkpoint plus the updated `state.json` to HF, writes a 1-line summary to `/kaggle/working/summary.txt`, and exits cleanly *before* Kaggle kills the session.
   5. On any exception, it uploads the traceback to `logs/` on HF and sets `state.json["last_error"]`.
4. **"Shards" instead of long epochs.** Split `train_list.txt` into K shards so that **one shard takes at most about 60 minutes** on T4 ×2, and save and upload after each shard. A killed session then loses at most one hour.
   - Measure the real time per shard in the first session and pick K: K = ceil(time per full pass ÷ 60 min).
   - One "epoch" in the recipe means K shards. The learning-rate scheduler steps per *iteration*, so the scheduler state must be saved and restored (the `SCHED-SAVE` patch).
5. **Orchestrator: `orchestrator.py`, run by a GitHub Actions workflow** in a private repo `${GITHUB_USER}/kokoro-urdu-orchestrator`.
   - Schedule: `cron: "7 * * * *"` (hourly) plus `workflow_dispatch`.
   - Repo secrets: `KAGGLE_USERNAME`, `KAGGLE_KEY`, `HF_TOKEN`. Set them with `gh secret set`.
   - Logic on each run:
     ```
     status = kaggle kernels status <user>/kokoro-urdu-train
     if status in {queued, running}: exit                      # nothing to do
     state = read state.json from HF
     if state.done: exit                                      # training finished, notify once
     if state.last_error repeated 3× on the same unit: open a GitHub issue with the log and exit (needs a human)
     q = kaggle quota  (GPU remaining, refreshAt)
     if q.remaining < 1.5 h: exit                              # wait for refreshAt; hourly cron re-checks
     timeout = min(11.5 h, q.remaining − 0.25 h) in seconds
     kaggle kernels push -p kaggle/ --accelerator NvidiaTeslaT4 -t <timeout>
     append a line to progress.md on HF (time, phase, epoch/shard, metrics)
     ```
   - GitHub Actions free minutes: about 1 min per run × 24 runs a day × 30 days = **about 720 min a month**. That is under the free 2,000 min a month for private repos.
   - GitHub may disable scheduled workflows in repos with no activity for 60 days. The orchestrator commits `progress.md` weekly to keep the repo active.
6. **Notifications:** after every session, the orchestrator updates a pinned GitHub issue, "Training progress", with the phase, epoch, latest CER/PCER and the weeks remaining. The user can watch it from the GitHub mobile app.

### 14.3 Fitting the recipe into 2 × 16 GB T4

The two GPUs **do not pool memory**: each one holds a full copy of the model, and only the batch is split between them. T4s have **no bf16**, so use **fp16 autocast + GradScaler**. The NaN-skip patch matters even more here.

- **Stage 1:** `accelerate launch --multi_gpu --num_processes 2 --mixed_precision fp16 train_first.py`. Start at batch 4 per GPU (8 total) with `max_len` 400; the German recipe runs Stage 1 at batch 4 on 12 GB. If it runs out of memory, use `max_len` 300.
- **Stage 2:** the full recipe needs 80 GB, so **probe these variants in order with a 30-step memory test each, and keep the first that fits:**
  1. **On `NvidiaL4` (24 GB), if the account can use it:** the full recipe at batch 2, `max_len` 300.
  2. **T4 ×2, full recipe** (GAN + WavLM SLM adversary + diffusion) at batch 1 per GPU, `max_len` 300, gradient checkpointing on the predictor, text encoder and decoder, with the WavLM adversary in fp16 and frozen.
  3. **T4 ×2, "Stage-2-lite A":** keep the MPD/MSD discriminators and diffusion (`diff_epoch 1`, `lambda_diff 1.0`, `lambda_sty 1.0`), but **turn the WavLM SLM adversary off**: `lambda_slm 0`, skip the `slmadv` block, and don't load WavLM at all. Batch 2 per GPU, `max_len` 300–400.
  4. **T4 ×2, "Stage-2-lite B":** set `joint_epoch` 999, so there are no GANs; train diffusion plus duration, F0 and mel only. This last resort is the most robotic-sounding, so record in `REPORT.md` that quality is limited by hardware.

  Record the chosen variant and the measured peak memory per GPU in `DECISIONS.md`.
- `train_second.py` uses `DataParallel` over the visible GPUs. GPU 0 carries extra load, so if GPU 0 runs out of memory first, lower the batch size before lowering `max_len`.
- **Checkpoint size:** a full StyleTTS2 checkpoint is about 2 GB, because it includes the optimizer state. Upload it to HF, keep only the last 2 plus the best one in the work repo, and delete older ones.

### 14.4 Free-mode data size (default)

The free quota is small, so `kaggle` mode overrides Section 3.4 with these defaults:
- **Urdu (Rasa):** 6 h per speaker = **12 h**, using the neutral styles first.
- **English replay:** **3 h**. This is the minimum that still prevents forgetting; English WER is checked every epoch.
- Total = 12 + 3 = **15 h**. For comparison, the Kannada model used about 11 h.
- Epochs: Stage 1 = 8, Stage 2 = 5. Stop early when validation CER stops improving for 2 epochs.
- **Data prep runs on Kaggle as well** (session type `prep`). Whisper ASR filtering uses `openai/whisper-large-v3-turbo` on the T4s to save quota, and the bandwidth audit runs on CPU.

### 14.5 How long it will take (the agent must replace these guesses with measured numbers after session 1)

Assumptions: the Kannada run cost under $100 at about $1.80/h on an A100; a T4 is 5–8× slower than an A100 for this workload; two T4s give about 1.7× the speed of one; the weekly quota is about 30 h.
1. Upper bound in A100-hours for 11 h of data: $100 ÷ $1.79 = 55.87 h.
2. Scale to 15 h of data: 15 ÷ 11 = 1.364, so 55.87 × 1.364 = 76.2 A100-hours.
3. On one T4: 76.2 × 5 = 381 h to 76.2 × 8 = 610 h.
4. On T4 ×2: 381 ÷ 1.7 = 224 h to 610 ÷ 1.7 = 359 h.
5. At 30 quota-hours a week: 224 ÷ 30 = **7.5 weeks** to 359 ÷ 30 = **12 weeks**.
6. "Under $100" is an upper bound, so the real number may be about half: **about 4–6 weeks**.

**Milestones, so the user gets something early:**
- **M1:** frontend plus the Hindi-voice baseline. No GPU quota is needed; this happens on day 1.
- **M2:** after the **best Stage-1 epoch**, export a Kokoro-format **v0.1** (Phase 8, steps 1–5) and upload it as a separate revision (`revision="v0.1-stage1"`). It is already intelligible Urdu, with flatter prosody.
- **M3:** after Stage 2, the full v1.0 release (Phases 8–9).

### 14.6 One-time things the user must do for Kaggle mode (about 10 minutes)

1. kaggle.com → Settings → **verify your phone number**. Internet access in kernels is impossible without this.
2. kaggle.com → Settings → API → **Create New Token**. Put `username` and `key` into `KAGGLE_USERNAME` and `KAGGLE_KEY`.
3. Run `gh auth login` once on the machine where Claude Code runs.
4. **After the agent's first `kaggle kernels push --no-run`**: open `https://www.kaggle.com/code/${KAGGLE_USERNAME}/kokoro-urdu-train/edit` → **Add-ons → Secrets** → add the label `HF_TOKEN` with your HF write token, and make sure it is **attached** to this notebook. This step cannot be automated. The agent pauses at this point, gives the user the link, and checks the secret with a 2-minute test session before continuing.
5. Accept the Rasa terms on Hugging Face (already in Section 1).

### 14.7 Changes to the other phases in `kaggle` mode

- **Phase 4:** pin the same packages, but install them into the Kaggle image with `pip install --no-deps` where Kaggle already ships compatible torch/CUDA. Check `torch.cuda.device_count() == 2` at startup, and abort the session (with a logged error) if it isn't.
- **Phases 5–6:** run through `train_session.py` with shards and the time guard. Back up to HF after **every shard**, not just every epoch.
- **Phase 7:** run a light eval at the end of each session (10 Urdu, 5 mixed and 5 English sentences) and the full eval only on candidate checkpoints, to save quota.
- **Phase 8:** export and ONNX conversion run in a CPU-only Kaggle session (`enable_gpu: false`), which does not use GPU quota.
- **Phase 9:** the fresh-venv verification runs in GitHub Actions (CPU) or a CPU Kaggle session.

---

## Appendix A — Urdu test sentences (also used for `samples/`)

1. پاکستان ایک خوبصورت ملک ہے۔
2. میں نے کل بازار سے تین کتابیں خریدیں۔
3. بھائی، گھر میں دھوپ بہت تیز ہے۔
4. وزیرِ اعظم نے آج قوم سے خطاب کیا۔
5. لڑکی نے اپنی چھوٹی بہن کو کہانی پڑھ کر سنائی۔
6. ڈاکٹر صاحب نے کہا کہ تم جلد ٹھیک ہو جاؤ گے۔
7. ہاں، مجھے یاد ہے، ہم وہاں نہیں گئے تھے۔
8. اس کی قیمت ۲۵۰۰ روپے ہے۔
9. میری سالگرہ 14 اگست کو ہوتی ہے۔
10. ٹرین صبح ۷:۴۵ پر روانہ ہوئی۔
11. غالب کا شعر آج بھی ہر زبان پر ہے۔
12. محبت، خلوص اور اعتماد زندگی کے ستون ہیں۔
13. کیا آپ میری بات سمجھ رہے ہیں؟
14. شکریہ! آپ کا دن اچھا گزرے۔
15. قلم، خط اور غلطی جیسے الفاظ میں ق، خ اور غ کی آواز واضح ہونی چاہیے۔
16. میں نے اپنا laptop آن کیا اور email چیک کی۔
17. ڈاکٹر عبدالقدیر خان پاکستان کے مشہور سائنسدان تھے۔
18. یہ سڑک بہت چوڑی ہے۔
19. انہوں نے کہا تھا کہ وہ جمعرات کو آئیں گے۔
20. اردو ایک شیریں زبان ہے۔
21. ۱۹۴۷ء میں پاکستان آزاد ہوا۔
22. درجۂ حرارت ۳۸ اعشاریہ ۵ ڈگری تھا۔

## Appendix A2 — Urdu–English code-switched test sentences

1. آج کی meeting کینسل ہو گئی ہے۔
2. میں نے اپنا laptop آن کیا اور email چیک کی۔
3. Please اپنا password کسی کو share نہ کریں۔
4. ہمارا next step یہ ہے کہ client کو proposal بھیج دیں۔
5. یہ app بہت user friendly ہے لیکن battery جلدی ختم کرتی ہے۔
6. Thank you so much، آپ نے بہت help کی۔
7. Office میں آج internet کا problem تھا۔
8. اس project کی deadline اگلے Monday تک ہے۔
9. میں weekend پر Karachi جا رہا ہوں۔
10. Doctor نے کہا کہ یہ normal infection ہے، worry نہ کریں۔
11. ہمیں AI اور machine learning پر focus کرنا چاہیے۔
12. Update کے بعد phone ٹھیک سے کام کر رہا ہے۔
13. یہ meeting 3 pm پر Zoom پر ہو گی۔
14. Customer service نے میری complaint register کر لی۔
15. آج weather بہت pleasant ہے، let's go for a walk۔

## Appendix A3 — English-only test sentences

1. Good morning, how are you today?
2. The quick brown fox jumps over the lazy dog.
3. Please send me the report by Thursday afternoon.
4. Artificial intelligence is changing the way we work.
5. I think this is the best decision for our team.
6. The meeting has been moved to three thirty.
7. Thank you for your patience and understanding.
8. Karachi is the largest city in Pakistan.

## Appendix B — Gold pronunciation checks (canonical phone set, after the 4.3 mapping)

| Word | Expected |
|---|---|
| پاکستان | `pɑːkɪstɑːn` |
| خوبصورت | `xuːbsuːrət` |
| پڑھنا | `pəɽʰnɑː` |
| بھائی | `bʰɑːiː` |
| گھر | `ɡʰər` |
| نہیں | `nəhĩː` |
| ہاں | `hɑ̃ː` |
| قلم | `qələm` |
| غلط | `ɣələt` |
| خط | `xət` |
| کتاب | `kɪtɑːb` |
| لڑکی | `ləɽkiː` |
| چھوٹا | `ʧʰoːʈɑː` |
| ٹھیک | `ʈʰiːk` |
| ڈاکٹر | `ɖɑːktər` |
| شکریہ | `ʃʊkrijɑː` |
| آج | `ɑːʤ` |
| زندگی | `zɪndəɡiː` |
| دوست | `doːst` |
| ایک | `eːk` |
| محبت | `mʊhəbbət` |
| وزیرِ اعظم | `ʋəziːreː ɑːzəm` |

If a WikiPron entry disagrees with this table on vowel quality, record both in `DECISIONS.md` and keep the WikiPron form. **Consonants** in this table are authoritative.

---

## Appendix C — Order of work (checklist for the agent's task list)

In `kaggle` mode, steps 4–9 run inside Kaggle sessions driven by the orchestrator (Section 14). Before step 4, the agent must:
- create the HF work repo, the private Kaggle Dataset and the orchestrator GitHub repo;
- do the first `--no-run` push and wait for the user to attach the `HF_TOKEN` secret (14.6, step 4);
- pass a 2-minute secret and GPU test session (it must see 2 GPUs);
- turn the hourly workflow on.

1. Check inputs, GPU, HF write access and gated access → `env_check.md`.
2. Build the frontend (normalizer, phone set, mapping, lexicon, neural G2P, language tagging, misaki English + Pakistani-accent mapping) and its tests.
3. Run the Phase-3 Hindi-voice baseline.
4. Data download (Rasa Urdu + LibriTTS-R English replay, plus ASLP if enabled) → resample → segment → ASR filter → bandwidth audit → phonemize → lists with the target mix → OOD (Urdu, English, synthetic mixed) → upload to the work repo.
5. Set up the training environment, patches, symbol map and base weights; pass the smoke test.
6. Stage 1, with per-epoch eval, backup and a cost projection after epoch 1.
7. Stage 2 (diffusion recipe), with per-epoch eval and backup.
8. Select the checkpoint by the quality gates.
9. Export KModel weights, voicepacks, ONNX, and the diffusion package.
10. Package `kokoro_urdu`, write the model card, create the Space, upload.
11. Verify from a fresh venv and the Space; write `REPORT.md`; print the summary.

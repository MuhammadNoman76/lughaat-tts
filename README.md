# Lughaat-TTS-82M: Kokoro-82M fine-tuned for Urdu, English and Urdu–English code-switching

This repository implements the whole plan in [KOKORO_URDU_AGENT_PLAN.md](KOKORO_URDU_AGENT_PLAN.md):
an Urdu text frontend (`lughaat_tts/`), data preparation, a shard-based training pipeline that runs
on **free Kaggle GPUs (T4 x2)**, evaluation with quality gates, export to Kokoro format + ONNX, and
the Hugging Face upload (model repo + Gradio Space). Everything persistent lives in your own
Hugging Face account; the orchestrator and the notebook are both optional ways to run sessions.

## 1. What you do (about 10 minutes, once)

1. **Hugging Face**: create a token with *write* scope (Settings → Access Tokens). Open
   https://huggingface.co/datasets/ai4bharat/Rasa while logged in and click **Agree and access**
   (the Urdu training data is gated).
2. **Kaggle**: verify your phone number (Settings → Phone verification), otherwise notebooks cannot use the internet.
3. **Get the notebook into Kaggle**: kaggle.com → Code → New Notebook → File → **Import Notebook** → GitHub tab →
   `MuhammadNoman76/lughaat-tts`, file `kaggle/lughaat-tts-train.ipynb`. (Alternative: download that file from
   GitHub and use the Upload tab.) The notebook clones the repository itself when it runs.
4. In the notebook: **Settings → Accelerator = GPU T4 x2, Internet = On**; **Add-ons → Secrets → `HF_TOKEN`**
   (your write token), ticked so it is attached. Optional: `AZURE_OPENAI_API_KEY` (your Azure OpenAI key) so the
   lexicon step can ask that model for pronunciation candidates; endpoint and model name sit in the first cell.
   If the GitHub repo is private, also add a secret `GITHUB_TOKEN`; or make the repo public and skip this.
5. **Save Version → Save & Run All (Commit)**. The session runs up to ~11 h, backs up after every shard, and stops
   cleanly. **Re-run it** (each time the weekly 30 GPU-hour quota allows) until the output says `done=True`.
   With the defaults (12 h Urdu + 3 h English, 8 + 5 epochs) expect roughly 4–8 weeks of free quota; the first
   session prints a measured projection into `REPORT.md`.

Hands-off alternative: copy [orchestrator/workflows/orchestrate.yml](orchestrator/workflows/orchestrate.yml)
into the GitHub repo as `.github/workflows/orchestrate.yml`, add the secrets `KAGGLE_USERNAME`,
`KAGGLE_KEY`, `HF_TOKEN`, and it will push a new session every hour whenever quota is available and
report progress in a GitHub issue ([kaggle/README_KAGGLE.md](kaggle/README_KAGGLE.md) has the details).

## 2. What comes out

In `https://huggingface.co/<you>/lughaat-tts-82m` (private by default):

| file | what |
|---|---|
| `lughaat-tts-82m.pth` | KModel weights (`bert`, `bert_encoder`, `predictor`, `text_encoder`, `decoder`), 82 M params |
| `config.json` | Kokoro config (178-token vocab) + `urdu_frontend_version` + `languages` |
| `voices/uf_rasa.pt`, `voices/um_rasa.pt` | voicepacks `[510, 1, 256]` (female / male) |
| `onnx/lughaat-tts-82m.onnx` (+ `-fp16.onnx` if parity passes) | ONNX export |
| `lughaat_tts/` + `pyproject.toml` | pip-installable frontend + `UrduPipeline` |
| `diffusion/` | full Stage-2 checkpoint + `synthesize_diffusion.py` (highest naturalness) |
| `samples/` | WAVs for every Appendix A / A2 / A3 sentence, both voices |
| `README.md`, `REPORT.md` | model card with metrics and the full training report |

Revision `v0.1-stage1` (after Stage 1) is already intelligible Urdu with flatter prosody; `main` is v1.0.
A Gradio demo is created at `https://huggingface.co/spaces/<you>/lughaat-tts-demo` (free CPU).

```bash
pip install git+https://huggingface.co/<you>/lughaat-tts-82m
#   private repo: pip install "git+https://user:$HF_TOKEN@huggingface.co/<you>/lughaat-tts-82m" and export HF_TOKEN for the weights
python -c "from lughaat_tts import UrduPipeline; UrduPipeline().save('پاکستان ایک خوبصورت ملک ہے۔', 'out.wav')"
python -c "from lughaat_tts import UrduPipeline; print(UrduPipeline().phonemize('آج کی meeting کینسل ہو گئی ہے۔'))"
lughaat-tts --phonemes "آج کی meeting کینسل ہو گئی ہے۔"      # frontend only, no model needed
```

## 3. Repository map

```
lughaat_tts/        the shipped package: normalize.py numbers.py phoneset.py vocab.py skeleton.py rules.py
                    lexicon.py g2p_model.py g2p.py english.py codeswitch.py chunker.py pipeline.py onnx_infer.py cli.py
                    data/: lexicon.tsv (CC-BY-SA, from WikiPron + gold), gold_overrides.tsv, wikipron_test.tsv, g2p_model.pt
tests/              frontend unit tests (Appendix B gold words, A2 tags, Pakistani anchors, vocab, determinism)
scripts/            00_env_check 01_prepare_data 02_build_lexicon 03_train_g2p 04_baseline 05_setup_training
                    06_train_shard 07_export 08_voicepack 09_onnx 10_make_samples 11_model_card 99_upload kaggle_push
training/           kokoro_symbols.py (178-token map) kokoro_config.py (recipes) patches.py (trainer patches) convert_base.py shard.py
eval/               asr.py (Whisper CER/WER) pcer.py (phonetic CER) metrics.py (MOS, ECAPA, roll-off, comb) evaluate.py audit_data.py
kaggle/             lughaat-tts-train.ipynb (the notebook) train_session.py (idempotent session state machine) metadata
orchestrator/       hourly GitHub Actions driver
space/              Gradio demo
diffusion/          full-StyleTTS2 inference with the trained diffusion prosody sampler
DECISIONS.md        every non-obvious choice and why; REPORT.md is written by the pipeline
```

## 4. Running pieces locally (optional)

```bash
uv venv -p 3.12 .venv && uv pip install -p .venv -e ".[dev]"      # frontend only; no GPU needed
.venv/bin/python -m pytest -q tests
.venv/bin/python scripts/03_train_g2p.py --epochs 40               # retrain the neural G2P (CPU ok)
.venv/bin/python scripts/02_build_lexicon.py --wiki-top 20000      # expand the lexicon (needs espeak-ng for candidate A)
```

The training itself needs Linux + CUDA (Kaggle). To run on a rented GPU instead of Kaggle, run
`python kaggle/train_session.py` on that machine with `HF_TOKEN` set and `SESSION_BUDGET_MIN=100000`.

## 5. Quality gates (plan section 9)

Out-of-domain CER ≤ ASR floor + 0.03 and better than the Hindi-voice baseline; Appendix-A CER ≤ floor + 0.06;
code-switched PCER ≤ 0.10 (English words ≤ 0.15, none lost); English WER in the Urdu voices ≤ base `af_heart` WER + 0.05;
MOS ≥ 3.5 (or within 0.3 of the real recordings); no clipping, long silences or comb artefacts. The session runner selects
checkpoints by these gates automatically and records everything in `REPORT.md`.

Licences: code Apache-2.0 (see [NOTICE](NOTICE)); `lughaat_tts/data/lexicon.tsv` is CC-BY-SA-4.0 (WikiPron/Wiktionary).

## Pronunciation review (2026-10-09)

The files in `baseline_samples` use unmodified Kokoro Hindi voices. They are baseline
recordings, not the output of an Urdu-trained acoustic model. Passing frontend tests
or signal checks does not establish correct spoken pronunciation.

To audit every saved recording and create changed candidates alongside the originals:

```bash
python scripts/13_review_baseline.py --text-edits assets/baseline_pronunciation_edits.json --regenerate-changed --offline
```

Open `pronunciation_review/listening_test.html`; `audit.json` contains per-file signal
checks, old/current phonemes, input edits, and hashes. The command uses cached Kokoro
weights with `--offline`; omit that flag to allow downloads. It never overwrites the
original samples. An unchanged phoneme string is not a pronunciation-quality pass.

The frontend recognizes explicit `کِیا` (did) and `کْیا` (question), `جَلْد` (quickly)
and `جِلْد` (skin/volume), and `عَلَم` / `عِلْم`. For example, write
`کْیا آپ نے کام کِیا؟`. Unmarked homographs still require context review; this is
not a general context-sensitive Urdu pronunciation model. The optional JSON text
edits are explicit corrections for these sample sentences, not automatic predictions.

Frontend `ur-frontend-1.0.1` fixes productive izafat and Unicode vowel mapping, pins
`جیسے` and `بھیج`, and rejects unsupported output phones instead of deleting them.
Rebuild prepared phoneme labels before training with this frontend version.

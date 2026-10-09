# Running on Kaggle (free GPU T4 x2)

Two ways to run sessions. Both use the same notebook and the same state in your HF work repo, so you
can mix them (start manually, switch the orchestrator on later).

## A. Manual (simplest): re-run the notebook

1. kaggle.com → Code → New Notebook → File → **Import Notebook** → GitHub tab → `MuhammadNoman76/lughaat-tts`,
   file `kaggle/lughaat-tts-train.ipynb` (or download the file from GitHub and use the Upload tab).
   The notebook clones the repository from `PROJECT_GIT_URL` (first cell) when it runs; no Kaggle dataset is needed.
2. Right panel: **Accelerator: GPU T4 x2**, **Internet: On**.
3. **Add-ons → Secrets**: `HF_TOKEN` (write scope), attached. Private GitHub repo: also `GITHUB_TOKEN` (read access),
   or make the repo public. Optional: `HF_USERNAME`, `ANTHROPIC_API_KEY`.
4. **Save Version → Save & Run All (Commit)**. Close the browser; the run continues in the background for up to
   ~11 h 15 min and writes `summary.txt` / `state.json` into the notebook output and into
   `https://huggingface.co/datasets/<you>/lughaat-tts-work`.
5. Repeat step 4 whenever `kaggle.com → Settings → Accelerator quota` shows ≥ 2 free GPU hours, until the
   output says `done=True`. Progress is in `progress.md` and `REPORT.md` in the work repo.

Phases and what each session does: `env_check → data_prep (select, audio, asr, audit, lexicon, phonemize,
lists, ood, pack) → baseline → setup (+ smoke test) → stage1 (shards) → select_s1 → export_v01 → stage2 (shards)
→ select_s2 → export → eval_full → upload → done`. Every unit is idempotent; a killed session resumes from
the last uploaded shard checkpoint.

## B. Hands-off: GitHub Actions orchestrator

1. `pip install kaggle` locally, put your `kaggle.json` in `~/.kaggle/`, then
   `python scripts/kaggle_push.py --code --kernel --no-run` (creates the code dataset and the kernel).
   Open `https://www.kaggle.com/code/<you>/lughaat-tts-train/edit` → Add-ons → Secrets → add and attach `HF_TOKEN`.
2. Create a private GitHub repo containing this folder; copy `orchestrator/workflows/orchestrate.yml` to
   `.github/workflows/orchestrate.yml`; set secrets `KAGGLE_USERNAME`, `KAGGLE_KEY`, `HF_TOKEN`
   (`gh secret set NAME`), and optionally variables `HF_USERNAME`, `KAGGLE_KERNEL`, `ACCELERATOR`.
3. Run the workflow once by hand (Actions → Run workflow). From then on it runs hourly: it checks the kernel
   status and `kaggle quota`, pushes a session when ≥ 1.5 GPU-hours are available, and updates the pinned
   issue "Training progress". Repeated failures open "Training needs attention".

## Knobs (first notebook cell / environment)

| name | default | meaning |
|---|---|---|
| `URDU_HOURS_PER_SPEAKER` | 6 | Rasa hours per speaker; 15 = the plan's paid-GPU setting (3x longer) |
| `ENGLISH_REPLAY_HOURS` | 3 | never below 3; raise to 4–5 if the English WER gate fails |
| `STAGE1_EPOCHS` / `STAGE2_EPOCHS` | 8 / 5 | early stopping after 2 epochs without CER improvement |
| `SHARD_MINUTES` | 60 | shard length; smaller = more frequent backups, more start-up overhead |
| `SESSION_BUDGET_MIN` | 675 | clean exit time; keep below the Kaggle 12 h limit minus the setup time |
| `REPO_PRIVATE` | 1 | create the HF repos private |
| `USE_INDICVOICES_R` | 0 | add IndicVoices-R Urdu (gated) to Stage 1 |
| `LLM_LEXICON_BUDGET_USD` | 0 | > 0 with `ANTHROPIC_API_KEY` secret: LLM pronunciation candidates |
| `FORCE_PHASE` | – | jump to a phase (e.g. `stage2` after raising `ENGLISH_REPLAY_HOURS`) |
| `ACCELERATOR` (orchestrator) | NvidiaTeslaT4 | `NvidiaL4` if your account has it: Stage 2 runs the full recipe |

## STOP conditions you must resolve yourself (plan section 12)

- `HF_TOKEN` secret missing / not attached, or without write scope.
- Rasa gated access not accepted (the session prints the URL).
- No Stage-2 variant fits in memory (try `ACCELERATOR=NvidiaL4`).
- The same unit fails 3 times (see `logs/` in the work repo).

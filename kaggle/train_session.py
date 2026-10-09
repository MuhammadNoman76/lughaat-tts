#!/usr/bin/env python3
"""Idempotent Kaggle session: run the next units of work, back everything up to HF, exit before
the 12 h limit (plan 14.2.3). Re-run it (manually or through the orchestrator) until state.done.

Inputs (Kaggle Secrets or environment)
    HF_TOKEN            required (write scope). Kaggle: Add-ons -> Secrets -> HF_TOKEN
    HF_USERNAME         optional; defaults to the token owner (whoami)
    PROJECT_DIR         where this repo was copied (default: the parent of this file)
    SESSION_BUDGET_MIN  wall-clock budget for this session (default 675 = 11 h 15 min)
    URDU_HOURS_PER_SPEAKER (6) ENGLISH_REPLAY_HOURS (3) STAGE1_EPOCHS (8) STAGE2_EPOCHS (5)
    SHARD_MINUTES (60) REPO_PRIVATE (1) USE_INDICVOICES_R (0) LLM_LEXICON_MAX_WORDS (0)
    AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_MODEL + secret AZURE_OPENAI_API_KEY  optional (LLM lexicon candidates)
    FORCE_PHASE optional

State (work repo <user>/lughaat-tts-work, file state.json)
    phase: env_check -> data_prep -> baseline -> setup -> stage1 -> select_s1 -> export_v01
           -> stage2 -> select_s2 -> export -> eval_full -> upload -> done
"""
from __future__ import annotations

import glob
import json
import math
import os
import shutil
import subprocess
import sys
import time
import traceback

T0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.environ.get("PROJECT_DIR") or os.path.dirname(HERE)
sys.path.insert(0, PROJECT)

def _pick_tmp() -> str:
    """Large scratch space: /kaggle/tmp (ephemeral) if usable, else /kaggle/temp, /tmp, or a project dir."""
    if os.environ.get("KU_TMP"):
        return os.environ["KU_TMP"]
    cands = ["/kaggle/tmp", "/kaggle/temp", "/tmp/lughaat"] if os.path.isdir("/kaggle") else [os.path.join(PROJECT, ".tmp")]
    for c in cands:
        try:
            os.makedirs(c, exist_ok=True)
            test = os.path.join(c, ".w")
            open(test, "w").write("ok")
            os.remove(test)
            return c
        except Exception:
            continue
    return os.path.join(PROJECT, ".tmp")


TMP = _pick_tmp()
WORKING = os.environ.get("KU_WORKING", "/kaggle/working" if os.path.isdir("/kaggle") else os.path.join(PROJECT, ".working"))
DATA = os.path.join(TMP, "data")
WORK = os.path.join(TMP, "work")
EXPORT = os.path.join(TMP, "export")
for d in (TMP, WORKING, DATA, WORK, EXPORT):
    os.makedirs(d, exist_ok=True)

PHASES = ["env_check", "data_prep", "baseline", "setup", "stage1", "select_s1", "export_v01", "stage2", "select_s2",
          "export", "eval_full", "upload", "done"]
DATA_STEPS = ["select", "audio", "asr", "audit", "lexicon", "phonemize", "lists", "ood", "pack"]
# rough minutes each unit needs (used by the time guard before it is measured)
EST_MIN = {"env_check": 3, "select": 60, "audio": 25, "asr": 45, "audit": 5, "lexicon": 40, "phonemize": 10, "lists": 2,
           "ood": 20, "pack": 20, "baseline": 25, "setup": 35, "select_s1": 10, "export_v01": 45, "select_s2": 10,
           "export": 60, "eval_full": 90, "upload": 40}


def env(name: str, default=None):
    v = os.environ.get(name)
    return v if v not in (None, "") else default


class StopSession(Exception):
    """Raised to end the session cleanly (budget) or to STOP for a human (plan 12)."""


def log(msg: str) -> None:
    line = f"[session {time.strftime('%H:%M:%S')} +{(time.time()-T0)/60:5.1f}m] {msg}"
    print(line, flush=True)
    with open(os.path.join(WORKING, "session.log"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


def minutes_left() -> float:
    return float(env("SESSION_BUDGET_MIN", 675)) - (time.time() - T0) / 60


def need(minutes: float, what: str) -> None:
    if minutes_left() < minutes:
        raise StopSession(f"not enough time left ({minutes_left():.0f} min) for {what} (~{minutes:.0f} min); ending session")


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    log("$ " + " ".join(str(c) for c in cmd))
    return subprocess.run([str(c) for c in cmd], check=True, **kw)


# ---------------------------------------------------------------------------
class Session:
    def __init__(self):
        self.token = self._token()
        from huggingface_hub import HfApi, login
        login(token=self.token, add_to_git_credential=False)
        self.api = HfApi(token=self.token)
        self.user = env("HF_USERNAME") or self.api.whoami()["name"]
        self.work_repo = f"{self.user}/lughaat-tts-work"
        self.model_repo = f"{self.user}/lughaat-tts-82m"
        self.space_repo = f"{self.user}/lughaat-tts-demo"
        os.environ.update({"HF_TOKEN": self.token, "HF_USERNAME": self.user, "WORK_REPO": self.work_repo, "MODEL_REPO": self.model_repo,
                           "HF_HUB_ENABLE_HF_TRANSFER": "0", "TOKENIZERS_PARALLELISM": "false"})
        self.api.create_repo(self.work_repo, repo_type="dataset", private=True, exist_ok=True)
        self.state = self._load_state()
        self.state.setdefault("sessions", []).append({"start": time.strftime("%Y-%m-%d %H:%M:%S"), "gpu": self._gpu_name()})
        self.py = sys.executable

    @staticmethod
    def _token() -> str:
        t = env("HF_TOKEN")
        if not t:
            try:
                from kaggle_secrets import UserSecretsClient  # type: ignore
                t = UserSecretsClient().get_secret("HF_TOKEN")
            except Exception:
                t = None
        if not t:
            raise StopSession("HF_TOKEN missing. Kaggle: Add-ons -> Secrets -> add HF_TOKEN (write scope) and attach it to this notebook.")
        return t

    @staticmethod
    def _gpu_name() -> str:
        try:
            import torch
            return f"{torch.cuda.device_count()}x {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "?"

    # -- HF helpers ----------------------------------------------------------------
    def dl(self, path_in_repo: str, local: str, repo: str | None = None, repo_type: str = "dataset") -> str | None:
        from huggingface_hub import hf_hub_download
        try:
            p = hf_hub_download(repo or self.work_repo, path_in_repo, repo_type=repo_type, token=self.token, local_dir=os.path.join(TMP, "hfdl"))
            os.makedirs(os.path.dirname(local) or ".", exist_ok=True)
            if os.path.abspath(p) != os.path.abspath(local):
                shutil.copy(p, local)
            return local
        except Exception:
            return None

    def up(self, local: str, path_in_repo: str, repo: str | None = None, repo_type: str = "dataset", retries: int = 3) -> None:
        for i in range(retries):
            try:
                if os.path.isdir(local):
                    self.api.upload_folder(folder_path=local, path_in_repo=path_in_repo, repo_id=repo or self.work_repo, repo_type=repo_type)
                else:
                    self.api.upload_file(path_or_fileobj=local, path_in_repo=path_in_repo, repo_id=repo or self.work_repo, repo_type=repo_type)
                return
            except Exception as e:
                log(f"upload failed ({e!r}); retry {i+1}/{retries}")
                time.sleep(20)
        raise

    def exists(self, path_in_repo: str, repo: str | None = None, repo_type: str = "dataset") -> bool:
        try:
            return self.api.file_exists(repo or self.work_repo, path_in_repo, repo_type=repo_type)
        except Exception:
            return False

    def _load_state(self) -> dict:
        p = os.path.join(WORKING, "state.json")
        if self.dl("state.json", p):
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        return {"phase": "env_check", "done": False, "stage1": {}, "stage2": {}, "history": [], "errors": {}}

    def save_state(self, note: str = "") -> None:
        p = os.path.join(WORKING, "state.json")
        self.state["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.state, f, ensure_ascii=False, indent=2)
        self.up(p, "state.json")
        if note:
            with open(os.path.join(WORKING, "progress.md"), "a", encoding="utf-8") as f:
                f.write(f"- {time.strftime('%Y-%m-%d %H:%M')} {self.state['phase']}: {note}\n")
            self.up(os.path.join(WORKING, "progress.md"), "progress.md")

    def record_error(self, unit: str, exc: BaseException) -> None:
        tb = traceback.format_exc()
        errs = self.state.setdefault("errors", {})
        errs[unit] = errs.get(unit, 0) + 1
        self.state["last_error"] = {"unit": unit, "error": repr(exc)[:500], "time": time.strftime("%Y-%m-%d %H:%M:%S"), "count": errs[unit]}
        p = os.path.join(WORKING, f"error_{int(time.time())}.log")
        open(p, "w", encoding="utf-8").write(tb)
        try:
            self.up(p, f"logs/{os.path.basename(p)}")
            self.save_state(f"ERROR in {unit}: {repr(exc)[:200]}")
        except Exception:
            pass
        if errs[unit] >= 3:
            log(f"unit {unit} failed {errs[unit]} times; the orchestrator will open an issue. STOP.")

    # -- phases -----------------------------------------------------------------------
    def advance(self, note: str = "") -> None:
        i = PHASES.index(self.state["phase"])
        self.state["phase"] = PHASES[min(i + 1, len(PHASES) - 1)]
        if self.state["phase"] == "done":
            self.state["done"] = True
        self.save_state(note or f"-> {self.state['phase']}")

    def phase_env_check(self) -> None:
        need(EST_MIN["env_check"], "env_check")
        import torch
        n = torch.cuda.device_count()
        gpu = torch.cuda.get_device_name(0) if n else "none"
        lines = [f"# env_check", f"- time: {time.strftime('%Y-%m-%d %H:%M:%S')}", f"- GPUs: {n} x {gpu}", f"- HF user: {self.user}",
                 f"- work repo: {self.work_repo} (private)", f"- model repo: {self.model_repo}"]
        if n == 0:
            raise StopSession("No GPU. Set the accelerator to GPU T4 x2 in the notebook settings.")
        # write access
        self.api.create_repo(self.model_repo, repo_type="model", private=bool(int(env("REPO_PRIVATE", 1))), exist_ok=True)
        lines.append("- HF write access: OK (repos created)")
        # gated Rasa access
        try:
            self.api.get_paths_info("ai4bharat/Rasa", ["Urdu/test-00000-of-00007.parquet"], repo_type="dataset", token=self.token)
            lines.append("- ai4bharat/Rasa access: OK")
        except Exception as e:
            lines.append(f"- ai4bharat/Rasa access: DENIED ({e!r})")
            open(os.path.join(WORKING, "env_check.md"), "w").write("\n".join(lines))
            self.up(os.path.join(WORKING, "env_check.md"), "env_check.md")
            raise StopSession("Access to the gated dataset ai4bharat/Rasa is denied. Open https://huggingface.co/datasets/ai4bharat/Rasa "
                              f"while logged in as {self.user}, click 'Agree and access', then re-run.")
        # python deps
        import importlib
        missing = []
        for mod in ("misaki", "kokoro", "datasets", "soundfile", "librosa", "soxr", "transformers", "accelerate", "yaml"):
            try:
                importlib.import_module(mod)
                lines.append(f"- {mod}: ok")
            except Exception as e:
                lines.append(f"- {mod}: MISSING ({e!r})")
                missing.append(mod)
        open(os.path.join(WORKING, "env_check.md"), "w").write("\n".join(lines) + "\n")
        self.up(os.path.join(WORKING, "env_check.md"), "env_check.md")
        if missing:
            raise StopSession(f"Python packages missing in this session: {missing}. The notebook's dependency cell must "
                              "succeed first (it prints the pip errors); nothing is trained until misaki/kokoro import.")
        # run the frontend unit tests once
        try:
            run([self.py, "-m", "pytest", "-q", "-x", os.path.join(PROJECT, "tests"), "-p", "no:cacheprovider"], cwd=PROJECT)
        except subprocess.CalledProcessError:
            log("WARNING: frontend tests failed; continuing (see output above)")
        self.advance("env check passed: " + lines[2])

    def _restore_data(self) -> bool:
        """Download the prepared data (lists + audio.tar + frontend) from the work repo if present."""
        if os.path.exists(os.path.join(DATA, "train_list.txt")) and os.path.isdir(os.path.join(DATA, "audio")):
            return True
        if not self.exists("data/train_list.txt"):
            return False
        for fn in ("train_list.txt", "val_list.txt", "OOD_texts.txt", "data_report.json", "voicepack_refs.json", "bandwidth.json", "manifest.jsonl"):
            self.dl(f"data/{fn}", os.path.join(DATA, fn))
        from huggingface_hub import snapshot_download
        snapshot_download(self.work_repo, repo_type="dataset", token=self.token, allow_patterns=["data/eval/*", "frontend/data/*"], local_dir=os.path.join(TMP, "hfsnap"))
        if os.path.isdir(os.path.join(TMP, "hfsnap", "data", "eval")):
            shutil.copytree(os.path.join(TMP, "hfsnap", "data", "eval"), os.path.join(DATA, "eval"), dirs_exist_ok=True)
        fd = os.path.join(TMP, "hfsnap", "frontend", "data")
        if os.path.isdir(fd):
            shutil.copytree(fd, os.path.join(PROJECT, "lughaat_tts", "data"), dirs_exist_ok=True)
        tar = os.path.join(DATA, "audio.tar")
        if not os.path.isdir(os.path.join(DATA, "audio")):
            log("downloading audio.tar ...")
            self.dl("data/audio.tar", tar)
            run(["tar", "-xf", tar, "-C", DATA])
            os.remove(tar)
        return True

    def phase_data_prep(self) -> None:
        st = self.state.setdefault("data_prep", {"steps_done": []})
        if self.exists("data/train_list.txt") and self.exists("data/audio.tar"):
            self.advance("data already prepared in the work repo")
            return
        for step in DATA_STEPS:
            if step in st["steps_done"]:
                continue
            need(EST_MIN[step] * 1.3, f"data step {step}")
            t = time.time()
            cmd = [self.py, os.path.join(PROJECT, "scripts", "01_prepare_data.py"), "--out", DATA, "--step", step,
                   "--work-repo", self.work_repo, "--urdu-hours-per-speaker", env("URDU_HOURS_PER_SPEAKER", 6),
                   "--english-hours", env("ENGLISH_REPLAY_HOURS", 3), "--use-indicvoices-r", env("USE_INDICVOICES_R", 0),
                   "--llm-max-words", env("LLM_LEXICON_MAX_WORDS", 0)]
            if step == "pack":
                cmd.append("--upload")
            run(cmd, cwd=PROJECT)
            st["steps_done"].append(step)
            st[f"min_{step}"] = round((time.time() - t) / 60, 1)
            self.save_state(f"data step {step} done in {st[f'min_{step}']} min")
        rep = os.path.join(DATA, "data_report.json")
        if os.path.exists(rep):
            self.state["data_report"] = json.load(open(rep))
        self.advance("data prepared and uploaded")

    def phase_baseline(self) -> None:
        need(EST_MIN["baseline"], "baseline")
        out = os.path.join(TMP, "baseline")
        run([self.py, os.path.join(PROJECT, "scripts", "04_baseline.py"), "--out", out], cwd=PROJECT)
        self.up(out, "baseline")
        self.state["baseline"] = json.load(open(os.path.join(out, "baseline_metrics.json"))).get("metrics")
        self.advance(f"baseline: {self.state['baseline']}")

    def phase_setup(self) -> None:
        need(EST_MIN["setup"], "setup")
        self._restore_data()
        run([self.py, os.path.join(PROJECT, "scripts", "05_setup_training.py"), "--work", WORK, "--data", DATA, "--smoke"], cwd=PROJECT)
        info = json.load(open(os.path.join(WORK, "setup.json")))
        self.state["setup"] = {k: v for k, v in info.items() if k in ("n_gpu", "gpu", "smoke")}
        self.up(os.path.join(WORK, "setup.json"), "setup.json")
        self.up(os.path.join(WORK, "smoke.log"), "logs/smoke.log")
        self.advance(f"setup + smoke test passed: {info.get('smoke')}")

    # -- training ---------------------------------------------------------------------
    def _ensure_env(self) -> None:
        self._restore_data()
        if not os.path.exists(os.path.join(WORK, "setup.json")):
            need(EST_MIN["setup"], "setup (re-create training env)")
            run([self.py, os.path.join(PROJECT, "scripts", "05_setup_training.py"), "--work", WORK, "--data", DATA], cwd=PROJECT)

    def _stage_variants(self, stage: int) -> list[dict]:
        from training.kokoro_config import STAGE1_VARIANTS, STAGE2_VARIANTS
        if stage == 1:
            return STAGE1_VARIANTS
        gpu = self._gpu_name()
        return [v for v in STAGE2_VARIANTS if v["accelerator"] != "NvidiaL4" or "L4" in gpu]

    def _train_stage(self, stage: int) -> None:
        st = self.state.setdefault(f"stage{stage}", {})
        st.setdefault("epoch", 0); st.setdefault("shard", 0); st.setdefault("history", [])
        epochs = int(env("STAGE1_EPOCHS", 8)) if stage == 1 else int(env("STAGE2_EPOCHS", 5))
        shard_min = float(env("SHARD_MINUTES", 60))
        self._ensure_env()
        latest = os.path.join(WORK, f"latest_s{stage}.pth")
        # resume checkpoint
        if st.get("latest_ckpt") and not os.path.exists(latest):
            need(10, "checkpoint download")
            self.dl(st["latest_ckpt"], latest)
        first_stage = ""
        if stage == 2 and not st.get("latest_ckpt"):
            first_stage = os.path.join(WORK, "first_stage.pth")
            if not os.path.exists(first_stage):
                self.dl(self.state["best_s1"], first_stage)
        # pick the variant (memory probe)
        if not st.get("variant"):
            for var in self._stage_variants(stage):
                need(20, f"stage {stage} memory probe {var['name']}")
                res = self._run_shard(stage, 0, 0, 1, var["name"], resume="", first_stage=first_stage, probe_steps=30)
                log(f"probe {var['name']}: {res['status']} peak_mem={res.get('peak_mem_gb')} {res.get('seconds')}s")
                st.setdefault("probes", []).append({k: res.get(k) for k in ("variant", "status", "peak_mem_gb", "seconds", "losses")})
                if res["status"] == "ok":
                    st["variant"] = var["name"]
                    break
                self.save_state(f"stage {stage} probe {var['name']} -> {res['status']}")
            if not st.get("variant"):
                raise StopSession(f"No Stage-{stage} variant fits in GPU memory (probes: {st.get('probes')}). Try the NvidiaL4 accelerator.")
            # a probe never leaves a usable checkpoint
            for p in glob.glob(os.path.join(WORK, f"latest_s{stage}.pth")):
                os.remove(p)
            self.save_state(f"stage {stage} variant {st['variant']}")
        # calibration shard -> number of shards
        if not st.get("num_shards"):
            need(shard_min * 1.5, "calibration shard")
            from training.shard import read_list, calibration_lines, write_list, num_shards_for
            lines = read_list(os.path.join(DATA, "train_list.txt"))
            cal = write_list(os.path.join(WORK, "shards", f"calib_s{stage}.txt"), calibration_lines(lines, 600))
            res = self._run_shard(stage, 0, 0, 1, st["variant"], resume="", first_stage=first_stage, shard_list=cal)
            if res["status"] != "ok":
                raise RuntimeError(f"calibration shard failed: {res}")
            K = num_shards_for(len(lines), res["lines"], res["seconds"], shard_min)
            st.update({"num_shards": K, "sec_per_clip": round((res["seconds"] - 240) / max(res["lines"], 1), 3), "calib": res})
            self._after_shard(stage, res, calibration=True)
            self.save_state(f"stage {stage}: {len(lines)} clips, {K} shards/epoch (~{shard_min} min each), first-pass projection")
            self._project(stage, len(lines), epochs)
        # shard loop
        while st["epoch"] < epochs:
            # a real shard is sized to shard_min; the calibration shard (600 clips) only bounds it from below
            est = max(shard_min * 1.2 + 8, (st.get("calib", {}).get("seconds", shard_min * 60) / 60) * 1.25 + 8)
            need(est, f"stage {stage} shard e{st['epoch']} s{st['shard']}")
            resume = latest if os.path.exists(latest) else ""
            res = self._run_shard(stage, st["epoch"], st["shard"], st["num_shards"], st["variant"], resume=resume,
                                  first_stage="" if resume else first_stage)
            if res["status"] != "ok":
                st.setdefault("failures", []).append(res)
                self.save_state(f"shard failed: {res['status']}")
                if res["status"] in ("oom", "hang") and len(st["failures"]) <= 2:
                    # fall back to the next variant
                    names = [v["name"] for v in self._stage_variants(stage)]
                    i = names.index(st["variant"])
                    if i + 1 < len(names):
                        st["variant"] = names[i + 1]
                        log(f"switching to variant {st['variant']}")
                        continue
                raise RuntimeError(f"stage {stage} shard failed: {res}")
            self._after_shard(stage, res)
            st["shard"] += 1
            if st["shard"] >= st["num_shards"]:
                st["shard"] = 0
                st["epoch"] += 1
                self._end_of_epoch(stage, st["epoch"] - 1)
                if self._early_stop(stage):
                    log("early stop: validation CER has not improved for 2 epochs")
                    break
            self.save_state(f"stage {stage} epoch {st['epoch']} shard {st['shard']} val_mel={res['losses'].get('val_mel')}")
        self.advance(f"stage {stage} finished ({st['epoch']} epochs)")

    def _run_shard(self, stage, epoch, shard, num_shards, variant, resume, first_stage, shard_list=None, probe_steps=0) -> dict:
        out_json = os.path.join(WORK, "shard_result.json")
        cmd = [self.py, os.path.join(PROJECT, "scripts", "06_train_shard.py"), "--stage", stage, "--epoch", epoch, "--shard", shard,
               "--num-shards", num_shards, "--work", WORK, "--data", DATA, "--variant", variant, "--out-json", out_json,
               "--timeout-minutes", max(30, minutes_left() - 10)]
        if resume:
            cmd += ["--resume", resume]
        if first_stage:
            cmd += ["--first-stage", first_stage]
        if shard_list:
            cmd += ["--shard-list", shard_list]
        if probe_steps:
            cmd += ["--probe-steps", probe_steps]
        subprocess.run([str(c) for c in cmd], cwd=PROJECT)
        return json.load(open(out_json))

    def _after_shard(self, stage: int, res: dict, calibration: bool = False) -> None:
        st = self.state[f"stage{stage}"]
        ck = res.get("checkpoint")
        if not ck or not os.path.exists(ck):
            return
        tag = f"s{stage}_e{res['epoch']:02d}_sh{res['shard']:02d}" + ("_calib" if calibration else "")
        path_in_repo = f"ckpts/{tag}.pth"
        # never time-guard this upload: a finished shard must always be backed up (plan rule 2.4)
        self.up(ck, path_in_repo)
        st["latest_ckpt"] = path_in_repo
        st.setdefault("uploaded", []).append(path_in_repo)
        self.up(os.path.join(WORK, f"train_s{stage}.log"), f"logs/train_s{stage}.log")
        st["history"].append({k: res.get(k) for k in ("epoch", "shard", "seconds", "losses", "variant")})
        # keep only the last 2 shard checkpoints (+ epoch-end and best ones)
        keep = set(st["uploaded"][-2:]) | set(st.get("epoch_ckpts", {}).values()) | {self.state.get("best_s1"), self.state.get("best_s2")}
        for old in list(st["uploaded"][:-2]):
            if old not in keep and "_e" in old and "_end" not in old:
                try:
                    self.api.delete_file(old, self.work_repo, repo_type="dataset")
                except Exception:
                    pass
                st["uploaded"].remove(old)

    def _end_of_epoch(self, stage: int, epoch: int) -> None:
        """Copy the epoch-end checkpoint, run the light evaluation, record metrics."""
        st = self.state[f"stage{stage}"]
        latest = os.path.join(WORK, f"latest_s{stage}.pth")
        tag = f"ckpts/s{stage}_e{epoch:02d}_end.pth"
        self.up(latest, tag)
        st.setdefault("epoch_ckpts", {})[str(epoch)] = tag
        val_mel = (st["history"][-1].get("losses") or {}).get("val_mel")
        metrics = {"epoch": epoch, "val_mel": val_mel}
        if minutes_left() > 35:
            try:
                metrics.update(self._light_eval(stage, epoch))
            except Exception as e:
                log(f"light eval failed: {e!r}")
                metrics["eval_error"] = repr(e)
        st.setdefault("epochs", []).append(metrics)
        self.save_state(f"stage {stage} epoch {epoch} done: {metrics}")

    def _light_eval(self, stage: int, epoch: int) -> dict:
        """Quick export + voicepack + 10/5/5-sentence eval (plan 14.7)."""
        exp = os.path.join(TMP, f"light_s{stage}_e{epoch:02d}")
        os.makedirs(exp, exist_ok=True)
        latest = os.path.join(WORK, f"latest_s{stage}.pth")
        run([self.py, os.path.join(PROJECT, "scripts", "07_export.py"), "--ckpt", latest, "--out", exp], cwd=PROJECT)
        style_ckpt = latest if stage == 1 else (os.path.join(WORK, "first_stage.pth") if os.path.exists(os.path.join(WORK, "first_stage.pth")) else latest)
        run([self.py, os.path.join(PROJECT, "scripts", "08_voicepack.py"), "--style-ckpt", style_ckpt, "--predictor-ckpt", latest,
             "--audio-dir", os.path.join(DATA, "audio"), "--refs", os.path.join(DATA, "voicepack_refs.json"),
             "--bandwidth", os.path.join(DATA, "bandwidth.json"), "--out", os.path.join(exp, "voices"), "--num", "60", "--no-mos"], cwd=PROJECT)
        base = os.path.join(TMP, "baseline", "baseline_metrics.json")
        if not os.path.exists(base):
            self.dl("baseline/baseline_metrics.json", base)
        cmd = [self.py, os.path.join(PROJECT, "eval", "evaluate.py"), "--config", os.path.join(exp, "config.json"),
               "--model", os.path.join(exp, "lughaat-tts-82m.pth"), "--voices", os.path.join(exp, "voices"), "--out", os.path.join(exp, "eval"),
               "--light", "--tag", f"s{stage}_e{epoch}", "--data", DATA, "--asr-model", env("ASR_EVAL_MODEL", "openai/whisper-large-v3-turbo")]
        if os.path.exists(base):
            cmd += ["--baseline", base]
        run(cmd, cwd=PROJECT)
        m = json.load(open(os.path.join(exp, "eval", "metrics.json")))
        self.up(os.path.join(exp, "eval"), f"eval/light_s{stage}_e{epoch:02d}")
        shutil.rmtree(exp, ignore_errors=True)
        return {k: m.get(k) for k in ("appendixA_cer", "mixed_pcer_pakistani", "english_wer_native", "dnsmos", "gates")}

    def _early_stop(self, stage: int) -> bool:
        eps = self.state[f"stage{stage}"].get("epochs", [])
        cers = [e.get("appendixA_cer") for e in eps if e.get("appendixA_cer") is not None]
        if len(cers) < 4:
            return False
        best = min(cers[:-2])
        return all(c >= best - 0.005 for c in cers[-2:])

    def _project(self, stage: int, n_clips: int, epochs: int) -> None:
        st = self.state[f"stage{stage}"]
        spc = st.get("sec_per_clip", 0.5)
        hours = n_clips * spc * epochs / 3600 * 1.15
        quota = float(env("WEEKLY_GPU_HOURS", 30))
        st["projection"] = {"gpu_hours": round(hours, 1), "weeks_at_quota": round(hours / quota, 2), "weekly_quota_h": quota}
        with open(os.path.join(WORKING, "REPORT.md"), "a", encoding="utf-8") as f:
            f.write(f"\n## Stage {stage} projection ({time.strftime('%Y-%m-%d')})\n\n{n_clips} clips, {spc} s/clip, {epochs} epochs -> "
                    f"~{hours:.1f} GPU-hours = ~{hours/quota:.1f} weeks at {quota} h/week.\n")
        self.up(os.path.join(WORKING, "REPORT.md"), "REPORT.md")

    def phase_stage1(self) -> None:
        self._train_stage(1)

    def phase_stage2(self) -> None:
        self._train_stage(2)

    def _select(self, stage: int) -> str:
        st = self.state[f"stage{stage}"]
        eps = st.get("epochs", [])
        if not eps:
            raise RuntimeError(f"no completed stage-{stage} epochs")

        def score(e):
            cer = e.get("appendixA_cer")
            pcer = e.get("mixed_pcer_pakistani")
            wer = e.get("english_wer_native")
            val = e.get("val_mel") or 9
            s = (cer if cer is not None else 1.0) * 3 + (pcer if pcer is not None else 0.5) + (wer if wer is not None else 0.5) + val
            return s
        best = min(eps, key=score)
        tag = st["epoch_ckpts"][str(best["epoch"])]
        self.state[f"best_s{stage}"] = tag
        self.state[f"best_s{stage}_epoch"] = best["epoch"]
        return tag

    def phase_select_s1(self) -> None:
        need(EST_MIN["select_s1"], "select_s1")
        tag = self._select(1)
        # first_stage.pth = best S1 checkpoint with epoch reset to 0
        local = os.path.join(WORK, "best_s1.pth")
        self.dl(tag, local)
        import torch
        st = torch.load(local, map_location="cpu", weights_only=False)
        st["epoch"] = 0
        fs = os.path.join(WORK, "first_stage.pth")
        torch.save(st, fs)
        self.up(fs, "ckpts/first_stage.pth")
        self.state["best_s1"] = "ckpts/first_stage.pth"
        self.advance(f"best stage-1 epoch {self.state['best_s1_epoch']} -> first_stage.pth")

    def _export_bundle(self, s1_ckpt: str, s2_ckpt: str | None, out: str, full: bool) -> None:
        os.makedirs(out, exist_ok=True)
        main_ckpt = s2_ckpt or s1_ckpt
        run([self.py, os.path.join(PROJECT, "scripts", "07_export.py"), "--ckpt", main_ckpt, "--out", out,
             "--extra-config", json.dumps({"training": {"stage1_epoch": self.state.get("best_s1_epoch"), "stage2_epoch": self.state.get("best_s2_epoch"),
                                                        "stage2_variant": self.state.get("stage2", {}).get("variant")}})], cwd=PROJECT)
        run([self.py, os.path.join(PROJECT, "scripts", "08_voicepack.py"), "--style-ckpt", s1_ckpt, "--predictor-ckpt", main_ckpt,
             "--audio-dir", os.path.join(DATA, "audio"), "--refs", os.path.join(DATA, "voicepack_refs.json"),
             "--bandwidth", os.path.join(DATA, "bandwidth.json"), "--out", os.path.join(out, "voices"), "--num", "200"], cwd=PROJECT)
        if full:
            run([self.py, os.path.join(PROJECT, "scripts", "09_onnx.py"), "--config", os.path.join(out, "config.json"),
                 "--model", os.path.join(out, "lughaat-tts-82m.pth"), "--voice", os.path.join(out, "voices", "uf_rasa.pt"),
                 "--out", os.path.join(out, "onnx")], cwd=PROJECT)
            run([self.py, os.path.join(PROJECT, "scripts", "10_make_samples.py"), "--config", os.path.join(out, "config.json"),
                 "--model", os.path.join(out, "lughaat-tts-82m.pth"), "--voices", os.path.join(out, "voices"), "--out", os.path.join(out, "samples")], cwd=PROJECT)

    def phase_export_v01(self) -> None:
        need(EST_MIN["export_v01"], "export_v01")
        self._ensure_env()
        s1 = os.path.join(WORK, "first_stage.pth")
        if not os.path.exists(s1):
            self.dl(self.state["best_s1"], s1)
        out = os.path.join(EXPORT, "v0.1")
        self._export_bundle(s1, None, out, full=True)
        run([self.py, os.path.join(PROJECT, "scripts", "99_upload.py"), "--export", out, "--revision", "v0.1-stage1", "--no-space",
             "--model-repo", self.model_repo, "--private", env("REPO_PRIVATE", 1)], cwd=PROJECT)
        self.advance("v0.1 (Stage-1) exported and uploaded")

    def phase_select_s2(self) -> None:
        need(EST_MIN["select_s2"], "select_s2")
        tag = self._select(2)
        local = os.path.join(WORK, "best_s2.pth")
        self.dl(tag, local)
        self.up(local, "ckpts/best_s2.pth")
        self.state["best_s2"] = "ckpts/best_s2.pth"
        self.advance(f"best stage-2 epoch {self.state['best_s2_epoch']}")

    def phase_export(self) -> None:
        need(EST_MIN["export"], "export")
        self._ensure_env()
        s1 = os.path.join(WORK, "first_stage.pth")
        s2 = os.path.join(WORK, "best_s2.pth")
        for tag, local in ((self.state["best_s1"], s1), (self.state["best_s2"], s2)):
            if not os.path.exists(local):
                self.dl(tag, local)
        out = os.path.join(EXPORT, "v1.0")
        self._export_bundle(s1, s2, out, full=True)
        # diffusion package: full Stage-2 checkpoint + config + synthesize script
        dif = os.path.join(out, "diffusion")
        os.makedirs(dif, exist_ok=True)
        shutil.copy(s2, os.path.join(dif, "stage2_full.pth"))
        shutil.copy(os.path.join(PROJECT, "diffusion", "synthesize_diffusion.py"), dif)
        cfgs = sorted(glob.glob(os.path.join(WORK, "config_s2_*.yml")))
        if cfgs:
            shutil.copy(cfgs[-1], os.path.join(dif, "config_s2.yml"))
        self.advance("v1.0 exported")

    def phase_eval_full(self) -> None:
        need(EST_MIN["eval_full"], "eval_full")
        self._restore_data()
        out = os.path.join(EXPORT, "v1.0")
        base = os.path.join(TMP, "baseline", "baseline_metrics.json")
        if not os.path.exists(base):
            self.dl("baseline/baseline_metrics.json", base)
        cmd = [self.py, os.path.join(PROJECT, "eval", "evaluate.py"), "--config", os.path.join(out, "config.json"),
               "--model", os.path.join(out, "lughaat-tts-82m.pth"), "--voices", os.path.join(out, "voices"),
               "--out", os.path.join(out, "eval"), "--data", DATA, "--tag", "v1.0", "--asr-model", env("ASR_EVAL_MODEL", "openai/whisper-large-v3")]
        if os.path.exists(base):
            cmd += ["--baseline", base]
        run(cmd, cwd=PROJECT)
        m = json.load(open(os.path.join(out, "eval", "metrics.json")))
        self.state["final_metrics"] = {k: v for k, v in m.items() if k not in ("gates_doc",)}
        self.up(os.path.join(out, "eval"), "eval/full_v1.0")
        if not m.get("gates_passed"):
            attempts = self.state.get("stage2_attempts", 0)
            log(f"quality gates failed: {m.get('gates')}")
            if attempts < 2 and self.state.get("final_metrics", {}).get("english_wer_native", 0) > (m.get("baseline", {}).get("A3_wer_af_heart", 0) + 0.05):
                self.state["stage2_attempts"] = attempts + 1
                self.state["notes"] = (self.state.get("notes") or []) + ["English WER gate failed: ship auto accent with pakistani fallback (plan 9)"]
        self.advance(f"full eval: gates_passed={m.get('gates_passed')} {m.get('gates')}")

    def phase_upload(self) -> None:
        need(EST_MIN["upload"], "upload")
        out = os.path.join(EXPORT, "v1.0")
        self._write_report()
        run([self.py, os.path.join(PROJECT, "scripts", "11_model_card.py"), "--export", out, "--state", os.path.join(WORKING, "state.json"),
             "--model-repo", self.model_repo, "--out", os.path.join(out, "README.md")], cwd=PROJECT)
        shutil.copy(os.path.join(WORKING, "REPORT.md"), os.path.join(out, "REPORT.md"))
        run([self.py, os.path.join(PROJECT, "scripts", "99_upload.py"), "--export", out, "--model-repo", self.model_repo,
             "--space-repo", self.space_repo, "--private", env("REPO_PRIVATE", 1), "--work-repo", self.work_repo], cwd=PROJECT)
        # final verification from a fresh venv (plan 11.2.5); non-fatal, results go to the work repo
        if minutes_left() > 30:
            try:
                vout = os.path.join(TMP, "verify_install")
                run([self.py, os.path.join(PROJECT, "scripts", "12_verify_install.py"), "--model-repo", self.model_repo, "--out", vout,
                     "--reference", os.path.join(out, "eval", "metrics.json")], cwd=PROJECT)
                self.up(vout, "eval/verify_install")
                self.state["verify_install"] = json.load(open(os.path.join(vout, "verify.json")))
            except Exception as e:
                log(f"fresh-venv verification failed (non-fatal): {e!r}")
                self.state["verify_install"] = {"error": repr(e)}
        self.advance(f"uploaded to https://huggingface.co/{self.model_repo} and https://huggingface.co/spaces/{self.space_repo}")

    def _write_report(self) -> None:
        s = self.state
        lines = ["# Lughaat-TTS-82M training report", "", f"Generated {time.strftime('%Y-%m-%d %H:%M')} for `{self.model_repo}`", "",
                 "## Data", "", "```", json.dumps(s.get("data_report", {}).get("hours_final", {}), ensure_ascii=False, indent=1), "```",
                 f"Mix by hours: {s.get('data_report', {}).get('mix_percent_by_hours')}",
                 f"ASR filter: {json.dumps(s.get('data_report', {}).get('asr', {}), ensure_ascii=False)[:600]}",
                 f"G2P: {s.get('data_report', {}).get('g2p')}", "",
                 "## Baseline (base Kokoro, Hindi voices)", "", f"{s.get('baseline')}", "",
                 "## Stage 1", "", f"variant {s.get('stage1', {}).get('variant')}, {s.get('stage1', {}).get('num_shards')} shards/epoch, projection {s.get('stage1', {}).get('projection')}", ""]
        for e in s.get("stage1", {}).get("epochs", []):
            lines.append(f"- epoch {e.get('epoch')}: val_mel {e.get('val_mel')} CER {e.get('appendixA_cer')} PCER {e.get('mixed_pcer_pakistani')} EN-WER {e.get('english_wer_native')} DNSMOS {e.get('dnsmos')}")
        lines += ["", f"Best Stage-1 epoch: {s.get('best_s1_epoch')}", "", "## Stage 2", "",
                  f"variant {s.get('stage2', {}).get('variant')} (probes: {s.get('stage2', {}).get('probes')}), projection {s.get('stage2', {}).get('projection')}", ""]
        for e in s.get("stage2", {}).get("epochs", []):
            lines.append(f"- epoch {e.get('epoch')}: val_mel {e.get('val_mel')} CER {e.get('appendixA_cer')} PCER {e.get('mixed_pcer_pakistani')} EN-WER {e.get('english_wer_native')} DNSMOS {e.get('dnsmos')}")
        lines += ["", f"Best Stage-2 epoch: {s.get('best_s2_epoch')}", "", "## Final metrics (full evaluation)", "", "```",
                  json.dumps(s.get("final_metrics", {}), ensure_ascii=False, indent=1, default=str), "```", "",
                  "## Sessions", ""] + [f"- {x}" for x in s.get("sessions", [])] + ["", "## Notes", ""] + [f"- {n}" for n in (s.get("notes") or [])]
        gpu_h = sum((h.get("seconds") or 0) for st in ("stage1", "stage2") for h in s.get(st, {}).get("history", [])) / 3600
        lines += ["", f"Total measured training GPU time: {gpu_h:.1f} h (Kaggle free quota; cost $0).", ""]
        with open(os.path.join(WORKING, "REPORT.md"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        self.up(os.path.join(WORKING, "REPORT.md"), "REPORT.md")

    # -- main loop ---------------------------------------------------------------------
    def loop(self) -> None:
        forced = env("FORCE_PHASE")
        if forced in PHASES:
            self.state["phase"] = forced
        log(f"user={self.user} phase={self.state['phase']} budget={env('SESSION_BUDGET_MIN', 675)} min gpu={self._gpu_name()}")
        while not self.state.get("done"):
            phase = self.state["phase"]
            if self.state.get("errors", {}).get(phase, 0) >= 3 and not forced:
                raise StopSession(f"phase {phase} failed 3 times; needs a human (see logs/ in {self.work_repo})")
            try:
                getattr(self, f"phase_{phase}")()
            except StopSession:
                raise
            except Exception as e:
                self.record_error(phase, e)
                raise
        log("ALL DONE")


def main() -> None:
    summary = os.path.join(WORKING, "summary.txt")
    try:
        s = Session()
        try:
            s.loop()
        except StopSession as e:
            log(f"STOP: {e}")
            s.state["stop_reason"] = str(e)
            s.save_state(f"session ended: {e}")
        finally:
            open(summary, "w", encoding="utf-8").write(f"{time.strftime('%Y-%m-%d %H:%M')} phase={s.state.get('phase')} done={s.state.get('done')} {s.state.get('stop_reason','')}\n")
    except StopSession as e:
        log(f"STOP: {e}")
        open(summary, "w", encoding="utf-8").write(f"STOP: {e}\n")
        sys.exit(2)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Local helper: upload this repo as the Kaggle Dataset `lughaat-tts-code` and create/push the kernel.

    pip install kaggle   (and put kaggle.json in ~/.kaggle/ or set KAGGLE_USERNAME / KAGGLE_KEY)
    python scripts/kaggle_push.py --code                 # create/version the code dataset
    python scripts/kaggle_push.py --kernel --no-run      # create the kernel (then add the HF_TOKEN secret in the UI)
    python scripts/kaggle_push.py --kernel               # push and run a session now (T4 x2)
    python scripts/kaggle_push.py --kernel --accelerator NvidiaL4

Writes kaggle/kernel-metadata.json / dataset-metadata.json with your username filled in.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXCLUDE = {".git", ".venv", "venv", ".tmp", ".working", "__pycache__", ".pytest_cache", "reports", "assets_large", "node_modules"}


def kaggle_user() -> str:
    u = os.environ.get("KAGGLE_USERNAME")
    if u:
        return u
    p = os.path.expanduser("~/.kaggle/kaggle.json")
    if os.path.exists(p):
        return json.load(open(p))["username"]
    raise SystemExit("set KAGGLE_USERNAME/KAGGLE_KEY or create ~/.kaggle/kaggle.json")


def sh(cmd: list[str]) -> None:
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def stage_code(dst: str) -> None:
    for name in os.listdir(ROOT):
        if name in EXCLUDE:
            continue
        src = os.path.join(ROOT, name)
        if os.path.isdir(src):
            shutil.copytree(src, os.path.join(dst, name), ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git"))
        else:
            shutil.copy(src, dst)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", action="store_true", help="create or version the lughaat-tts-code dataset")
    ap.add_argument("--kernel", action="store_true", help="push the notebook kernel")
    ap.add_argument("--no-run", action="store_true")
    ap.add_argument("--accelerator", default="NvidiaTeslaT4")
    ap.add_argument("--timeout", type=int, default=41400, help="seconds (default 11.5 h)")
    a = ap.parse_args()
    user = kaggle_user()
    if a.code:
        stage = tempfile.mkdtemp(prefix="lt_code_")
        stage_code(os.path.join(stage, "lughaat-tts"))
        tmp = tempfile.mkdtemp(prefix="lt_ds_")
        # ONE archive so the notebook can unpack it regardless of Kaggle's directory handling
        shutil.make_archive(os.path.join(tmp, "project"), "zip", stage)
        shutil.rmtree(stage, ignore_errors=True)
        meta = json.load(open(os.path.join(ROOT, "kaggle", "dataset-metadata.json")))
        meta["id"] = f"{user}/lughaat-tts-code"
        json.dump(meta, open(os.path.join(tmp, "dataset-metadata.json"), "w"), indent=2)
        r = subprocess.run(["kaggle", "datasets", "status", meta["id"]], capture_output=True, text=True)
        if "ready" in (r.stdout + r.stderr).lower():
            sh(["kaggle", "datasets", "version", "-p", tmp, "-m", "update", "-r", "skip", "-q"])
        else:
            sh(["kaggle", "datasets", "create", "-p", tmp, "-r", "skip", "-q"])
        shutil.rmtree(tmp, ignore_errors=True)
        print(f"dataset: https://www.kaggle.com/datasets/{user}/lughaat-tts-code")
    if a.kernel:
        kd = os.path.join(ROOT, "kaggle")
        meta = json.load(open(os.path.join(kd, "kernel-metadata.json")))
        meta["id"] = f"{user}/lughaat-tts-train"
        meta["dataset_sources"] = [f"{user}/lughaat-tts-code"]
        meta["machine_shape"] = a.accelerator
        json.dump(meta, open(os.path.join(kd, "kernel-metadata.json"), "w"), indent=2)
        cmd = ["kaggle", "kernels", "push", "-p", kd, "--accelerator", a.accelerator, "-t", str(a.timeout)]
        if a.no_run:
            cmd.append("--no-run")
        sh(cmd)
        print(f"kernel: https://www.kaggle.com/code/{user}/lughaat-tts-train")
        if a.no_run:
            print("NOW: open the kernel editor -> Add-ons -> Secrets -> add HF_TOKEN (write) and attach it; then push again without --no-run")


if __name__ == "__main__":
    main()

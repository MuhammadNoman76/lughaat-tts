#!/usr/bin/env python3
"""Phase 9: upload the model repo (and the Gradio Space) with huggingface_hub (plan 11.2).

    python scripts/99_upload.py --export export/v1.0 --model-repo USER/lughaat-tts-82m \
        --space-repo USER/lughaat-tts-demo --private 1 [--revision v0.1-stage1] [--no-space]

Uploads: lughaat-tts-82m.pth, config.json, voices/, onnx/, samples/, diffusion/, README.md,
REPORT.md, DECISIONS.md, pyproject.toml and the lughaat_tts/ package (with DEFAULT_REPO_ID
rewritten to the model repo so `UrduPipeline()` needs no arguments).
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def stage_package(tmp: str, model_repo: str) -> None:
    pkg_src = os.path.join(ROOT, "lughaat_tts")
    pkg_dst = os.path.join(tmp, "lughaat_tts")
    shutil.copytree(pkg_src, pkg_dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    rp = os.path.join(pkg_dst, "_repo.py")
    s = open(rp, encoding="utf-8").read()
    s = re.sub(r'DEFAULT_REPO_ID = ".*?"', f'DEFAULT_REPO_ID = "{model_repo}"', s)
    open(rp, "w", encoding="utf-8").write(s)
    shutil.copy(os.path.join(ROOT, "pyproject.toml"), tmp)
    for fn in ("DECISIONS.md", "LICENSE", "NOTICE"):
        if os.path.exists(os.path.join(ROOT, fn)):
            shutil.copy(os.path.join(ROOT, fn), tmp)
    # setup.py shim so very old pip can install from the HF git URL too
    open(os.path.join(tmp, "setup.py"), "w").write("from setuptools import setup\nsetup()\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--model-repo", required=True)
    ap.add_argument("--space-repo", default=None)
    ap.add_argument("--work-repo", default=None)
    ap.add_argument("--private", default="1")
    ap.add_argument("--revision", default=None, help="upload to a branch/revision (e.g. v0.1-stage1)")
    ap.add_argument("--no-space", action="store_true")
    a = ap.parse_args()
    from huggingface_hub import HfApi
    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    private = str(a.private) not in ("0", "false", "False")
    api.create_repo(a.model_repo, repo_type="model", private=private, exist_ok=True)
    if a.revision:
        try:
            api.create_branch(a.model_repo, branch=a.revision, exist_ok=True)
        except Exception as e:
            print(f"branch: {e!r}")
    tmp = tempfile.mkdtemp(prefix="lt_upload_")
    stage_package(tmp, a.model_repo)
    for fn in ("lughaat-tts-82m.pth", "config.json", "README.md", "REPORT.md", "export_report.json"):
        p = os.path.join(a.export, fn)
        if os.path.exists(p):
            shutil.copy(p, tmp)
    for d in ("voices", "onnx", "samples", "diffusion"):
        p = os.path.join(a.export, d)
        if os.path.isdir(p):
            shutil.copytree(p, os.path.join(tmp, d))
    if not os.path.exists(os.path.join(tmp, "README.md")):
        open(os.path.join(tmp, "README.md"), "w", encoding="utf-8").write(
            f"---\nlicense: apache-2.0\nlanguage: [ur, en]\nbase_model: hexgrad/Kokoro-82M\npipeline_tag: text-to-speech\ntags: [kokoro, styletts2, urdu, tts]\n---\n# Kokoro-82M Urdu ({a.revision or 'work in progress'})\n")
    print(f"uploading {tmp} -> {a.model_repo} ({a.revision or 'main'}) ...")
    api.upload_large_folder(repo_id=a.model_repo, repo_type="model", folder_path=tmp, revision=a.revision) if hasattr(api, "upload_large_folder") and a.revision is None else \
        api.upload_folder(repo_id=a.model_repo, repo_type="model", folder_path=tmp, revision=a.revision)
    print(f"model: https://huggingface.co/{a.model_repo}" + (f"/tree/{a.revision}" if a.revision else ""))

    if a.space_repo and not a.no_space:
        api.create_repo(a.space_repo, repo_type="space", space_sdk="gradio", private=private, exist_ok=True)
        space_dir = os.path.join(ROOT, "space")
        stmp = tempfile.mkdtemp(prefix="lt_space_")
        for fn in os.listdir(space_dir):
            shutil.copy(os.path.join(space_dir, fn), stmp)
        req = open(os.path.join(stmp, "requirements.txt"), encoding="utf-8").read().replace("{MODEL_REPO}", a.model_repo)
        open(os.path.join(stmp, "requirements.txt"), "w", encoding="utf-8").write(req)
        app = open(os.path.join(stmp, "app.py"), encoding="utf-8").read().replace("{MODEL_REPO}", a.model_repo)
        open(os.path.join(stmp, "app.py"), "w", encoding="utf-8").write(app)
        api.upload_folder(repo_id=a.space_repo, repo_type="space", folder_path=stmp)
        if token and private:
            try:
                api.add_space_secret(a.space_repo, "HF_TOKEN", token)
            except Exception as e:
                print(f"could not set the Space secret HF_TOKEN automatically: {e!r}")
        try:
            api.add_space_variable(a.space_repo, "LUGHAAT_TTS_REPO", a.model_repo)
        except Exception:
            pass
        print(f"space: https://huggingface.co/spaces/{a.space_repo}")
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()

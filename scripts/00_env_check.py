#!/usr/bin/env python3
"""Check inputs, GPU, HF write access and gated-dataset access; write env_check.md (plan Appendix C.1).

    HF_TOKEN=hf_xxx python scripts/00_env_check.py [--hf-username USER] [--out env_check.md]
Exit code 0 = ready; 2 = a STOP condition (missing token / no Rasa access / no write scope).
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
import time


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-username", default=os.environ.get("HF_USERNAME"))
    ap.add_argument("--out", default="env_check.md")
    ap.add_argument("--compute-mode", default=os.environ.get("COMPUTE_MODE", "kaggle"))
    a = ap.parse_args()
    lines = [f"# env_check ({time.strftime('%Y-%m-%d %H:%M:%S')})", ""]
    stop = []
    token = os.environ.get("HF_TOKEN")
    if not token:
        try:
            from kaggle_secrets import UserSecretsClient  # type: ignore
            token = UserSecretsClient().get_secret("HF_TOKEN")
        except Exception:
            token = None
    if not token:
        stop.append("HF_TOKEN missing (write scope). Kaggle: Add-ons -> Secrets -> HF_TOKEN.")
        lines.append("- HF_TOKEN: MISSING")
    else:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        try:
            who = api.whoami()
            user = a.hf_username or who["name"]
            lines.append(f"- HF token owner: {who['name']}; repos will be created under: {user}")
            role = (who.get("auth") or {}).get("accessToken", {}).get("role")
            if role and role not in ("write", "fineGrained"):
                stop.append(f"HF token role is '{role}', needs write scope")
            try:
                api.create_repo(f"{user}/lughaat-tts-work", repo_type="dataset", private=True, exist_ok=True)
                lines.append(f"- HF write access: OK (dataset {user}/lughaat-tts-work)")
            except Exception as e:
                stop.append(f"HF write failed: {e!r}")
            try:
                api.get_paths_info("ai4bharat/Rasa", ["Urdu/test-00000-of-00007.parquet"], repo_type="dataset")
                lines.append("- ai4bharat/Rasa (gated): access OK")
            except Exception as e:
                stop.append(f"Rasa access denied: open https://huggingface.co/datasets/ai4bharat/Rasa as {who['name']} and click 'Agree and access' ({e!r})")
        except Exception as e:
            stop.append(f"HF token invalid: {e!r}")
    try:
        import torch
        n = torch.cuda.device_count()
        lines.append(f"- GPUs: {n} x {torch.cuda.get_device_name(0) if n else 'none'}; torch {torch.__version__}")
        if a.compute_mode == "kaggle" and n == 0:
            stop.append("No GPU visible: set Accelerator = GPU T4 x2")
        if a.compute_mode == "gpu" and n and torch.cuda.get_device_properties(0).total_memory < 70e9:
            stop.append("gpu mode needs an 80 GB GPU for Stage 2 (plan 12)")
    except Exception as e:
        lines.append(f"- torch: not importable ({e!r})")
    for mod in ("kokoro", "misaki", "datasets", "soundfile", "librosa", "soxr", "transformers", "accelerate", "speechmos", "onnxruntime"):
        try:
            importlib.import_module(mod)
            lines.append(f"- {mod}: ok")
        except Exception:
            lines.append(f"- {mod}: missing (pip install -r requirements-kaggle.txt)")
    if a.compute_mode == "kaggle":
        for k in ("KAGGLE_USERNAME", "KAGGLE_KEY"):
            lines.append(f"- {k}: {'set' if os.environ.get(k) else 'not set (only needed for the orchestrator / kaggle_push.py)'}")
    lines += ["", "## STOP conditions", ""] + ([f"- {s}" for s in stop] or ["- none"])
    text = "\n".join(lines) + "\n"
    open(a.out, "w", encoding="utf-8").write(text)
    print(text)
    return 2 if stop else 0


if __name__ == "__main__":
    sys.exit(main())

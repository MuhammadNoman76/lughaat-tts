#!/usr/bin/env python3
"""Hourly orchestrator (GitHub Actions cron) that keeps Kaggle sessions running until done (plan 14.2.5).

    python orchestrator/orchestrator.py --kernel-dir kaggle

Env: KAGGLE_USERNAME, KAGGLE_KEY, HF_TOKEN, optional HF_USERNAME, KAGGLE_KERNEL (user/lughaat-tts-train),
ACCELERATOR (NvidiaTeslaT4 | NvidiaL4), GH_TOKEN + GH_REPO for the progress issue.

Logic per run
    status = kaggle kernels status <kernel>;  running/queued -> exit
    state  = state.json from the HF work repo; done -> exit (notify once)
    last_error repeated 3x on the same unit -> open a GitHub issue, exit
    quota  = kaggle quota -v; remaining < 1.5 h -> exit (wait for refreshAt)
    push the notebook with timeout = min(11.5 h, remaining - 0.25 h); append progress.md
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import subprocess
import sys
import time


def sh(cmd: list[str], check: bool = False) -> subprocess.CompletedProcess:
    print("$ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


def kernel_status(kernel: str) -> str:
    r = sh(["kaggle", "kernels", "status", kernel])
    out = (r.stdout + r.stderr).lower()
    for s in ("running", "queued", "complete", "error", "cancel"):
        if s in out:
            return s
    return "unknown"


def gpu_quota() -> tuple[float | None, str]:
    r = sh(["kaggle", "quota", "-v"])
    if r.returncode != 0 or "resource" not in r.stdout:
        return None, ""
    for row in csv.DictReader(io.StringIO(r.stdout)):
        if row.get("resource", "").upper() == "GPU":
            try:
                return float(str(row.get("remaining", "0")).rstrip("h")), row.get("refreshAt", "")
            except ValueError:
                return None, row.get("refreshAt", "")
    return None, ""


def read_state(work_repo: str, token: str) -> dict:
    from huggingface_hub import hf_hub_download
    try:
        p = hf_hub_download(work_repo, "state.json", repo_type="dataset", token=token, local_dir="/tmp/lt_state")
        return json.load(open(p, encoding="utf-8"))
    except Exception as e:
        print(f"no state.json yet ({e!r})")
        return {}


def gh_issue(title: str, body: str) -> None:
    if not os.environ.get("GH_TOKEN") or not os.environ.get("GH_REPO"):
        return
    r = sh(["gh", "issue", "list", "--search", f"in:title {title}", "--json", "number,title", "--state", "open"])
    try:
        issues = json.loads(r.stdout or "[]")
    except Exception:
        issues = []
    if issues:
        sh(["gh", "issue", "comment", str(issues[0]["number"]), "--body", body])
    else:
        sh(["gh", "issue", "create", "--title", title, "--body", body])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel-dir", default="kaggle")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    token = os.environ["HF_TOKEN"]
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    user = os.environ.get("HF_USERNAME") or api.whoami()["name"]
    work_repo = f"{user}/lughaat-tts-work"
    kernel = os.environ.get("KAGGLE_KERNEL") or f"{os.environ['KAGGLE_USERNAME']}/lughaat-tts-train"

    status = kernel_status(kernel)
    print(f"kernel {kernel}: {status}")
    if status in ("running", "queued"):
        return
    state = read_state(work_repo, token)
    phase = state.get("phase", "env_check")
    if state.get("done"):
        print("training finished")
        gh_issue("Training progress", f"✅ DONE. Phase {phase}. Final metrics: {json.dumps(state.get('final_metrics', {}), default=str)[:1500]}")
        return
    err = state.get("last_error") or {}
    if err.get("count", 0) >= 3:
        gh_issue("Training needs attention", f"Unit `{err.get('unit')}` failed {err.get('count')} times: `{err.get('error')}`\nSee logs/ in https://huggingface.co/datasets/{work_repo}")
        print("repeated failure; not pushing")
        return
    if state.get("stop_reason") and "Agree and access" in state["stop_reason"]:
        gh_issue("Training needs attention", state["stop_reason"])
        return

    remaining, refresh = gpu_quota()
    print(f"GPU quota remaining: {remaining} h (refresh {refresh})")
    if remaining is not None and remaining < 1.5:
        print("waiting for the weekly quota")
        return
    budget_h = 11.5 if remaining is None else min(11.5, remaining - 0.25)
    timeout = int(budget_h * 3600)
    if a.dry_run:
        print(f"would push {a.kernel_dir} with timeout {timeout}s")
        return
    # the session's own budget must be shorter than the Kaggle timeout
    os.environ["SESSION_BUDGET_MIN"] = str(int(budget_h * 60 - 20))
    acc = os.environ.get("ACCELERATOR", "NvidiaTeslaT4")
    # fill the Kaggle username into the kernel metadata (the repo copy carries a placeholder)
    meta_path = os.path.join(a.kernel_dir, "kernel-metadata.json")
    meta = json.load(open(meta_path))
    ku = os.environ["KAGGLE_USERNAME"]
    meta["id"] = kernel
    meta["dataset_sources"] = [s.replace("KAGGLE_USERNAME", ku) for s in meta.get("dataset_sources", [])]
    meta["machine_shape"] = acc
    json.dump(meta, open(meta_path, "w"), indent=2)
    r = sh(["kaggle", "kernels", "push", "-p", a.kernel_dir, "--accelerator", acc, "-t", str(timeout)])
    print(r.stdout, r.stderr)
    line = f"- {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())} pushed session (phase {phase}, quota {remaining} h, timeout {budget_h:.1f} h)\n"
    with open("progress.md", "a", encoding="utf-8") as f:
        f.write(line)
    s1 = state.get("stage1", {}); s2 = state.get("stage2", {})
    gh_issue("Training progress", f"Phase **{phase}** | Stage1 epoch {s1.get('epoch')} shard {s1.get('shard')}/{s1.get('num_shards')} | "
             f"Stage2 epoch {s2.get('epoch')} shard {s2.get('shard')}/{s2.get('num_shards')} | last epoch metrics: "
             f"{json.dumps((s2.get('epochs') or s1.get('epochs') or [{}])[-1], default=str)[:600]} | projection: {s1.get('projection') or s2.get('projection')}")


if __name__ == "__main__":
    main()

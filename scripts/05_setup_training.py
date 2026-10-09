#!/usr/bin/env python3
"""Phase 4: training environment (plan 6, 14.7).

    python scripts/05_setup_training.py --work /kaggle/tmp/work --data /kaggle/tmp/data [--smoke]

1. apt: espeak-ng libespeak-ng1 libsndfile1 (best effort).
2. Clone kikiri-tts at the pinned commit (with submodules), apply training/patches.py,
   install the 178-token symbol map.
3. Create a uv venv with the pinned torch 2.4.1 stack (the Kannada-proven versions) unless
   --system-python is given; build monotonic_align.
4. Convert hexgrad/Kokoro-82M -> kokoro_base.pth; pre-download microsoft/wavlm-base-plus.
5. --smoke: 2 Stage-1 steps on 8 clips; all losses must be finite (NaN mel = wrong symbol map).
Writes <work>/setup.json with the resolved paths; later scripts read it.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from training.patches import apply_all, KIKIRI_SHA, STYLETTS2_SHA, KOKORO_SHA  # noqa: E402

KIKIRI_URL = "https://github.com/semidark/kikiri-tts"
PINNED = ("torch==2.4.1", "torchaudio==2.4.1", "torchvision==0.19.1", "numpy<2", "accelerate==0.34.2",
          "transformers==4.49.0", "phonemizer==3.3.0", "munch", "einops", "einops-exts", "pyyaml", "soundfile",
          "librosa", "scipy", "tqdm", "loguru", "tensorboard", "Cython", "nltk", "pydub", "matplotlib==3.7.5",
          "pandas", "huggingface_hub", "requests", "click", "git+https://github.com/resemble-ai/monotonic_align.git")


def log(m: str) -> None:
    print(f"[setup {time.strftime('%H:%M:%S')}] {m}", flush=True)


def sh(cmd: str, check: bool = True, **kw) -> subprocess.CompletedProcess:
    log("$ " + cmd)
    return subprocess.run(cmd, shell=True, check=check, **kw)


def clone(work: str) -> str:
    kk = os.path.join(work, "kikiri-tts")
    if not os.path.exists(os.path.join(kk, "StyleTTS2", "train_first.py")):
        shutil.rmtree(kk, ignore_errors=True)
        sh(f"git clone -q {KIKIRI_URL} {kk}")
        sh(f"git -C {kk} checkout -q {KIKIRI_SHA}")
        sh(f"git -C {kk} submodule update --init --recursive -q")
        sh(f"git -C {kk}/StyleTTS2 checkout -q {STYLETTS2_SHA}", check=False)
        sh(f"git -C {kk}/kokoro checkout -q {KOKORO_SHA}", check=False)
    for sub in ("StyleTTS2", "kokoro"):
        assert os.listdir(os.path.join(kk, sub)), f"submodule {sub} is empty"
    for f in ("Utils/JDC/bst.t7", "Utils/ASR/config.yml", "Utils/ASR/epoch_00080.pth", "Utils/PLBERT/step_1000000.t7"):
        p = os.path.join(kk, "StyleTTS2", f)
        assert os.path.exists(p), f"missing utility model {f}; fetch it from yl4579/StyleTTS2"
    return kk


def make_venv(work: str, system_python: bool) -> str:
    if system_python:
        return sys.executable
    venv = os.path.join(work, "venv")
    py = os.path.join(venv, "bin", "python")
    if not os.path.exists(py):
        sh("pip install -q uv 2>&1 | tail -1", check=False)
        sh(f"uv venv {venv} --python 3.12 -q || uv venv {venv} --python 3.11 -q")
        pk = " ".join(f"'{p}'" for p in PINNED)
        sh(f"uv pip install --python {py} -q {pk}")
        # force IPv4 inside the venv (pitfall 13)
        for sp in glob.glob(os.path.join(venv, "lib", "python*", "site-packages")):
            open(os.path.join(sp, "usercustomize.py"), "w").write(
                "import socket as _s\n_o=_s.getaddrinfo\n_s.getaddrinfo=lambda h,p,f=0,*a,**k:_o(h,p,_s.AF_INET,*a,**k)\n")
    return py


def build_monotonic_align(py: str, kk: str) -> None:
    r = subprocess.run([py, "-c", "import monotonic_align, monotonic_align.core; print('ok')"], capture_output=True, text=True)
    if "ok" in r.stdout:
        log("monotonic_align importable")
        return
    ma = os.path.join(kk, "StyleTTS2", "monotonic_align")
    if not os.path.isdir(ma):
        sh(f"git clone -q https://github.com/resemble-ai/monotonic_align.git {ma}")
    sh(f"cd {ma} && {py} setup.py build_ext --inplace", check=False)
    sh(f"{py} -m pip install -q {ma}", check=False)


def convert_base(py: str, work: str) -> str:
    out = os.path.join(work, "kokoro_base.pth")
    if not os.path.exists(out):
        sh(f"{py} {os.path.join(ROOT, 'training', 'convert_base.py')} --out {out}")
    return out


def fetch_wavlm(work: str) -> str:
    d = os.path.join(work, "wavlm-base-plus")
    if not os.path.exists(os.path.join(d, "config.json")):
        from huggingface_hub import snapshot_download
        snapshot_download("microsoft/wavlm-base-plus", local_dir=d, allow_patterns=["*.json", "*.bin", "*.safetensors", "*.txt"])
    return d


def write_accelerate_config(work: str, n_gpu: int, mixed: str = "fp16") -> str:
    p = os.path.join(work, "accelerate_config.yaml")
    with open(p, "w") as f:
        f.write("compute_environment: LOCAL_MACHINE\n")
        f.write(f"distributed_type: {'MULTI_GPU' if n_gpu > 1 else 'NO'}\n")
        f.write(f"num_processes: {n_gpu}\nmixed_precision: {mixed}\nmachine_rank: 0\nnum_machines: 1\n")
        f.write("gpu_ids: all\nmain_training_function: main\nrdzv_backend: static\nsame_network: true\nuse_cpu: false\ndowncast_bf16: 'no'\n")
    return p


def smoke_test(py: str, kk: str, work: str, data: str, base: str, wavlm: str, n_gpu: int) -> dict:
    """2 Stage-1 steps on 8 clips. Returns parsed losses."""
    import yaml
    from training.kokoro_config import build_config
    from training.shard import read_list, write_list
    st2 = os.path.join(kk, "StyleTTS2")
    lines = read_list(os.path.join(data, "train_list.txt"))
    smoke_list = write_list(os.path.join(work, "smoke_list.txt"), lines[:16])
    val_list = write_list(os.path.join(work, "smoke_val.txt"), lines[16:20])
    log_dir = os.path.join(work, "logs_smoke")
    shutil.rmtree(log_dir, ignore_errors=True)
    cfg = build_config(log_dir, os.path.join(data, "audio"), smoke_list, val_list, os.path.join(data, "OOD_texts.txt"), base,
                       epochs_1st=1, batch_size=4 if n_gpu < 2 else 8, max_len=400, slm_model_dir=wavlm, num_workers=2, log_interval=1)
    cfg_path = os.path.join(work, "config_smoke.yml")
    yaml.safe_dump(cfg, open(cfg_path, "w"), allow_unicode=True, sort_keys=False)
    acc = write_accelerate_config(work, n_gpu, "fp16")
    env = {**os.environ, "PYTHONPATH": st2, "WANDB_DISABLED": "true", "TOKENIZERS_PARALLELISM": "false"}
    acc_bin = os.path.join(os.path.dirname(py), "accelerate")
    cmd = [acc_bin, "launch", "--config_file", acc, "train_first.py", "--config_path", cfg_path]
    log("smoke: " + " ".join(cmd))
    r = subprocess.run(cmd, cwd=st2, env=env, capture_output=True, text=True, timeout=1800)
    out = r.stdout + r.stderr
    open(os.path.join(work, "smoke.log"), "w").write(out)
    m = re.findall(r"Mel Loss: ([0-9.naninf]+), Gen Loss: ([0-9.naninf\-]+), Disc Loss: ([0-9.naninf\-]+), Mono Loss: ([0-9.naninf\-]+), S2S Loss: ([0-9.naninf\-]+), SLM Loss: ([0-9.naninf\-]+)", out)
    res = {"returncode": r.returncode, "steps_logged": len(m)}
    if m:
        last = m[-1]
        res.update({"mel": float(last[0]), "gen": float(last[1]), "disc": float(last[2]), "mono": float(last[3]), "s2s": float(last[4]), "slm": float(last[5])})
        res["finite"] = all(str(v) not in ("nan", "inf") and v == v for v in [res["mel"], res["gen"], res["disc"], res["mono"], res["s2s"], res["slm"]])
    ckpts = glob.glob(os.path.join(log_dir, "epoch_1st_*.pth")) + glob.glob(os.path.join(log_dir, "first_stage.pth"))
    res["checkpoint_saved"] = bool(ckpts)
    if r.returncode != 0 or not m:
        tail = "\n".join(out.splitlines()[-40:])
        log(f"smoke test FAILED (rc={r.returncode}); log tail:\n{tail}")
    else:
        log(f"smoke test: {res}")
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--data", required=True, help="dir with audio/, train_list.txt, val_list.txt, OOD_texts.txt")
    ap.add_argument("--system-python", action="store_true", help="use the current interpreter instead of a pinned venv")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    sh("apt-get -qq install -y espeak-ng libespeak-ng1 libsndfile1 >/dev/null 2>&1 || true", check=False)
    kk = clone(a.work)
    apply_all(kk, log)
    py = make_venv(a.work, a.system_python)
    build_monotonic_align(py, kk)
    base = convert_base(py, a.work)
    wavlm = fetch_wavlm(a.work)
    import torch
    n_gpu = torch.cuda.device_count()
    gpu = torch.cuda.get_device_name(0) if n_gpu else "cpu"
    info = {"kikiri": kk, "styletts2": os.path.join(kk, "StyleTTS2"), "python": py, "base": base, "wavlm": wavlm,
            "n_gpu": n_gpu, "gpu": gpu, "accelerate_config": write_accelerate_config(a.work, max(n_gpu, 1))}
    if a.smoke:
        info["smoke"] = smoke_test(py, kk, a.work, a.data, base, wavlm, n_gpu)
        ok = info["smoke"].get("finite", False) and info["smoke"].get("checkpoint_saved", False)
        if not ok:
            with open(os.path.join(a.work, "setup.json"), "w") as f:
                json.dump(info, f, indent=2)
            raise SystemExit("SMOKE TEST FAILED: non-finite losses or no checkpoint. Check the symbol map / data (plan 6.1).")
        mel = info["smoke"].get("mel", 9)
        if not (0.3 <= mel <= 3.0):
            log(f"WARNING: first-step mel loss {mel} outside the expected 0.8-1.5 range")
    with open(os.path.join(a.work, "setup.json"), "w") as f:
        json.dump(info, f, indent=2)
    log(f"setup complete: {info}")


if __name__ == "__main__":
    main()

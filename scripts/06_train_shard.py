#!/usr/bin/env python3
"""Run ONE training shard (Stage 1 or Stage 2) and return the checkpoint (plan 7, 8, 14.2.4).

    python scripts/06_train_shard.py --stage 1 --epoch 0 --shard 0 --num-shards 8 \
        --work /kaggle/tmp/work --data /kaggle/tmp/data --resume /kaggle/tmp/work/latest_s1.pth \
        --variant s1_fp16_b8_ml400 --out-json /kaggle/tmp/work/shard_result.json

Semantics
---------
* The trainer runs exactly one of its "epochs" over the shard list (epochs_X = epoch+1,
  start epoch = `epoch` taken from the resumed checkpoint), so the adversarial/diffusion
  gating (TMA_epoch, diff_epoch, joint_epoch) sees the LOGICAL epoch number.
* First Stage-1 shard: loads kokoro_base.pth with load_only_params. First Stage-2 shard:
  loads first_stage.pth through the trainer's own path (predictor_encoder initialised
  from style_encoder). Every other shard: full resume (weights + optimizer) from --resume,
  with the checkpoint's `epoch` rewritten to the logical epoch.
* A hang watchdog kills the trainer when the log stops advancing for --hang-minutes
  (the 40 GB-class silent OOM deadlock, pitfall 1) and reports status "hang".
* --probe-steps N: memory probe; stops after the shard list (sized to N steps) and reports
  peak GPU memory; status "oom" on CUDA OOM.
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
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from training.kokoro_config import build_config, STAGE1_VARIANTS, STAGE2_VARIANTS, STAGE1_LR, STAGE2_LR  # noqa: E402
from training.shard import read_list, write_list, shard_lines  # noqa: E402


def log(m: str) -> None:
    print(f"[shard {time.strftime('%H:%M:%S')}] {m}", flush=True)


def _rewrite_epoch(src: str, dst: str, epoch: int) -> None:
    import torch
    st = torch.load(src, map_location="cpu", weights_only=False)
    st["epoch"] = epoch
    torch.save(st, dst)


def run_trainer(cmd: list[str], cwd: str, env: dict, log_path: str, hang_minutes: float, timeout_s: float) -> tuple[int, str]:
    """Run, tee to log_path, kill on hang. Returns (returncode, status)."""
    lf = open(log_path, "a", encoding="utf-8", errors="replace")
    proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace", bufsize=1)
    last = [time.time()]
    status = ["ok"]

    def reader():
        for line in proc.stdout:
            lf.write(line)
            lf.flush()
            last[0] = time.time()
            if "CUDA out of memory" in line or "OutOfMemoryError" in line:
                status[0] = "oom"
            print(line.rstrip()[:300], flush=True)

    t = threading.Thread(target=reader, daemon=True)
    t.start()
    t0 = time.time()
    while proc.poll() is None:
        time.sleep(10)
        if time.time() - last[0] > hang_minutes * 60:
            status[0] = "hang"
            log(f"no log progress for {hang_minutes} min: killing (silent OOM deadlock?)")
            proc.kill()
            break
        if time.time() - t0 > timeout_s:
            status[0] = "timeout"
            log("shard timeout: killing")
            proc.kill()
            break
    proc.wait()
    t.join(timeout=5)
    lf.close()
    rc = proc.returncode
    if rc != 0 and status[0] == "ok":
        status[0] = "error"
    return rc, status[0]


def parse_losses(log_text: str, stage: int) -> dict:
    out = {}
    if stage == 1:
        m = re.findall(r"Mel Loss: ([0-9.]+), Gen Loss: ([0-9.\-]+), Disc Loss: ([0-9.\-]+), Mono Loss: ([0-9.\-]+)", log_text)
        if m:
            mel = [float(x[0]) for x in m]
            out = {"mel_first": mel[0], "mel_last": mel[-1], "mel_mean": sum(mel) / len(mel), "mono_last": float(m[-1][3]), "steps_logged": len(m)}
    else:
        m = re.findall(r"Loss: ([0-9.]+), Disc Loss: ([0-9.\-]+), Dur Loss: ([0-9.\-]+), CE Loss: ([0-9.\-]+), Norm Loss: ([0-9.\-]+), F0 Loss: ([0-9.\-]+), LM Loss: ([0-9.\-]+), Gen Loss: ([0-9.\-]+), Sty Loss: ([0-9.\-]+), Diff Loss: ([0-9.\-]+)", log_text)
        if m:
            mel = [float(x[0]) for x in m]
            out = {"mel_first": mel[0], "mel_last": mel[-1], "mel_mean": sum(mel) / len(mel), "dur_last": float(m[-1][2]),
                   "f0_last": float(m[-1][5]), "diff_last": float(m[-1][9]), "steps_logged": len(m)}
    v = re.findall(r"Validation loss: ([0-9.]+)", log_text)
    if v:
        out["val_mel"] = float(v[-1])
    out["aligner_exc"] = log_text.count("ALIGNER-EXC")
    out["slmadv_none"] = log_text.count("SLMADV-NONE")
    out["step_exc"] = log_text.count("STEP-EXC")
    out["nan_skip"] = log_text.count("NAN-SKIP")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, required=True, choices=[1, 2])
    ap.add_argument("--epoch", type=int, required=True)
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--num-shards", type=int, required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--resume", default="", help="checkpoint to resume from (full state); empty = fresh")
    ap.add_argument("--first-stage", default="", help="Stage 2 only: first_stage.pth for the very first shard")
    ap.add_argument("--variant", required=True)
    ap.add_argument("--shard-list", default=None, help="explicit list file (overrides epoch/shard slicing)")
    ap.add_argument("--probe-steps", type=int, default=0)
    ap.add_argument("--hang-minutes", type=float, default=25.0)
    ap.add_argument("--timeout-minutes", type=float, default=150.0)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()

    import yaml
    setup = json.load(open(os.path.join(a.work, "setup.json")))
    st2, py, base, wavlm = setup["styletts2"], setup["python"], setup["base"], setup["wavlm"]
    n_gpu = int(setup.get("n_gpu", 1))
    variants = {v["name"]: v for v in (STAGE1_VARIANTS if a.stage == 1 else STAGE2_VARIANTS)}
    var = variants[a.variant]

    # shard list
    train_list = os.path.join(a.data, "train_list.txt")
    lines = read_list(train_list)
    if a.shard_list:
        shard_path = a.shard_list
    elif a.probe_steps:
        shard_path = write_list(os.path.join(a.work, "shards", f"probe_s{a.stage}.txt"), shard_lines(lines, 0, 0, max(1, len(lines) // (a.probe_steps * var["batch_size"])), a.seed))
    else:
        shard_path = write_list(os.path.join(a.work, "shards", f"s{a.stage}_e{a.epoch:02d}_sh{a.shard:02d}.txt"),
                                shard_lines(lines, a.epoch, a.shard, a.num_shards, a.seed))
    n_lines = len(read_list(shard_path))
    log_dir = os.path.join(a.work, f"logs_s{a.stage}")
    os.makedirs(log_dir, exist_ok=True)
    for old in glob.glob(os.path.join(log_dir, "epoch_*st_*.pth")) + glob.glob(os.path.join(log_dir, "epoch_2nd_*.pth")):
        os.remove(old)

    lr = STAGE1_LR if a.stage == 1 else STAGE2_LR
    common = dict(log_dir=log_dir, data_root=os.path.join(a.data, "audio"), train_list=shard_path,
                  val_list=os.path.join(a.data, "val_list.txt"), ood_list=os.path.join(a.data, "OOD_texts.txt"),
                  batch_size=var["batch_size"], max_len=var["max_len"], slm_model_dir=wavlm, num_workers=4, save_freq=1,
                  tb_inference=False, lr=lr["lr"], bert_lr=lr["bert_lr"], ft_lr=lr["ft_lr"])
    resume_path = ""
    if a.resume:
        resume_path = os.path.join(a.work, f"resume_s{a.stage}.pth")
        _rewrite_epoch(a.resume, resume_path, a.epoch)
    if a.stage == 1:
        cfg = build_config(pretrained_model=resume_path or base, epochs_1st=a.epoch + 1, load_only_params=not bool(resume_path), **common)
        if resume_path:
            cfg["load_only_params"] = False
    else:
        if resume_path:
            cfg = build_config(pretrained_model=resume_path, epochs_2nd=a.epoch + 1, resume_ckpt=resume_path,
                               joint_epoch=var["joint_epoch"], diff_epoch=1, lambda_diff=1.0, lambda_sty=1.0, lambda_slm=0.2,
                               disable_slm=var["disable_slm"], init_predictor_encoder=False, **common)
        else:
            assert a.first_stage, "Stage 2 first shard needs --first-stage"
            shutil.copy(a.first_stage, os.path.join(log_dir, "first_stage.pth"))
            cfg = build_config(pretrained_model=base, epochs_2nd=1, joint_epoch=var["joint_epoch"], diff_epoch=1,
                               lambda_diff=1.0, lambda_sty=1.0, lambda_slm=0.2, disable_slm=var["disable_slm"],
                               init_predictor_encoder=True, first_stage_path="first_stage.pth", **common)
    cfg_path = os.path.join(a.work, f"config_s{a.stage}_e{a.epoch:02d}_sh{a.shard:02d}.yml")
    yaml.safe_dump(cfg, open(cfg_path, "w"), allow_unicode=True, sort_keys=False)

    env = {**os.environ, "PYTHONPATH": st2, "WANDB_DISABLED": "true", "TOKENIZERS_PARALLELISM": "false", "HF_HUB_ENABLE_HF_TRANSFER": "0"}
    env.pop("PYTORCH_CUDA_ALLOC_CONF", None)   # we WANT a loud OOM, not a silent deadlock
    if a.stage == 1:
        acc_cfg = os.path.join(a.work, f"accelerate_{var['mixed_precision']}.yaml")
        with open(acc_cfg, "w") as f:
            f.write("compute_environment: LOCAL_MACHINE\n")
            f.write(f"distributed_type: {'MULTI_GPU' if n_gpu > 1 else 'NO'}\nnum_processes: {n_gpu}\n")
            f.write(f"mixed_precision: {var['mixed_precision']}\nmachine_rank: 0\nnum_machines: 1\ngpu_ids: all\n")
            f.write("main_training_function: main\nrdzv_backend: static\nsame_network: true\nuse_cpu: false\ndowncast_bf16: 'no'\n")
        cmd = [os.path.join(os.path.dirname(py), "accelerate"), "launch", "--config_file", acc_cfg, "train_first.py", "--config_path", cfg_path]
    else:
        cmd = [py, "train_second.py", "--config_path", cfg_path]
    log_path = os.path.join(a.work, f"train_s{a.stage}_e{a.epoch:02d}_sh{a.shard:02d}{'_probe' if a.probe_steps else ''}.log")
    if os.path.exists(log_path):
        os.remove(log_path)
    t0 = time.time()
    log(f"stage {a.stage} epoch {a.epoch} shard {a.shard}/{a.num_shards} variant {a.variant} lines {n_lines} resume={'yes' if resume_path else 'no'}")
    rc, status = run_trainer(cmd, st2, env, log_path, a.hang_minutes, a.timeout_minutes * 60)
    elapsed = time.time() - t0
    text = open(log_path, encoding="utf-8", errors="replace").read()
    with open(os.path.join(a.work, f"train_s{a.stage}.log"), "a", encoding="utf-8") as cum:   # cumulative history
        cum.write(f"\n===== {os.path.basename(log_path)} {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n{text}")
    losses = parse_losses(text[-400000:], a.stage)
    pattern = "epoch_1st_*.pth" if a.stage == 1 else "epoch_2nd_*.pth"
    ckpts = sorted(glob.glob(os.path.join(log_dir, pattern)))
    if a.stage == 1 and not ckpts:
        ckpts = sorted(glob.glob(os.path.join(log_dir, "first_stage.pth")))
    out_ckpt = ""
    if ckpts and status == "ok":
        out_ckpt = os.path.join(a.work, f"latest_s{a.stage}.pth")
        shutil.move(ckpts[-1], out_ckpt)
    if resume_path and os.path.exists(resume_path):
        os.remove(resume_path)
    peak = None
    try:
        import torch
        peak = [round(torch.cuda.max_memory_allocated(i) / 1e9, 2) for i in range(torch.cuda.device_count())]
    except Exception:
        pass
    result = {"stage": a.stage, "epoch": a.epoch, "shard": a.shard, "num_shards": a.num_shards, "variant": a.variant, "lines": n_lines,
              "status": status if (out_ckpt or a.probe_steps) else ("error" if status == "ok" else status),
              "returncode": rc, "seconds": round(elapsed, 1), "checkpoint": out_ckpt, "losses": losses, "probe": bool(a.probe_steps)}
    if a.probe_steps:
        pm = re.findall(r"peak_mem_gb=([0-9.]+)", text)
        result["peak_mem_gb"] = float(pm[-1]) if pm else peak
        result["status"] = status if status != "ok" or losses.get("steps_logged", 0) > 0 else "error"
    with open(a.out_json, "w") as f:
        json.dump(result, f, indent=2)
    log(f"result: {result}")


if __name__ == "__main__":
    main()

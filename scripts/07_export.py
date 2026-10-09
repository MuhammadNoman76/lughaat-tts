#!/usr/bin/env python3
"""Phase 8.1-8.3: export a StyleTTS2 checkpoint as Kokoro KModel weights + config.json (plan 10).

    python scripts/07_export.py --ckpt work/best_s2.pth --out export/ [--kikiri work/kikiri-tts]

* Takes bert, bert_encoder, predictor, text_encoder, decoder; strips `module.`; saves
  `lughaat-tts-82m.pth` WITHOUT the DataParallel prefix so KModel's strict load path is used.
* Verifies: builds KModel(config, model) and asserts matched == total tensors for every module.
* config.json: base Kokoro config unchanged + "urdu_frontend_version" + "languages": ["ur","en"].
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from lughaat_tts.version import FRONTEND_VERSION  # noqa: E402

COMPONENTS = ("bert", "bert_encoder", "predictor", "text_encoder", "decoder")
BASE_CONFIG = os.path.join(ROOT, "training", "base_config.json")


def export_weights(ckpt_path: str, out_pth: str) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    net = ckpt["net"]
    out, counts = {}, {}
    for key in COMPONENTS:
        assert key in net, f"{key} missing from checkpoint"
        sd = {(k[7:] if k.startswith("module.") else k): v.detach().cpu().float() if v.is_floating_point() else v.detach().cpu()
              for k, v in net[key].items()}
        out[key] = sd
        counts[key] = len(sd)
    os.makedirs(os.path.dirname(out_pth) or ".", exist_ok=True)
    torch.save(out, out_pth)
    size_mb = os.path.getsize(out_pth) / 1e6
    return {"tensors": counts, "size_mb": round(size_mb, 1), "epoch": ckpt.get("epoch"), "val_loss": ckpt.get("val_loss")}


def write_config(out_json: str, extra: dict | None = None) -> dict:
    with open(BASE_CONFIG, encoding="utf-8") as f:
        cfg = json.load(f)
    assert cfg["n_token"] == 178 and len(cfg["vocab"]) == 114
    cfg["urdu_frontend_version"] = FRONTEND_VERSION
    cfg["languages"] = ["ur", "en"]
    if extra:
        cfg.update(extra)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    return cfg


def verify(config_path: str, model_path: str) -> dict:
    """Strict per-module verification: every tensor in the file must load into KModel."""
    from kokoro import KModel
    saved = torch.load(model_path, map_location="cpu", weights_only=True)
    model = KModel(repo_id="hexgrad/Kokoro-82M", config=config_path, model=model_path)
    report = {}
    # KModel's AdaIN1d wraps an InstanceNorm1d(affine=True) whose weight/bias are in no Kokoro checkpoint
    # (hexgrad/Kokoro-82M itself lacks them; StyleTTS2 uses affine=False). They stay at their init (1, 0).
    allowed_missing = re.compile(r"(^|\.)norm\.(weight|bias)$")
    for key, sd in saved.items():
        mod = getattr(model, key)
        res = mod.load_state_dict(sd, strict=False)
        matched = len(sd) - len(res.unexpected_keys)
        bad_missing = [k for k in res.missing_keys if not allowed_missing.search(k)]
        report[key] = {"matched": matched, "total": len(sd), "missing_allowed": len(res.missing_keys) - len(bad_missing),
                       "missing": len(bad_missing), "unexpected": len(res.unexpected_keys)}
        assert matched == len(sd) and not bad_missing, f"{key}: {report[key]} (missing {bad_missing[:5]}, unexpected {res.unexpected_keys[:5]})"
    n_params = sum(p.numel() for p in model.parameters())
    report["params_M"] = round(n_params / 1e6, 2)
    # a forward pass must work
    pack = torch.zeros(510, 1, 256)
    with torch.no_grad():
        audio = model("pɑːkɪstɑːn eːk xuːbsuːrət mʊlk hɛː.", pack[20], 1.0)
    report["forward_ok"] = bool(audio.numel() > 24000)
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="lughaat-tts-82m.pth")
    ap.add_argument("--extra-config", default="{}", help="JSON merged into config.json (e.g. training notes)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    pth = os.path.join(a.out, a.name)
    info = export_weights(a.ckpt, pth)
    cfg_path = os.path.join(a.out, "config.json")
    write_config(cfg_path, json.loads(a.extra_config))
    rep = verify(cfg_path, pth)
    info["verify"] = rep
    with open(os.path.join(a.out, "export_report.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()

"""Convert hexgrad/Kokoro-82M weights into the StyleTTS2 checkpoint layout (plan 6.5).

    python training/convert_base.py --out work/kokoro_base.pth

Kokoro stores {'bert': {'module.key': tensor}, ...}; StyleTTS2's load_checkpoint expects
{'net': {'bert': state_dict, ...}} without the DataParallel prefix. Set load_only_params:true
so the trainer loads with strict=False (diffusion / SLM / discriminators are new).
"""
from __future__ import annotations

import argparse
import os

import torch

COMPONENTS = ("bert", "bert_encoder", "predictor", "text_encoder", "decoder")
EXPECTED_PARAMS_M = 81.76


def convert(src: str, dst: str) -> dict:
    raw = torch.load(src, map_location="cpu", weights_only=True)
    net, counts, total = {}, {}, 0
    for comp, sd in raw.items():
        assert comp in COMPONENTS, f"unexpected component {comp}"
        cleaned = {k.removeprefix("module."): v for k, v in sd.items()}
        net[comp] = cleaned
        counts[comp] = len(cleaned)
        total += sum(v.numel() for v in cleaned.values())
    missing = [c for c in COMPONENTS if c not in net]
    assert not missing, f"missing components {missing}"
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    torch.save({"net": net, "epoch": 0, "iters": 0}, dst)
    info = {"tensors": counts, "params_M": round(total / 1e6, 2), "path": dst}
    print(f"[convert_base] {info}")
    assert abs(info["params_M"] - EXPECTED_PARAMS_M) < 1.0, info
    return info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=None, help="local kokoro-v1_0.pth (downloaded from HF if omitted)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    src = a.src
    if src is None:
        from huggingface_hub import hf_hub_download
        src = hf_hub_download("hexgrad/Kokoro-82M", "kokoro-v1_0.pth")
    convert(src, a.out)


if __name__ == "__main__":
    main()

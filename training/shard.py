"""Shard utilities for the Kaggle session runner (plan 14.2.4).

A "shard" is a slice of train_list.txt that one trainer process can finish in about
SHARD_MINUTES. One logical epoch = all K shards. Shards are deterministic given the
epoch number (a different shuffle every epoch) so that a resumed session reproduces
exactly the same slices.
"""
from __future__ import annotations

import math
import os
import random
from typing import Sequence


def read_list(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f if ln.strip()]


def write_list(path: str, lines: Sequence[str]) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    return path


def epoch_order(lines: Sequence[str], epoch: int, seed: int = 1234) -> list[str]:
    rng = random.Random(seed + epoch)
    order = list(lines)
    rng.shuffle(order)
    return order


def shard_lines(lines: Sequence[str], epoch: int, shard: int, num_shards: int, seed: int = 1234) -> list[str]:
    order = epoch_order(lines, epoch, seed)
    n = len(order)
    per = math.ceil(n / num_shards)
    return order[shard * per: (shard + 1) * per]


def calibration_lines(lines: Sequence[str], n: int = 600, seed: int = 1234) -> list[str]:
    return epoch_order(lines, 0, seed)[:n]


def num_shards_for(clips_total: int, clips_done: int, seconds_taken: float, shard_minutes: float, overhead_s: float = 240.0) -> int:
    """K such that one shard (plus start-up overhead) fits in shard_minutes."""
    per_clip = max(seconds_taken - overhead_s, 1.0) / max(clips_done, 1)
    budget = shard_minutes * 60 - overhead_s
    clips_per_shard = max(int(budget / per_clip), 50)
    return max(1, math.ceil(clips_total / clips_per_shard))


def make_shard_file(train_list: str, out_dir: str, epoch: int, shard: int, num_shards: int, seed: int = 1234) -> str:
    lines = read_list(train_list)
    sel = shard_lines(lines, epoch, shard, num_shards, seed)
    return write_list(os.path.join(out_dir, f"shard_e{epoch:02d}_s{shard:02d}.txt"), sel)


__all__ = ["read_list", "write_list", "shard_lines", "calibration_lines", "num_shards_for", "make_shard_file", "epoch_order"]

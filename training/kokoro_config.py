"""build_config(): the StyleTTS2/Kokoro training config (plan 6.9, 7, 8, 14.3).

Ported from Kokoro-Indic-Fine-Tuning/training/kokoro_config.py (the winning Kannada recipe)
and extended with the knobs the Kaggle shard runner needs. ALL critical keys are at the TOP
LEVEL of the YAML (anything nested under ``training:`` is ignored by the trainers).

Stage-2 variants (plan 14.3), probed in order by the session runner:
    s2_l4_full   : NvidiaL4 24 GB, full recipe, batch 2, max_len 300
    s2_t4_full   : T4x2, full recipe (GAN + WavLM SLM + diffusion), batch 2 (1/GPU), max_len 300
    s2_lite_a    : T4x2, GAN + diffusion, WavLM SLM adversary OFF, batch 4, max_len 400
    s2_lite_b    : T4x2, no GAN (joint_epoch 999), diffusion + duration/F0/mel only, batch 4
"""
from __future__ import annotations

import copy
from typing import Optional

SR = 24000

MODEL_PARAMS = {
    "dim_in": 64, "n_token": 178, "hidden_dim": 512, "style_dim": 128,
    "max_dur": 50, "multispeaker": False, "n_mels": 80, "dropout": 0.2,
    "n_layer": 3, "text_encoder_kernel_size": 5,
    "decoder": {"type": "istftnet", "upsample_rates": [10, 6],
                "upsample_kernel_sizes": [20, 12], "upsample_initial_channel": 512,
                "resblock_kernel_sizes": [3, 7, 11],
                "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                "gen_istft_n_fft": 20, "gen_istft_hop_size": 5},
    "diffusion": {"embedding_mask_proba": 0.1,
                  "transformer": {"num_layers": 3, "num_heads": 8, "head_features": 64, "multiplier": 2},
                  "dist": {"sigma_data": 0.2, "estimate_sigma_data": True, "mean": -3.0, "std": 1.0}},
    "plbert": {"hidden_size": 768, "num_attention_heads": 12, "intermediate_size": 2048,
               "max_position_embeddings": 512, "num_hidden_layers": 12, "dropout": 0.1},
    "slm": {"model": "microsoft/wavlm-base-plus", "sr": 16000, "hidden": 768, "nlayers": 13, "initial_channel": 64},
}


def build_config(
    log_dir: str,
    data_root: str,
    train_list: str,
    val_list: str,
    ood_list: str,
    pretrained_model: str,
    *,
    epochs_1st: int = 8,
    epochs_2nd: int = 5,
    batch_size: int = 8,
    max_len: int = 400,
    save_freq: int = 1,
    joint_epoch: int = 1,
    diff_epoch: int = 1,
    lambda_diff: float = 1.0,
    lambda_sty: float = 1.0,
    lambda_slm: float = 0.2,
    lambda_gen: float = 1.0,
    lr: float = 1e-4,
    bert_lr: float = 1e-5,
    ft_lr: Optional[float] = None,
    resume_ckpt: str = "",
    load_only_params: Optional[bool] = None,
    slm_model_dir: Optional[str] = None,
    num_workers: int = 4,
    disable_slm: bool = False,
    tb_inference: bool = False,
    init_predictor_encoder: bool = False,
    first_stage_path: str = "first_stage.pth",
    second_stage_load_pretrained: bool = False,
    slmadv_max_len: int = 200,
    log_interval: int = 10,
) -> dict:
    mp = copy.deepcopy(MODEL_PARAMS)
    if slm_model_dir:
        mp["slm"]["model"] = slm_model_dir
    cfg = {
        "batch_size": batch_size,
        "max_len": max_len,
        "epochs": max(epochs_1st, epochs_2nd),
        "epochs_1st": epochs_1st,
        "epochs_2nd": epochs_2nd,
        "save_freq": save_freq,
        "log_interval": log_interval,
        "pretrained_model": pretrained_model,
        "first_stage_path": first_stage_path,
        "load_only_params": True if load_only_params is None else load_only_params,
        "second_stage_load_pretrained": second_stage_load_pretrained,
        "log_dir": log_dir,
        # shard-runner knobs read by the patched trainers
        "tb_inference": tb_inference,
        "disable_slm": disable_slm,
        "init_predictor_encoder": init_predictor_encoder,
        "data_params": {
            "train_data": train_list, "val_data": val_list, "root_path": data_root,
            "OOD_data": ood_list, "min_length": 50, "num_workers": num_workers,
        },
        "preprocess_params": {
            "sr": SR,
            "spect_params": {"n_fft": 2048, "win_length": 1200, "hop_length": 300, "n_mels": 80, "fmin": 0, "fmax": 8000},
        },
        "model_params": mp,
        "loss_params": {
            "lambda_gen": lambda_gen, "lambda_mel": 5.0, "lambda_dur": 1.0, "lambda_ce": 20.0,
            "lambda_F0": 1.0, "lambda_norm": 1.0, "lambda_s2s": 1.0, "lambda_mono": 1.0,
            "lambda_slm": 0.0 if disable_slm else lambda_slm, "lambda_diff": lambda_diff, "lambda_sty": lambda_sty,
            "TMA_epoch": 0, "diff_epoch": diff_epoch, "joint_epoch": joint_epoch,
        },
        "optimizer_params": {"lr": lr, "bert_lr": bert_lr, "ft_lr": ft_lr if ft_lr is not None else lr},
        "F0_path": "Utils/JDC/bst.t7",
        "ASR_config": "Utils/ASR/config.yml",
        "ASR_path": "Utils/ASR/epoch_00080.pth",
        "PLBERT_dir": "Utils/PLBERT/",
        "slmadv_params": {"min_len": 100, "max_len": slmadv_max_len, "batch_percentage": 0.5,
                          "iter": 10, "thresh": 5, "scale": 0.01, "sig": 1.5},
    }
    if resume_ckpt:
        cfg["second_stage_load_pretrained"] = True
        cfg["pretrained_model"] = resume_ckpt
        cfg["load_only_params"] = False
    return cfg


# ---------------------------------------------------------------------------
# Stage presets
# ---------------------------------------------------------------------------
STAGE1_VARIANTS = [
    # name, total batch (split across GPUs by accelerate), max_len, mixed precision
    {"name": "s1_fp16_b8_ml400", "batch_size": 8, "max_len": 400, "mixed_precision": "fp16"},
    {"name": "s1_fp16_b4_ml400", "batch_size": 4, "max_len": 400, "mixed_precision": "fp16"},
    {"name": "s1_fp32_b4_ml400", "batch_size": 4, "max_len": 400, "mixed_precision": "no"},
    {"name": "s1_fp32_b4_ml300", "batch_size": 4, "max_len": 300, "mixed_precision": "no"},
]

# NOTE: train_second.py uses DataParallel and several .squeeze() calls, so every GPU must see >= 2
# samples: total batch = 2 x number of GPUs at minimum.
STAGE2_VARIANTS = [
    {"name": "s2_l4_full", "accelerator": "NvidiaL4", "batch_size": 2, "max_len": 300, "disable_slm": False, "joint_epoch": 1},
    {"name": "s2_t4_full", "accelerator": "NvidiaTeslaT4", "batch_size": 4, "max_len": 300, "disable_slm": False, "joint_epoch": 1},
    {"name": "s2_lite_a", "accelerator": "NvidiaTeslaT4", "batch_size": 4, "max_len": 400, "disable_slm": True, "joint_epoch": 1},
    {"name": "s2_lite_a_ml300", "accelerator": "NvidiaTeslaT4", "batch_size": 4, "max_len": 300, "disable_slm": True, "joint_epoch": 1},
    {"name": "s2_lite_b", "accelerator": "NvidiaTeslaT4", "batch_size": 4, "max_len": 400, "disable_slm": True, "joint_epoch": 999},
]

STAGE1_LR = {"lr": 1e-4, "bert_lr": 1e-5, "ft_lr": 1e-4}
STAGE2_LR = {"lr": 5e-5, "bert_lr": 1e-5, "ft_lr": 5e-5}

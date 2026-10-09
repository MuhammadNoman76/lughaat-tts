#!/usr/bin/env python3
"""Highest-naturalness inference path: full StyleTTS2 model with the TRAINED diffusion prosody
sampler (plan 10.6; port of Kokoro-Indic-Fine-Tuning/inference/synthesize.py).

    python synthesize_diffusion.py --text "پاکستان ایک خوبصورت ملک ہے۔" --out out.wav --best-of 3

Requires the full Stage-2 checkpoint `stage2_full.pth` and `config_s2.yml` next to this file
(both uploaded in the model repo's `diffusion/` folder), a clone of kikiri-tts (cloned on first
run), and the `lughaat_tts` frontend. Style reference: a Rasa clip (`--ref`) or the voicepack.
Defaults: alpha 0.3, beta 0.7, 10 diffusion steps, embedding scale 1.5, best-of-N by DNSMOS.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FORK_URL = "https://github.com/semidark/kikiri-tts"
FORK_SHA = "a12d0410e89841e6f3c09958ae5c072f90ae1d49"
WORK = os.path.join(HERE, ".cache")


def ensure_fork() -> str:
    kk = os.path.join(WORK, "kikiri-tts")
    if not os.path.isdir(os.path.join(kk, "StyleTTS2")):
        os.makedirs(WORK, exist_ok=True)
        subprocess.run(["git", "clone", "-q", FORK_URL, kk], check=True)
        subprocess.run(["git", "-C", kk, "checkout", "-q", FORK_SHA], check=True)
        subprocess.run(["git", "-C", kk, "submodule", "update", "--init", "--recursive", "-q"], check=True)
        sys.path.insert(0, os.path.dirname(HERE))
        try:
            from training.patches import apply_all
            apply_all(kk, print)
        except Exception:
            # minimal: install the symbol map
            import shutil
            shutil.copy(os.path.join(kk, "training", "kokoro_symbols.py"), os.path.join(kk, "StyleTTS2", "kokoro_symbols.py"))
    return os.path.join(kk, "StyleTTS2")


def load_model(st2: str, ckpt: str, cfg_path: str, device: str):
    import torch, yaml
    os.chdir(st2)
    sys.path.insert(0, st2)
    from models import build_model, load_ASR_models, load_F0_models
    from utils import recursive_munch
    from Utils.PLBERT.util import load_plbert
    from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule
    config = yaml.safe_load(open(cfg_path))
    text_aligner = load_ASR_models(config["ASR_path"], config["ASR_config"])
    pitch_extractor = load_F0_models(config["F0_path"])
    plbert = load_plbert(config["PLBERT_dir"])
    mp = recursive_munch(config["model_params"])
    model = build_model(mp, text_aligner, pitch_extractor, plbert)
    params = torch.load(ckpt, map_location="cpu", weights_only=False)["net"]
    for key in model:
        if key in params:
            sd = {(k[7:] if k.startswith("module.") else k): v for k, v in params[key].items()}
            model[key].load_state_dict(sd, strict=False)
    _ = [model[k].eval().to(device) for k in model]
    sampler = DiffusionSampler(model.diffusion.diffusion, sampler=ADPM2Sampler(),
                               sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0), clamp=False)
    return model, sampler


def compute_style(model, ref_wav: str | None, voicepack: str | None, device: str):
    import torch
    if ref_wav:
        import librosa, torchaudio
        wave, _ = librosa.load(ref_wav, sr=24000)
        audio, _ = librosa.effects.trim(wave, top_db=30)
        to_mel = torchaudio.transforms.MelSpectrogram(n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
        mel = (torch.log(1e-5 + to_mel(torch.from_numpy(audio).float()).unsqueeze(0)) - (-4)) / 4
        with torch.no_grad():
            return torch.cat([model.style_encoder(mel.to(device).unsqueeze(1)), model.predictor_encoder(mel.to(device).unsqueeze(1))], dim=1)
    pack = torch.load(voicepack, map_location="cpu", weights_only=True)
    return pack[100].to(device)  # [1, 256]


def synthesize(model, sampler, ref_s, phonemes: str, device: str, alpha=0.3, beta=0.7, steps=10, escale=1.5, seed=0):
    import numpy as np, torch
    from kokoro_symbols import TextCleaner
    torch.manual_seed(seed); np.random.seed(seed)
    tokens = TextCleaner()(phonemes)
    tokens.insert(0, 0)
    tokens = torch.LongTensor(tokens).to(device).unsqueeze(0)
    with torch.no_grad():
        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(device)
        text_mask = torch.gt(torch.arange(input_lengths.max()).unsqueeze(0).expand(1, -1).type_as(input_lengths) + 1, input_lengths.unsqueeze(1)).to(device)
        t_en = model.text_encoder(tokens, input_lengths, text_mask)
        bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2)
        s_pred = sampler(noise=torch.randn((1, 256)).unsqueeze(1).to(device), embedding=bert_dur, embedding_scale=escale,
                         features=ref_s, num_steps=steps).squeeze(1)
        s = beta * s_pred[:, 128:] + (1 - beta) * ref_s[:, 128:]
        ref = alpha * s_pred[:, :128] + (1 - alpha) * ref_s[:, :128]
        d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        x, _ = model.predictor.lstm(d)
        duration = torch.sigmoid(model.predictor.duration_proj(x)).sum(axis=-1)
        pred_dur = torch.round(duration.squeeze()).clamp(min=1)
        aln = torch.zeros(input_lengths, int(pred_dur.sum().data))
        c = 0
        for i in range(aln.size(0)):
            aln[i, c:c + int(pred_dur[i].data)] = 1
            c += int(pred_dur[i].data)
        en = d.transpose(-1, -2) @ aln.unsqueeze(0).to(device)
        F0_pred, N_pred = model.predictor.F0Ntrain(en, s)
        asr = t_en @ aln.unsqueeze(0).to(device)
        wav = model.decoder(asr, F0_pred, N_pred, ref.squeeze().unsqueeze(0))
    return wav.squeeze().cpu().numpy()[..., :-50]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--out", default="out.wav")
    ap.add_argument("--ckpt", default=os.path.join(HERE, "stage2_full.pth"))
    ap.add_argument("--config", default=os.path.join(HERE, "config_s2.yml"))
    ap.add_argument("--ref", default=None, help="reference wav (Rasa clip); else --voicepack")
    ap.add_argument("--voicepack", default=os.path.join(os.path.dirname(HERE), "voices", "uf_rasa.pt"))
    ap.add_argument("--accent", default="auto")
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--beta", type=float, default=0.7)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--escale", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--best-of", type=int, default=1)
    a = ap.parse_args()
    import soundfile as sf, torch
    sys.path.insert(0, os.path.dirname(HERE))
    from lughaat_tts.codeswitch import MixedFrontend
    fe = MixedFrontend()
    ps = fe(a.text, english_accent=a.accent)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st2 = ensure_fork()
    model, sampler = load_model(st2, a.ckpt, a.config, device)
    ref_s = compute_style(model, a.ref, a.voicepack, device)
    takes = [synthesize(model, sampler, ref_s, ps, device, a.alpha, a.beta, a.steps, a.escale, a.seed + i) for i in range(a.best_of)]
    audio = takes[0]
    if len(takes) > 1:
        try:
            from eval.metrics import dnsmos
            scores = [dnsmos(t) for t in takes]
            audio = takes[max(range(len(takes)), key=lambda i: scores[i])]
            print(f"best-of-{len(takes)} DNSMOS {max(scores):.3f}")
        except Exception:
            pass
    sf.write(a.out, audio, 24000)
    print(f"wrote {a.out} ({len(audio)/24000:.2f}s) phonemes: {ps}")


if __name__ == "__main__":
    main()

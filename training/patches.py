"""Trainer patches for the pinned kikiri-tts / StyleTTS2 checkout (plan 6.7, 13, 14.3).

Every patch is an exact-anchor text replacement with an assertion, so a changed upstream
file fails loudly instead of training with a half-applied patch. Patches are idempotent
(each one checks for its own marker).

    python training/patches.py --kikiri work/kikiri-tts

Pinned commits (verified against these anchors):
    kikiri-tts        a12d0410e89841e6f3c09958ae5c072f90ae1d49
    StyleTTS2 (fork)  b1956da84bf4a6ccc88f2440078024f1c4bfec7d
    kokoro (fork)     b96fef95e6a746495f92443fac7c688f90fc57fc

Patch list
----------
models.py
  RESUME-FIX-A   strip DataParallel 'module.' prefixes in load_checkpoint and print matched/total
train_first.py
  TB-SKIP        skip TensorBoard voicepack extraction + inference when config.tb_inference is false
  NAN-SKIP       skip a batch whose mel loss is not finite instead of poisoning the weights
  GRADCLIP       clip generator gradients at 100 (accelerator-aware, fp16 safe)
  SCHED-SAVE     save scheduler state in each checkpoint
train_second.py
  ANOMALY-OFF    torch.autograd.set_detect_anomaly(True) -> False (large speed-up)
  RESUME-FIX-B   never clobber the trained predictor_encoder on a resume (config.init_predictor_encoder)
  FT-LR          re-apply the configured learning rates after optimizer.load_state_dict
  SCHED-SAVE     save scheduler state in each checkpoint
  ALIGNER-EXC    verbose logging for the aligner exception (was a bare `except: continue`)
  SLMADV-NONE    log when the SLM adversary returns None; keep logging/iteration counters alive
  SLM-OFF        config.disable_slm: do not load WavLM at all and skip the slmadv block (Stage-2-lite A)
  STEP-EXC       print the validation exception
  GRADCLIP       clip discriminator and generator gradients at 100
  NAN-SKIP       skip the batch on a non-finite mel loss instead of sys.exit(1)
  TB-SKIP        as above
"""
from __future__ import annotations

import argparse
import os
import re
import sys

KIKIRI_SHA = "a12d0410e89841e6f3c09958ae5c072f90ae1d49"
STYLETTS2_SHA = "b1956da84bf4a6ccc88f2440078024f1c4bfec7d"
KOKORO_SHA = "b96fef95e6a746495f92443fac7c688f90fc57fc"


class PatchError(RuntimeError):
    pass


def _apply(src: str, marker: str, old: str, new: str, count: int = 1) -> str:
    if marker in src:
        return src  # already applied
    n = src.count(old)
    if n != count:
        raise PatchError(f"{marker}: anchor found {n} times (expected {count}). Upstream file changed; pin the commit.")
    return src.replace(old, new)


def _write(path: str, src: str) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(src)


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
def patch_models(st2: str, log=print) -> None:
    p = os.path.join(st2, "models.py")
    s = _read(p)
    s = _apply(s, "RESUME-FIX-A",
        "            model[key].load_state_dict(params[key], strict=False)",
        '            _sd = {(_k[7:] if _k.startswith("module.") else _k): _v for _k, _v in params[key].items()}  # RESUME-FIX-A\n'
        "            _r = model[key].load_state_dict(_sd, strict=False)\n"
        '            print(f"  {key}: matched {len(_sd) - len(_r.unexpected_keys)}/{len(_sd)} tensors", flush=True)')
    _write(p, s)
    log("  models.py: RESUME-FIX-A")


def patch_train_first(st2: str, log=print) -> None:
    p = os.path.join(st2, "train_first.py")
    s = _read(p)
    # TB-SKIP (baseline + per-epoch)
    s = _apply(s, "TB-SKIP-A",
        "        _baseline_voicepack, _baseline_acoustic_norm, _baseline_prosodic_norm = (\n"
        "            extract_voicepack(model, root_path, device, n_samples=200)\n"
        "        )",
        "        _baseline_voicepack, _baseline_acoustic_norm, _baseline_prosodic_norm = (  # TB-SKIP-A\n"
        '            extract_voicepack(model, root_path, device, n_samples=200) if config.get("tb_inference", True) else (None, 0.0, 0.0)\n'
        "        )")
    s = _apply(s, "TB-SKIP-B",
        "            _vp, _acoustic_norm, _prosodic_norm = extract_voicepack(\n"
        "                model,\n"
        "                root_path,\n"
        "                device,\n"
        "                n_samples=200,\n"
        "            )",
        "            _vp, _acoustic_norm, _prosodic_norm = extract_voicepack(  # TB-SKIP-B\n"
        "                model,\n"
        "                root_path,\n"
        "                device,\n"
        "                n_samples=200,\n"
        '            ) if config.get("tb_inference", True) else (None, 0.0, 0.0)')
    # NAN-SKIP
    s = _apply(s, "NAN-SKIP",
        "            loss_mel = stft_loss(y_rec.squeeze(), wav.detach())\n\n"
        "            if epoch >= TMA_epoch:  # start TMA training",
        "            loss_mel = stft_loss(y_rec.squeeze(), wav.detach())\n"
        "            if not torch.isfinite(loss_mel):  # NAN-SKIP\n"
        '                print("NAN-SKIP loss_mel batch", i, flush=True)\n'
        "                optimizer.zero_grad()\n"
        "                continue\n\n"
        "            if epoch >= TMA_epoch:  # start TMA training")
    # GRADCLIP
    s = _apply(s, "GRADCLIP",
        "            accelerator.backward(g_loss)\n\n"
        '            optimizer.step("text_encoder")',
        "            accelerator.backward(g_loss)\n"
        "            accelerator.clip_grad_norm_([p for _k in ('text_encoder', 'style_encoder', 'decoder') for p in model[_k].parameters()], 100.0)  # GRADCLIP\n\n"
        '            optimizer.step("text_encoder")')
    # SCHED-SAVE
    s = _apply(s, "SCHED-SAVE",
        '                    "val_loss": loss_test / iters_test,\n'
        '                    "epoch": epoch,\n'
        "                }\n"
        '                save_path = osp.join(log_dir, "epoch_1st_%05d.pth" % epoch)',
        '                    "val_loss": loss_test / iters_test,\n'
        '                    "epoch": epoch,\n'
        '                    "schedulers": [(_sk, optimizer.schedulers[_sk].state_dict()) for _sk in optimizer.schedulers],  # SCHED-SAVE\n'
        "                }\n"
        '                save_path = osp.join(log_dir, "epoch_1st_%05d.pth" % epoch)')
    _write(p, s)
    log("  train_first.py: TB-SKIP, NAN-SKIP, GRADCLIP, SCHED-SAVE")


def patch_train_second(st2: str, log=print) -> None:
    p = os.path.join(st2, "train_second.py")
    s = _read(p)
    s = _apply(s, "ANOMALY-OFF",
        "torch.autograd.set_detect_anomaly(True)",
        "torch.autograd.set_detect_anomaly(False)  # ANOMALY-OFF")
    # RESUME-FIX-B (resume path only; the first_stage path keeps its deepcopy)
    s = _apply(s, "RESUME-FIX-B",
        "        # Initialize predictor_encoder from trained style_encoder\n"
        "        # (predictor_encoder is not trained in Stage 1)\n"
        "        model.predictor_encoder = copy.deepcopy(model.style_encoder)",
        "        # Initialize predictor_encoder from trained style_encoder\n"
        "        # (predictor_encoder is not trained in Stage 1)\n"
        '        if config.get("init_predictor_encoder", False):  # RESUME-FIX-B: never clobber a trained predictor_encoder on resume\n'
        "            model.predictor_encoder = copy.deepcopy(model.style_encoder)")
    # FT-LR
    s = _apply(s, "FT-LR",
        "        # Ensure all modules are in train() mode after loading\n"
        "        # (load_checkpoint sets them to eval(), which breaks spectral_norm)\n"
        "        _ = [model[key].train() for key in model]",
        "        # Ensure all modules are in train() mode after loading\n"
        "        # (load_checkpoint sets them to eval(), which breaks spectral_norm)\n"
        "        _ = [model[key].train() for key in model]\n"
        "        for _fk, _fo in optimizer.optimizers.items():  # FT-LR: load_state_dict restored the OLD lrs\n"
        '            _flr = optimizer_params.bert_lr if _fk == "bert" else (optimizer_params.ft_lr if _fk in ("decoder", "style_encoder") else optimizer_params.lr)\n'
        "            for _fg in _fo.param_groups:\n"
        '                _fg["lr"] = _flr\n'
        '        print("FT-LR: learning rates reset from config", flush=True)')
    # SCHED-SAVE
    s = _apply(s, "SCHED-SAVE",
        '                "val_loss": loss_test / iters_test,\n'
        '                "epoch": epoch,\n'
        "            }\n"
        '            save_path = osp.join(log_dir, "epoch_2nd_%05d.pth" % epoch)',
        '                "val_loss": loss_test / iters_test,\n'
        '                "epoch": epoch,\n'
        '                "schedulers": [(_sk, optimizer.schedulers[_sk].state_dict()) for _sk in optimizer.schedulers],  # SCHED-SAVE\n'
        "            }\n"
        '            save_path = osp.join(log_dir, "epoch_2nd_%05d.pth" % epoch)')
    # ALIGNER-EXC
    s = _apply(s, "ALIGNER-EXC",
        "                except:\n"
        "                    continue",
        "                except Exception as _e:  # ALIGNER-EXC\n"
        "                    print('ALIGNER-EXC:', repr(_e), flush=True); continue")
    # STEP-EXC
    s = _apply(s, "STEP-EXC",
        '                    print(f"run into exception", e)',
        '                    print("STEP-EXC:", repr(e), flush=True)')
    # SLM-OFF: dummy WavLM loss when disabled
    s = _apply(s, "SLM-OFF-A",
        "    wl = WavLMLoss(model_params.slm.model, model.wd, sr, model_params.slm.sr).to(device)",
        "    class _DummyWL(torch.nn.Module):  # SLM-OFF-A\n"
        "        def forward(self, wav, y_rec):\n"
        "            return torch.zeros((), device=y_rec.device, requires_grad=True) * 0\n"
        "        def generator(self, y_rec):\n"
        "            return torch.zeros((), device=y_rec.device)\n"
        "        def discriminator(self, wav, y_rec):\n"
        "            return torch.zeros((), device=y_rec.device)\n"
        "        def discriminator_forward(self, wav):\n"
        "            return torch.zeros((1,), device=wav.device)\n"
        '    wl = (_DummyWL() if config.get("disable_slm", False) else WavLMLoss(model_params.slm.model, model.wd, sr, model_params.slm.sr)).to(device)')
    # SLM-OFF / SLMADV-NONE: restructure the slmadv block so a None result keeps logging alive
    start_anchor = "                d_loss_slm, loss_gen_lm, y_pred = slm_out\n"
    end_anchor = (
        "                # SLM discriminator loss\n"
        "                if d_loss_slm != 0:\n"
        "                    optimizer.zero_grad()\n"
        "                    d_loss_slm.backward(retain_graph=True)\n"
        '                    optimizer.step("wd")\n')
    if "SLMADV-NONE" not in s:
        if s.count(start_anchor) != 1 or s.count(end_anchor) != 1:
            raise PatchError("SLMADV-NONE: block anchors not found exactly once")
        i0 = s.index(start_anchor)
        i1 = s.index(end_anchor) + len(end_anchor)
        block = s[i0:i1]
        indented = "".join(("    " + ln if ln.strip() else ln) for ln in block.splitlines(keepends=True))
        s = s[:i0] + indented + s[i1:]
        s = _apply(s, "SLMADV-NONE",
            "                slm_out = slmadv(\n",
            '                slm_out = None if config.get("disable_slm", False) else slmadv(  # SLM-OFF-B\n')
        s = s.replace(
            "                if slm_out is None:\n"
            "                    continue\n",
            "                if slm_out is None:  # SLMADV-NONE\n"
            '                    if not config.get("disable_slm", False):\n'
            "                        print('SLMADV-NONE step', i, flush=True)\n"
            "                    d_loss_slm, loss_gen_lm = 0, 0\n"
            "                else:\n", 1)
    # GRADCLIP
    s = _apply(s, "GRADCLIP-D",
        "                d_loss.backward()\n"
        '                optimizer.step("msd")',
        "                d_loss.backward()\n"
        "                torch.nn.utils.clip_grad_norm_(list(model.msd.parameters()) + list(model.mpd.parameters()), 100.0)  # GRADCLIP-D\n"
        '                optimizer.step("msd")')
    s = _apply(s, "GRADCLIP-G",
        "            g_loss.backward()\n\n"
        '            optimizer.step("bert_encoder")',
        "            g_loss.backward()\n"
        "            torch.nn.utils.clip_grad_norm_([p for _k in ('bert', 'bert_encoder', 'predictor', 'predictor_encoder', 'style_encoder', 'decoder', 'text_encoder') for p in model[_k].parameters()], 100.0)  # GRADCLIP-G\n\n"
        '            optimizer.step("bert_encoder")')
    # NAN-SKIP
    s = _apply(s, "NAN-SKIP",
        '                print("NaN detected in loss_mel after loss_mel = stft_loss(y_rec, wav)")\n'
        "                import sys\n\n"
        "                sys.exit(1)",
        '                print("NAN-SKIP loss_mel batch", i, flush=True)\n'
        "                optimizer.zero_grad(); continue")
    # TB-SKIP
    s = _apply(s, "TB-SKIP-A",
        "    _baseline_voicepack, _baseline_acoustic_norm, _baseline_prosodic_norm = (\n"
        "        extract_voicepack(\n"
        "            model,\n"
        "            root_path,\n"
        "            device,\n"
        "            n_samples=200,\n"
        "        )\n"
        "    )",
        "    _baseline_voicepack, _baseline_acoustic_norm, _baseline_prosodic_norm = (  # TB-SKIP-A\n"
        "        extract_voicepack(\n"
        "            model,\n"
        "            root_path,\n"
        "            device,\n"
        "            n_samples=200,\n"
        '        ) if config.get("tb_inference", True) else (None, 0.0, 0.0)\n'
        "    )")
    s = _apply(s, "TB-SKIP-B",
        "        _vp, _acoustic_norm, _prosodic_norm = extract_voicepack(\n"
        "            model,\n"
        "            root_path,\n"
        "            device,\n"
        "            n_samples=200,\n"
        "        )",
        "        _vp, _acoustic_norm, _prosodic_norm = extract_voicepack(  # TB-SKIP-B\n"
        "            model,\n"
        "            root_path,\n"
        "            device,\n"
        "            n_samples=200,\n"
        '        ) if config.get("tb_inference", True) else (None, 0.0, 0.0)')
    _write(p, s)
    log("  train_second.py: ANOMALY-OFF, RESUME-FIX-B, FT-LR, SCHED-SAVE, ALIGNER-EXC, STEP-EXC, SLM-OFF, SLMADV-NONE, GRADCLIP, NAN-SKIP, TB-SKIP")


def patch_kokoro_init(kikiri: str, log=print) -> None:
    """kokoro/__init__.py imports KPipeline -> misaki; not needed inside the trainer env."""
    for cand in (os.path.join(kikiri, "kokoro", "kokoro", "__init__.py"),):
        if os.path.exists(cand):
            s = _read(cand)
            if "from .pipeline import KPipeline" in s and "KPIPELINE-OPTIONAL" not in s:
                s = s.replace("from .pipeline import KPipeline",
                              "try:  # KPIPELINE-OPTIONAL\n    from .pipeline import KPipeline\nexcept Exception:  # misaki not installed in the trainer env\n    KPipeline = None")
                _write(cand, s)
                log("  kokoro/__init__.py: KPIPELINE-OPTIONAL")


def install_symbols(kikiri: str, log=print) -> None:
    import shutil
    here = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(here, "kokoro_symbols.py")
    dst = os.path.join(kikiri, "StyleTTS2", "kokoro_symbols.py")
    shutil.copy(src, dst)
    sys.path.insert(0, os.path.dirname(dst))
    import importlib
    ks = importlib.import_module("kokoro_symbols")
    assert len(ks.symbols) == 178 and ks.dicts["ʰ"] == 162 and ks.dicts["ɽ"] == 129 and ks.dicts["q"] == 59 and ks.dicts["̃"] == 17
    tu = _read(os.path.join(kikiri, "StyleTTS2", "text_utils.py"))
    md = _read(os.path.join(kikiri, "StyleTTS2", "meldataset.py"))
    assert "from kokoro_symbols import" in tu and "from kokoro_symbols import" in md, "text_utils/meldataset must import kokoro_symbols"
    log("  kokoro_symbols.py installed and verified (178 tokens, ʰ=162 ɽ=129 q=59 ̃=17)")


def verify_pins(kikiri: str, log=print) -> None:
    import subprocess
    def sha(path: str) -> str:
        try:
            return subprocess.run(["git", "-C", path, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        except Exception:
            return "?"
    got = {"kikiri": sha(kikiri), "StyleTTS2": sha(os.path.join(kikiri, "StyleTTS2")), "kokoro": sha(os.path.join(kikiri, "kokoro"))}
    want = {"kikiri": KIKIRI_SHA, "StyleTTS2": STYLETTS2_SHA, "kokoro": KOKORO_SHA}
    for k in want:
        if got[k] not in ("?", want[k]):
            log(f"  WARNING: {k} is at {got[k][:12]}, patches were verified against {want[k][:12]}")
    log(f"  commits: {got}")


def apply_all(kikiri: str, log=print) -> None:
    st2 = os.path.join(kikiri, "StyleTTS2")
    assert os.path.exists(os.path.join(st2, "train_first.py")), f"StyleTTS2 submodule missing in {kikiri}"
    verify_pins(kikiri, log)
    install_symbols(kikiri, log)
    patch_models(st2, log)
    patch_train_first(st2, log)
    patch_train_second(st2, log)
    patch_kokoro_init(kikiri, log)
    for f in ("train_first.py", "train_second.py", "models.py"):
        compile(_read(os.path.join(st2, f)), f, "exec")
    log("  all patched files compile")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kikiri", required=True, help="path to the kikiri-tts checkout (with submodules)")
    a = ap.parse_args()
    apply_all(a.kikiri)


if __name__ == "__main__":
    main()

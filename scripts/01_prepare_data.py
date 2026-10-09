#!/usr/bin/env python3
"""Phase 1: data download, selection, processing, filtering, phonemisation and lists (plan 3.2, 14.4).

Idempotent, step-based (each step writes a marker in <out>/.done/). Designed to run inside a
Kaggle session (the session runner calls one step at a time and checks the time budget), but
it also runs on any Linux box with an HF token.

    python scripts/01_prepare_data.py --out /kaggle/tmp/data --step all
    python scripts/01_prepare_data.py --out data --step select        # one step

Steps (in order)
    select     stream Rasa Urdu (+ optional IndicVoices-R) and LibriTTS-R; keep audio bytes for the
               hour budget (neutral styles first, emotional reserve) -> raw/ + manifest.jsonl
    audio      decode -> 24 kHz mono -> peak -1 dBFS -> trim -> 1.5-20 s -> audio/<spk>/<utt>.wav
    asr        Whisper consistency filter (+ FLEURS ASR floor)
    audit      per-speaker bandwidth audit (CPU)
    lexicon    expand the pronunciation lexicon with Rasa + Wikipedia words (scripts/02_build_lexicon.py)
               and retrain the neural G2P (scripts/03_train_g2p.py)
    phonemize  Urdu frontend for Urdu clips, misaki for English clips; vocab + 510 check
    lists      speaker ids, stratified val split, mix report, train_list/val_list
    ood        OOD_texts.txt (Urdu Wikipedia + English + synthetic code-switched) and eval text sets
    pack       audio.tar + data_report.md (+ --upload to the HF work repo)
"""
from __future__ import annotations

import argparse
import collections
import glob
import io
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tarfile
import time
from typing import Iterable, Optional

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from lughaat_tts.normalize import normalize, normalize_chars, has_latin, has_arabic_script  # noqa: E402
from lughaat_tts.vocab import VOCAB  # noqa: E402

SR = 24000
STEPS = ["select", "audio", "asr", "audit", "lexicon", "phonemize", "lists", "ood", "pack"]
EMOTION_KEYWORDS = ("HAPP", "SAD", "ANG", "FEAR", "DISG", "SURP", "EMO")
RASA_SPEAKERS = {"female": 0, "male": 1}
ENGLISH_SPEAKER_BASE = 2

# common English nouns/verbs for synthetic code-switched sentences (plan 3.2.11)
SWAP_WORDS = ["meeting", "office", "phone", "cancel", "problem", "update", "email", "laptop", "internet", "message",
              "schedule", "report", "deadline", "project", "manager", "system", "online", "class", "exam", "result",
              "ticket", "flight", "hotel", "market", "price", "budget", "plan", "team", "client", "service", "password",
              "account", "download", "upload", "video", "camera", "battery", "charger", "screen", "software", "app",
              "website", "link", "file", "folder", "printer", "doctor", "hospital", "medicine", "test", "appointment"]
# Urdu words that are commonly replaced by their English equivalents in speech
SWAP_TARGETS = {"ملاقات": "meeting", "دفتر": "office", "فون": "phone", "منسوخ": "cancel", "مسئلہ": "problem",
                "خط": "email", "انٹرنیٹ": "internet", "پیغام": "message", "رپورٹ": "report", "منصوبہ": "project",
                "نظام": "system", "امتحان": "exam", "نتیجہ": "result", "ٹکٹ": "ticket", "پرواز": "flight", "ہوٹل": "hotel",
                "بازار": "market", "قیمت": "price", "ٹیم": "team", "گاہک": "client", "ڈاکٹر": "doctor", "ہسپتال": "hospital",
                "دوا": "medicine", "ٹیسٹ": "test", "کلاس": "class", "ویڈیو": "video", "کیمرہ": "camera", "ویب سائٹ": "website",
                "فائل": "file", "پرنٹر": "printer", "سکول": "school", "اسکول": "school", "یونیورسٹی": "university", "کمپیوٹر": "computer",
                "گاڑی": "car", "سڑک": "road", "شہر": "city", "حکومت": "government", "پولیس": "police", "فیصلہ": "decision",
                "خبر": "news", "اخبار": "newspaper", "کتاب": "book", "سوال": "question", "جواب": "answer", "وقت": "time",
                "موسم": "weather", "پانی": "water", "کھانا": "food", "کام": "work", "پیسے": "money", "بینک": "bank"}


def log(msg: str) -> None:
    print(f"[prep {time.strftime('%H:%M:%S')}] {msg}", flush=True)


class Prep:
    def __init__(self, out: str, a: argparse.Namespace):
        self.out = out
        self.a = a
        self.raw = os.path.join(out, "raw")
        self.audio = os.path.join(out, "audio")
        self.done_dir = os.path.join(out, ".done")
        self.eval_dir = os.path.join(out, "eval")
        for d in (self.raw, self.audio, self.done_dir, self.eval_dir):
            os.makedirs(d, exist_ok=True)
        self.manifest_path = os.path.join(out, "manifest.jsonl")
        self.report: dict = self._load_json(os.path.join(out, "data_report.json"), {})
        self.token = a.hf_token or os.environ.get("HF_TOKEN")

    # -- helpers ----------------------------------------------------------------
    def done(self, step: str) -> bool:
        return os.path.exists(os.path.join(self.done_dir, step))

    def mark(self, step: str) -> None:
        open(os.path.join(self.done_dir, step), "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))
        self._save_json(os.path.join(self.out, "data_report.json"), self.report)

    @staticmethod
    def _load_json(p: str, default):
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        return default

    @staticmethod
    def _save_json(p: str, obj) -> None:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

    def read_manifest(self, require: bool = True) -> list[dict]:
        rows = []
        if os.path.exists(self.manifest_path):
            with open(self.manifest_path, encoding="utf-8") as f:
                for ln in f:
                    if ln.strip():
                        rows.append(json.loads(ln))
        if require and not rows:
            raise SystemExit(f"manifest {self.manifest_path} is missing or empty: the earlier data steps did not run in this "
                             "session (their outputs are session-local). Run from --step select again.")
        return rows

    def write_manifest(self, rows: list[dict]) -> None:
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # -- step: select ------------------------------------------------------------
    def step_select(self) -> None:
        from datasets import load_dataset, Audio  # type: ignore
        rows: list[dict] = []
        hours_cap = self.a.urdu_hours_per_speaker * 3600.0
        # composition target inside the cap: ~80 % neutral reading styles, ~20 % emotional styles
        # (plan 3.3: train on all styles; plan 14.4: neutral first). Extra emotional clips are kept as a
        # reserve (up to 0.5 x cap) in case a speaker's neutral material falls short of the cap.
        neutral_target = 0.8 * hours_cap
        emotional_target = 0.2 * hours_cap
        reserve_cap = 0.5 * hours_cap
        kept = collections.defaultdict(float)      # spk -> seconds of neutral kept
        reserve = collections.defaultdict(float)   # spk -> seconds of emotional kept
        styles = collections.Counter()
        total_seen = collections.defaultdict(float)
        log(f"streaming ai4bharat/Rasa Urdu (cap {self.a.urdu_hours_per_speaker} h/speaker neutral-first)")
        ds = load_dataset("ai4bharat/Rasa", "Urdu", split="train", streaming=True, token=self.token)
        ds = ds.cast_column("audio", Audio(decode=False))
        n = 0
        for ex in ds:
            n += 1
            gender = str(ex.get("gender", "")).strip().lower()
            spk = RASA_SPEAKERS.get(gender)
            if spk is None:
                spk = RASA_SPEAKERS.setdefault(gender, len(RASA_SPEAKERS))
            style = str(ex.get("style", "")).strip()
            styles[(gender, style)] += 1
            try:
                dur = float(ex.get("duration") or 0.0)
            except Exception:
                dur = 0.0
            total_seen[spk] += dur
            emotional = any(k in style.upper() for k in EMOTION_KEYWORDS)
            if dur <= 0 or dur > 25.0:
                continue
            take = False
            if not emotional and kept[spk] < hours_cap:
                kept[spk] += dur
                take = True
            elif emotional and reserve[spk] < reserve_cap:
                reserve[spk] += dur
                take = True
            if not take:
                speakers = list(RASA_SPEAKERS.values())
                if all(kept[s] >= hours_cap for s in speakers) and all(reserve[s] >= emotional_target for s in speakers):
                    log("all Rasa budgets full; stopping stream")
                    break
                continue
            audio = ex["audio"]
            fname = os.path.basename(str(ex.get("filename") or ex.get("wav_path") or audio.get("path") or f"rasa_{n:06d}.wav"))
            if not fname.lower().endswith(".wav"):
                fname += ".wav"
            d = os.path.join(self.raw, "rasa", str(spk))
            os.makedirs(d, exist_ok=True)
            p = os.path.join(d, fname)
            with open(p, "wb") as f:
                f.write(audio["bytes"])
            rows.append({"utt": f"rasa_{spk}_{os.path.splitext(fname)[0]}", "raw": p, "spk": spk, "lang": "ur",
                         "source": "rasa", "style": style, "gender": gender, "emotional": emotional,
                         "text_raw": str(ex.get("text", "")), "dur_raw": dur})
            if n % 2000 == 0:
                log(f"  seen {n} rows; neutral h: { {k: round(v/3600,2) for k,v in kept.items()} } reserve h: { {k: round(v/3600,2) for k,v in reserve.items()} }")
        self.report["rasa"] = {"rows_seen": n, "styles": {f"{g}|{s}": c for (g, s), c in styles.most_common()},
                               "hours_seen_per_spk": {str(k): round(v / 3600, 2) for k, v in total_seen.items()},
                               "neutral_hours_kept": {str(k): round(v / 3600, 2) for k, v in kept.items()},
                               "emotional_hours_reserve": {str(k): round(v / 3600, 2) for k, v in reserve.items()}}
        # final Rasa selection per speaker: neutral up to 80 % of the cap, emotional up to 20 %,
        # then fill any shortfall with whatever is left (neutral first), never exceeding the cap
        final_rows = []
        rng = random.Random(0)
        for spk in sorted(set(r["spk"] for r in rows)):
            neu = [r for r in rows if r["spk"] == spk and not r["emotional"]]
            emo = [r for r in rows if r["spk"] == spk and r["emotional"]]
            rng.shuffle(neu)
            rng.shuffle(emo)

            def take(pool: list[dict], budget: float, sel: list[dict], used: set) -> float:
                got = 0.0
                for r in pool:
                    if got >= budget:
                        break
                    if id(r) in used:
                        continue
                    sel.append(r)
                    used.add(id(r))
                    got += r["dur_raw"]
                return got

            sel: list[dict] = []
            used: set = set()
            secs = take(neu, neutral_target, sel, used)
            secs += take(emo, emotional_target, sel, used)
            if secs < hours_cap:
                secs += take(neu, hours_cap - secs, sel, used)
            if secs < hours_cap:
                secs += take(emo, hours_cap - secs, sel, used)
            final_rows += sel
        keep_ids = {id(r) for r in final_rows}
        for r in rows:
            if id(r) not in keep_ids:
                try:
                    os.remove(r["raw"])
                except OSError:
                    pass
        rows = final_rows
        self.report["rasa"]["selected_hours_per_spk"] = {
            str(s): {"neutral": round(sum(r["dur_raw"] for r in rows if r["spk"] == s and not r["emotional"]) / 3600, 2),
                     "emotional": round(sum(r["dur_raw"] for r in rows if r["spk"] == s and r["emotional"]) / 3600, 2)}
            for s in sorted(set(r["spk"] for r in rows))}
        log(f"Rasa selected {len(rows)} clips = {sum(r['dur_raw'] for r in rows)/3600:.2f} h")

        if self.a.use_indicvoices_r:
            rows += self._select_indicvoices_r()

        # English replay: LibriTTS-R train.clean.100
        eng_cap = self.a.english_hours * 3600.0
        per_spk = eng_cap / max(self.a.english_speakers, 1)
        log(f"streaming mythicinfinity/libritts_r clean/train.clean.100 (cap {self.a.english_hours} h, ~{self.a.english_speakers} speakers)")
        ds = load_dataset("mythicinfinity/libritts_r", "clean", split="train.clean.100", streaming=True, token=self.token)
        ds = ds.cast_column("audio", Audio(decode=False))
        eng_kept = collections.defaultdict(float)
        total = 0.0
        spk_ids: dict[str, int] = {}
        for ex in ds:
            if total >= eng_cap:
                break
            spk = str(ex.get("speaker_id"))
            # per-speaker cap = total / target speaker count, but keep accepting new speakers until the
            # hour cap is reached (a fixed first-N-speakers rule under-filled the cap: 2.45 of 3 h)
            if eng_kept[spk] >= per_spk:
                continue
            if spk not in spk_ids:
                spk_ids[spk] = ENGLISH_SPEAKER_BASE + len(spk_ids)
            audio = ex["audio"]
            b = audio["bytes"]
            try:
                import soundfile as sf
                info = sf.info(io.BytesIO(b))
                dur = info.frames / info.samplerate
            except Exception:
                continue
            if not (2.0 <= dur <= 15.0):
                continue
            text = str(ex.get("text_normalized") or ex.get("text_original") or "").strip()
            if not text or has_arabic_script(text):
                continue
            sid = spk_ids[spk]
            d = os.path.join(self.raw, "libritts", str(sid))
            os.makedirs(d, exist_ok=True)
            fname = os.path.basename(str(ex.get("id") or audio.get("path") or f"lt_{len(rows):06d}")) + ".wav"
            p = os.path.join(d, fname)
            with open(p, "wb") as f:
                f.write(b)
            eng_kept[spk] += dur
            total += dur
            rows.append({"utt": f"en_{sid}_{os.path.splitext(fname)[0]}", "raw": p, "spk": sid, "lang": "en", "source": "libritts_r",
                         "style": "read", "gender": "", "emotional": False, "text_raw": text, "dur_raw": dur})
        self.report["libritts_r"] = {"speakers": len(spk_ids), "hours": round(total / 3600, 2)}
        log(f"LibriTTS-R selected {sum(1 for r in rows if r['lang']=='en')} clips = {total/3600:.2f} h from {len(spk_ids)} speakers")
        self.write_manifest(rows)

    def _select_indicvoices_r(self) -> list[dict]:
        from datasets import load_dataset, Audio  # type: ignore
        rows = []
        cap = self.a.indicvoices_hours * 3600.0
        total = 0.0
        try:
            cfgs = None
            try:
                from datasets import get_dataset_config_names
                cfgs = get_dataset_config_names("ai4bharat/indicvoices_r", token=self.token)
            except Exception:
                pass
            cfg = next((c for c in (cfgs or []) if "urdu" in c.lower() or c.lower() in ("ur", "urd")), "urdu")
            ds = load_dataset("ai4bharat/indicvoices_r", cfg, split="train", streaming=True, token=self.token)
            ds = ds.cast_column("audio", Audio(decode=False))
            spk_map: dict[str, int] = {}
            base = 100
            for ex in ds:
                if total >= cap:
                    break
                text = str(ex.get("text") or ex.get("transcript") or "").strip()
                if not text:
                    continue
                spk_key = str(ex.get("speaker_id") or ex.get("speaker") or "ivr")
                sid = spk_map.setdefault(spk_key, base + len(spk_map))
                b = ex["audio"]["bytes"]
                import soundfile as sf
                info = sf.info(io.BytesIO(b))
                dur = info.frames / info.samplerate
                if not (1.5 <= dur <= 20.0):
                    continue
                d = os.path.join(self.raw, "ivr", str(sid))
                os.makedirs(d, exist_ok=True)
                p = os.path.join(d, f"ivr_{len(rows):06d}.wav")
                open(p, "wb").write(b)
                total += dur
                rows.append({"utt": f"ivr_{sid}_{len(rows):06d}", "raw": p, "spk": sid, "lang": "ur", "source": "indicvoices_r",
                             "style": "ivr", "gender": str(ex.get("gender", "")), "emotional": False, "text_raw": text, "dur_raw": dur})
            self.report["indicvoices_r"] = {"hours": round(total / 3600, 2), "speakers": len(spk_map), "config": cfg}
        except Exception as e:
            log(f"IndicVoices-R skipped: {e!r}")
        return rows

    # -- step: audio -------------------------------------------------------------
    def step_audio(self) -> None:
        import soundfile as sf
        import librosa
        try:
            import soxr  # type: ignore
        except Exception:
            soxr = None
        rows = self.read_manifest()
        out_rows = []
        stats = collections.Counter()
        for i, r in enumerate(rows):
            try:
                y, sr = sf.read(r["raw"], dtype="float32", always_2d=False)
            except Exception as e:
                stats["decode_error"] += 1
                continue
            if y.ndim > 1:
                y = y.mean(axis=1)
            if sr != SR:
                y = soxr.resample(y, sr, SR, quality="VHQ") if soxr is not None else librosa.resample(y, orig_sr=sr, target_sr=SR)
            peak = float(np.max(np.abs(y))) if y.size else 0.0
            if peak <= 1e-4:
                stats["silent"] += 1
                continue
            y = y / peak * (10 ** (-1 / 20))                      # -1 dBFS
            yt, idx = librosa.effects.trim(y, top_db=40)
            pad = int(0.1 * SR)
            s0, s1 = max(0, idx[0] - pad), min(len(y), idx[1] + pad)
            y = y[s0:s1]
            dur = len(y) / SR
            if dur < 1.5:
                stats["too_short"] += 1
                continue
            if dur > 20.0:
                parts = self._split_long(y, r["text_raw"])
                if not parts:
                    stats["too_long_dropped"] += 1
                    continue
                stats["split"] += 1
            else:
                parts = [(y, r["text_raw"])]
            for k, (seg, txt) in enumerate(parts):
                utt = r["utt"] if len(parts) == 1 else f"{r['utt']}_p{k}"
                rel = f"{r['spk']}/{utt}.wav"
                p = os.path.join(self.audio, rel)
                os.makedirs(os.path.dirname(p), exist_ok=True)
                sf.write(p, (np.clip(seg, -1, 1) * 32767).astype(np.int16), SR, subtype="PCM_16")
                nr = dict(r)
                nr.update({"utt": utt, "wav": rel, "dur": round(len(seg) / SR, 3), "text_raw": txt,
                           "text": normalize(txt) if r["lang"] == "ur" else txt})
                out_rows.append(nr)
            if (i + 1) % 1000 == 0:
                log(f"  processed {i+1}/{len(rows)}")
        self.report["audio"] = dict(stats)
        self.report["hours_after_audio"] = self._hours(out_rows)
        self.write_manifest(out_rows)
        shutil.rmtree(self.raw, ignore_errors=True)   # free disk
        log(f"audio step: {len(out_rows)} clips, {self.report['hours_after_audio']}, stats={dict(stats)}")

    @staticmethod
    def _split_long(y: np.ndarray, text: str, max_s: float = 20.0) -> list[tuple[np.ndarray, str]]:
        """Split at silences >= 300 ms; keep only if the sentence count matches the segment count."""
        import librosa
        intervals = librosa.effects.split(y, top_db=35, frame_length=2048, hop_length=512)
        if len(intervals) < 2:
            return []
        gaps = [(intervals[i + 1][0] - intervals[i][1]) / SR for i in range(len(intervals) - 1)]
        cuts = [i for i, g in enumerate(gaps) if g >= 0.3]
        sents = [s for s in re.split(r"(?<=[۔؟!?.])\s+", text.strip()) if s]
        if not cuts or len(cuts) + 1 != len(sents):
            return []
        segs = []
        start = 0
        bounds = [intervals[c][1] + int(0.15 * SR) for c in cuts] + [len(y)]
        for b, s in zip(bounds, sents):
            seg = y[start:b]
            if len(seg) / SR > max_s or len(seg) / SR < 1.5:
                return []
            segs.append((seg, s))
            start = max(0, b - int(0.1 * SR))
        return segs

    @staticmethod
    def _hours(rows: list[dict]) -> dict:
        h = collections.defaultdict(float)
        for r in rows:
            h[f"spk{r['spk']}_{r['lang']}"] += r.get("dur", r.get("dur_raw", 0))
        return {k: round(v / 3600, 3) for k, v in sorted(h.items())}

    # -- step: asr ---------------------------------------------------------------
    def step_asr(self) -> None:
        from eval.asr import WhisperASR, cer, wer, scoring_normalize_ur
        rows = self.read_manifest()
        if self.a.skip_asr:
            log("ASR filter skipped (--skip-asr)")
            self.report["asr"] = {"skipped": True}
            return
        asr = WhisperASR(self.a.asr_model, batch_size=self.a.asr_batch)
        # ASR floor on FLEURS ur_pk test
        floor = self._fleurs_floor(asr)
        thresh = max(0.25, floor + 0.15)
        self.report["asr"] = {"model": self.a.asr_model, "fleurs_floor_cer": round(floor, 4), "threshold": round(thresh, 4)}
        log(f"ASR floor (FLEURS ur_pk test CER) = {floor:.3f}; drop threshold = {thresh:.3f}")
        kept, dropped = [], collections.Counter()
        cers = []
        B = 64
        for lang in ("ur", "en"):
            sub = [r for r in rows if r["lang"] == lang]
            for i in range(0, len(sub), B):
                batch = sub[i:i + B]
                hyps = asr.transcribe([os.path.join(self.audio, r["wav"]) for r in batch], language=lang)
                for r, hyp in zip(batch, hyps):
                    c = cer(r["text"], hyp, lang=lang)
                    r["asr_cer"] = round(c, 4)
                    r["asr_hyp"] = hyp
                    cers.append(c)
                    lim = thresh if lang == "ur" else 0.25
                    if not math.isnan(c) and c <= lim:
                        kept.append(r)
                    else:
                        dropped[f"spk{r['spk']}"] += 1
                        try:
                            os.remove(os.path.join(self.audio, r["wav"]))
                        except OSError:
                            pass
                if (i // B) % 10 == 0:
                    log(f"  asr {lang}: {min(i+B, len(sub))}/{len(sub)}")
        hist = np.histogram([c for c in cers if not math.isnan(c)], bins=[0, .05, .1, .15, .2, .25, .3, .4, .5, 1.0, 10])[0].tolist()
        self.report["asr"].update({"dropped_per_speaker": dict(dropped), "kept": len(kept), "cer_histogram": hist,
                                   "hours_after_asr": self._hours(kept)})
        self.write_manifest(kept)
        log(f"ASR filter kept {len(kept)}/{len(rows)}; dropped {dict(dropped)}")

    def _fleurs_rows(self, n: int) -> list[tuple[str, np.ndarray]]:
        """(raw transcription, 16 kHz audio) for the first n FLEURS ur_pk test clips.

        google/fleurs is a script-based dataset (not loadable with datasets>=4), so the TSV and the
        audio tarball are fetched directly from the Hub: data/ur_pk/test.tsv + data/ur_pk/audio/test.tar.gz."""
        import soundfile as sf
        import librosa
        from huggingface_hub import hf_hub_download
        tsv = hf_hub_download("google/fleurs", "data/ur_pk/test.tsv", repo_type="dataset", token=self.token)
        tar_path = hf_hub_download("google/fleurs", "data/ur_pk/audio/test.tar.gz", repo_type="dataset", token=self.token)
        want: dict[str, str] = {}
        with open(tsv, encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 3:
                    want[parts[1]] = parts[2]      # id, filename, raw transcription, transcription, ...
        rows: list[tuple[str, np.ndarray]] = []
        with tarfile.open(tar_path, "r:gz") as tf:
            for m in tf:
                if not m.isfile():
                    continue
                name = os.path.basename(m.name)
                if name not in want:
                    continue
                fobj = tf.extractfile(m)
                if fobj is None:
                    continue
                y, sr = sf.read(io.BytesIO(fobj.read()), dtype="float32")
                if y.ndim > 1:
                    y = y.mean(axis=1)
                if sr != 16000:
                    y = librosa.resample(y, orig_sr=sr, target_sr=16000)
                rows.append((want[name], y))
                if len(rows) >= n:
                    break
        return rows

    def _fleurs_floor(self, asr) -> float:
        from eval.asr import cer
        try:
            rows = self._fleurs_rows(self.a.fleurs_n)
            if not rows:
                raise RuntimeError("no FLEURS rows")
            refs = [r for r, _ in rows]
            auds = [y for _, y in rows]
            hyps = []
            for i in range(0, len(auds), 32):
                hyps += asr.transcribe(auds[i:i + 32], language="ur", sr=16000)
            cs = [cer(r, h) for r, h in zip(refs, hyps)]
            with open(os.path.join(self.eval_dir, "fleurs_test.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(normalize_chars(t) for t in refs) + "\n")
            # keep a few real clips as naturalness references for the evaluation
            import soundfile as sf
            ref_dir = os.path.join(self.eval_dir, "fleurs_ref")
            os.makedirs(ref_dir, exist_ok=True)
            for i, (_, y) in enumerate(rows[:10]):
                sf.write(os.path.join(ref_dir, f"fleurs_{i:02d}.wav"), y, 16000)
            return float(np.nanmean(cs))
        except Exception as e:
            log(f"FLEURS floor unavailable ({e!r}); assuming 0.10")
            return 0.10

    # -- step: audit -------------------------------------------------------------
    def step_audit(self) -> None:
        from eval.audit_data import audit_dir
        res = audit_dir(self.audio, per_spk=40)
        self.report["bandwidth"] = res
        self._save_json(os.path.join(self.out, "bandwidth.json"), res)
        for spk, v in res.items():
            log(f"  speaker {spk}: rolloff99 {v['rolloff99_hz']:.0f} Hz  HF>7k {v['hf_ratio_pct_above_7k']}%  {'BAND-LIMITED' if v['band_limited'] else 'ok'}")

    # -- step: lexicon -----------------------------------------------------------
    def step_lexicon(self) -> None:
        rows = self.read_manifest()
        words_path = os.path.join(self.out, "rasa_words.txt")
        words = collections.Counter()
        for r in rows:
            if r["lang"] == "ur":
                for w in r["text"].split():
                    w = re.sub(r"[^ؠ-ٟٮ-ۓەۡ-ۯۺ-ۿ]", "", w)
                    if w:
                        words[w] += 1
        with open(words_path, "w", encoding="utf-8") as f:
            for w, c in words.most_common():
                f.write(f"{w}\t{c}\n")
        self.report["rasa_word_types"] = len(words)
        py = sys.executable
        cmd = [py, os.path.join(ROOT, "scripts", "02_build_lexicon.py"), "--words", words_path, "--wiki-top", str(self.a.wiki_words),
               "--out-dir", os.path.join(ROOT, "lughaat_tts", "data")]
        if self.a.llm_max_words > 0:
            cmd += ["--llm-max-words", str(self.a.llm_max_words)]
        log("building lexicon: " + " ".join(cmd))
        subprocess.run(cmd, check=True)
        cmd = [py, os.path.join(ROOT, "scripts", "03_train_g2p.py"), "--epochs", str(self.a.g2p_epochs)]
        log("training neural G2P: " + " ".join(cmd))
        subprocess.run(cmd, check=True)
        rep = os.path.join(ROOT, "reports", "g2p_metrics.json")
        if os.path.exists(rep):
            self.report["g2p"] = self._load_json(rep, {})

    # -- step: phonemize ---------------------------------------------------------
    def step_phonemize(self) -> None:
        from lughaat_tts.codeswitch import MixedFrontend
        from lughaat_tts.g2p import UrduG2P
        rows = self.read_manifest()
        fe = MixedFrontend(urdu=UrduG2P())
        try:
            from misaki import en  # type: ignore
            en_g2p = en.G2P(trf=False, british=False)
        except Exception as e:
            raise SystemExit(f"misaki[en] is required to phonemize the English replay data: {e!r}")
        kept, bad = [], collections.Counter()
        freq = collections.Counter()
        for i, r in enumerate(rows):
            try:
                if r["lang"] == "ur":
                    ps = fe(r["text"], english_accent="pakistani")
                else:
                    ps, _ = en_g2p(r["text"])
                    ps = "".join(c for c in ps if c in VOCAB or c == " ").strip()
            except Exception as e:
                bad["exception"] += 1
                continue
            oov = sorted({c for c in ps if c not in VOCAB})
            if oov:
                bad["oov"] += 1
                continue
            if not ps or len(ps) < 3:
                bad["empty"] += 1
                continue
            if len(ps) > 510:
                bad["too_long"] += 1
                continue
            r["phonemes"] = ps
            for c in ps:
                freq[c] += 1
            kept.append(r)
            if (i + 1) % 2000 == 0:
                log(f"  phonemized {i+1}/{len(rows)}")
        self.report["phonemize"] = {"kept": len(kept), "dropped": dict(bad), "g2p_stats": fe.urdu.stats,
                                    "phoneme_frequency": dict(freq.most_common())}
        self.write_manifest(kept)
        log(f"phonemize kept {len(kept)}/{len(rows)}; dropped {dict(bad)}; g2p {fe.urdu.stats}")

    # -- step: lists -------------------------------------------------------------
    def step_lists(self) -> None:
        rows = self.read_manifest()
        rng = random.Random(42)
        by_key = collections.defaultdict(list)
        for r in rows:
            by_key[(r["spk"], r["lang"])].append(r)
        val, train = [], []
        n_val = max(100, int(0.02 * len(rows)))
        for key, lst in sorted(by_key.items()):
            rng.shuffle(lst)
            k = max(1, round(n_val * len(lst) / len(rows)))
            val += lst[:k]
            train += lst[k:]
        hours = collections.defaultdict(float)
        for r in rows:
            hours[r["lang"]] += r["dur"]
        tot = sum(hours.values())
        mix = {k: round(100 * v / tot, 1) for k, v in hours.items()}
        self.report["mix_percent_by_hours"] = mix
        self.report["hours_final"] = self._hours(rows)
        self.report["train_lines"] = len(train)
        self.report["val_lines"] = len(val)
        if mix.get("en", 0) < 10 or mix.get("en", 0) > 25:
            log(f"WARNING: English share is {mix.get('en')}% (target 10-25%)")
        with open(os.path.join(self.out, "train_list.txt"), "w", encoding="utf-8", newline="\n") as f:
            for r in train:
                f.write(f"{r['wav']}|{r['phonemes']}|{r['spk']}\n")
        with open(os.path.join(self.out, "val_list.txt"), "w", encoding="utf-8", newline="\n") as f:
            for r in val:
                f.write(f"{r['wav']}|{r['phonemes']}|{r['spk']}\n")
        # reference clips for voicepacks: neutral, Rasa, per speaker
        refs = {str(s): [r["wav"] for r in rows if r["spk"] == s and r["source"] == "rasa" and not r["emotional"]] for s in RASA_SPEAKERS.values()}
        self._save_json(os.path.join(self.out, "voicepack_refs.json"), refs)
        # held-out in-domain eval texts
        with open(os.path.join(self.eval_dir, "val_texts_ur.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(r["text"] for r in val if r["lang"] == "ur") + "\n")
        with open(os.path.join(self.eval_dir, "val_texts_en.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(r["text"] for r in val if r["lang"] == "en") + "\n")
        log(f"lists: train {len(train)} val {len(val)}; mix by hours {mix}; hours {self.report['hours_final']}")

    # -- step: ood ---------------------------------------------------------------
    def step_ood(self) -> None:
        from lughaat_tts.codeswitch import MixedFrontend
        fe = MixedFrontend()
        rows = self.read_manifest()
        train_texts = {r["text"] for r in rows}
        ur_sents = self._wikipedia_sentences(self.a.ood_urdu + 1200, exclude=train_texts)
        self.report["ood_wikipedia_sentences"] = len(ur_sents)
        with open(os.path.join(self.eval_dir, "wiki_sentences.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(ur_sents) + "\n")
        # English OOD: LibriTTS texts not used + extra sentences
        en_sents = self._libritts_texts(self.a.ood_english + 200, exclude=train_texts)
        rng = random.Random(7)
        mixed_pool = [self._make_mixed(s, rng) for s in ur_sents[-1200:]]
        mixed_pool = [m for m in mixed_pool if m]
        mixed_ood, mixed_eval = mixed_pool[:self.a.ood_mixed], mixed_pool[self.a.ood_mixed:self.a.ood_mixed + 200]
        ur_ood = ur_sents[: self.a.ood_urdu]
        en_ood, en_eval = en_sents[: self.a.ood_english], en_sents[self.a.ood_english: self.a.ood_english + 200]
        lines, dropped = [], 0
        try:
            from misaki import en  # type: ignore
            en_g2p = en.G2P(trf=False, british=False)
        except Exception:
            en_g2p = None
        for s in ur_ood:
            ps = fe(s, english_accent="pakistani")
            if 20 <= len(ps) <= 500 and all(c in VOCAB for c in ps):
                lines.append(ps)
            else:
                dropped += 1
        for s in en_ood:
            if en_g2p is None:
                break
            ps, _ = en_g2p(s)
            ps = "".join(c for c in ps if c in VOCAB or c == " ").strip()
            if 20 <= len(ps) <= 500:
                lines.append(ps)
        for s in mixed_ood:
            ps = fe(s, english_accent="pakistani")
            if 20 <= len(ps) <= 500 and all(c in VOCAB for c in ps):
                lines.append(ps)
        assert len(lines) > 500, "OOD_texts.txt must not be (nearly) empty: Stage 2 silently disables adversarial training"
        with open(os.path.join(self.out, "OOD_texts.txt"), "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(lines) + "\n")
        with open(os.path.join(self.eval_dir, "synthetic_mixed_test.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(mixed_eval) + "\n")
        with open(os.path.join(self.eval_dir, "english_test.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(en_eval) + "\n")
        self.report["ood"] = {"lines": len(lines), "urdu": len(ur_ood), "english": len(en_ood), "mixed": len(mixed_ood), "dropped": dropped}
        log(f"OOD_texts.txt: {len(lines)} lines (ur {len(ur_ood)}, en {len(en_ood)}, mixed {len(mixed_ood)})")

    def _wikipedia_sentences(self, n: int, exclude: set) -> list[str]:
        out: list[str] = []
        try:
            from datasets import load_dataset  # type: ignore
            ds = load_dataset("wikimedia/wikipedia", "20231101.ur", split="train", streaming=True, token=self.token)
            rng = random.Random(3)
            for ex in ds:
                text = ex.get("text", "")
                for para in text.split("\n"):
                    para = para.strip()
                    if len(para) < 40 or "==" in para:
                        continue
                    for s in re.split(r"(?<=[۔؟!])\s+", para):
                        s = normalize_chars(s)
                        if not (20 <= len(s) <= 200) or has_latin(s) or re.search(r"[\[\]{}()<>|/=\\*_#]", s):
                            continue
                        if not s.endswith(("۔", "؟", "!")):
                            continue
                        if s in exclude:
                            continue
                        if re.search(r"\d", s) and rng.random() < 0.7:
                            continue
                        out.append(s)
                if len(out) >= n * 3:
                    break
            rng.shuffle(out)
        except Exception as e:
            log(f"Urdu Wikipedia unavailable ({e!r})")
        # dedupe
        seen, res = set(), []
        for s in out:
            if s not in seen:
                seen.add(s)
                res.append(s)
        return res[:n]

    def _libritts_texts(self, n: int, exclude: set) -> list[str]:
        out = []
        try:
            from datasets import load_dataset  # type: ignore
            ds = load_dataset("mythicinfinity/libritts_r", "clean", split="train.clean.360", streaming=True, token=self.token)
            ds = ds.remove_columns([c for c in ("audio",) if c in ds.column_names]) if hasattr(ds, "column_names") and ds.column_names else ds
            for ex in ds:
                t = str(ex.get("text_normalized") or "").strip()
                if 30 <= len(t) <= 160 and t not in exclude and not has_arabic_script(t):
                    out.append(t)
                if len(out) >= n:
                    break
        except Exception as e:
            log(f"LibriTTS-R texts unavailable ({e!r})")
        return out[:n]

    @staticmethod
    def _make_mixed(sentence: str, rng: random.Random) -> Optional[str]:
        words = sentence.split()
        idx = [i for i, w in enumerate(words) if w.strip("۔،؟!") in SWAP_TARGETS]
        if not idx:
            return None
        k = min(len(idx), rng.choice([1, 1, 2, 2, 3]))
        for i in rng.sample(idx, k):
            core = words[i].strip("۔،؟!")
            tail = words[i][len(core):]
            words[i] = SWAP_TARGETS[core] + tail
        return " ".join(words)

    # -- step: pack --------------------------------------------------------------
    def step_pack(self) -> None:
        tar_path = os.path.join(self.out, "audio.tar")
        if not os.path.exists(tar_path):
            log("packing audio.tar ...")
            with tarfile.open(tar_path, "w") as tf:
                tf.add(self.audio, arcname="audio")
        self.report["audio_tar_gb"] = round(os.path.getsize(tar_path) / 1e9, 3)
        self._write_report_md()
        if self.a.upload and self.a.work_repo:
            self.upload()

    def _write_report_md(self) -> None:
        r = self.report
        lines = ["# Data report", "", f"Generated {time.strftime('%Y-%m-%d %H:%M')}", ""]
        lines += ["## Hours", "", "| set | hours |", "|---|---|"]
        for k, v in (r.get("hours_final") or r.get("hours_after_audio") or {}).items():
            lines.append(f"| {k} | {v} |")
        lines += ["", f"Mix by hours: {r.get('mix_percent_by_hours')}", "",
                  f"Rasa: {json.dumps(r.get('rasa', {}), ensure_ascii=False)[:2000]}", "",
                  f"LibriTTS-R: {r.get('libritts_r')}", "",
                  f"Audio processing: {r.get('audio')}", "",
                  f"ASR filter: {json.dumps({k: v for k, v in (r.get('asr') or {}).items()}, ensure_ascii=False)}", "",
                  "## Bandwidth audit", "", "| speaker | rolloff99 Hz | HF>7k % | band-limited |", "|---|---|---|---|"]
        for spk, v in (r.get("bandwidth") or {}).items():
            lines.append(f"| {spk} | {v['rolloff99_hz']} | {v['hf_ratio_pct_above_7k']} | {v['band_limited']} |")
        lines += ["", f"Phonemize: kept {r.get('phonemize', {}).get('kept')} dropped {r.get('phonemize', {}).get('dropped')}", "",
                  f"G2P: {r.get('g2p')}", "", f"OOD: {r.get('ood')}", "",
                  "## Phoneme frequency (top 60)", ""]
        pf = (r.get("phonemize") or {}).get("phoneme_frequency") or {}
        lines.append(" ".join(f"`{k}`:{v}" for k, v in list(pf.items())[:60]))
        with open(os.path.join(self.out, "data_report.md"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    def upload(self) -> None:
        from huggingface_hub import HfApi
        api = HfApi(token=self.token)
        api.create_repo(self.a.work_repo, repo_type="dataset", private=True, exist_ok=True)
        files = ["train_list.txt", "val_list.txt", "OOD_texts.txt", "data_report.md", "data_report.json", "manifest.jsonl",
                 "bandwidth.json", "voicepack_refs.json"]
        for fn in files:
            p = os.path.join(self.out, fn)
            if os.path.exists(p):
                api.upload_file(path_or_fileobj=p, path_in_repo=f"data/{fn}", repo_id=self.a.work_repo, repo_type="dataset")
        api.upload_folder(folder_path=self.eval_dir, path_in_repo="data/eval", repo_id=self.a.work_repo, repo_type="dataset")
        tar_path = os.path.join(self.out, "audio.tar")
        log(f"uploading audio.tar ({os.path.getsize(tar_path)/1e9:.2f} GB) ...")
        api.upload_file(path_or_fileobj=tar_path, path_in_repo="data/audio.tar", repo_id=self.a.work_repo, repo_type="dataset")
        # frontend artefacts (lexicon, g2p model) so later sessions reuse the same frontend
        api.upload_folder(folder_path=os.path.join(ROOT, "lughaat_tts", "data"), path_in_repo="frontend/data", repo_id=self.a.work_repo, repo_type="dataset")
        log("upload complete")

    # -- driver --------------------------------------------------------------------
    def run(self, step: str) -> None:
        steps = STEPS if step == "all" else [step]
        for s in steps:
            if self.done(s) and not self.a.force:
                log(f"step {s}: already done")
                continue
            log(f"=== step {s} ===")
            getattr(self, f"step_{s}")()
            self.mark(s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--step", default="all", choices=["all"] + STEPS)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--hf-token", default=None)
    ap.add_argument("--work-repo", default=os.environ.get("WORK_REPO"))
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--urdu-hours-per-speaker", type=float, default=float(os.environ.get("URDU_HOURS_PER_SPEAKER", 6)))
    ap.add_argument("--english-hours", type=float, default=float(os.environ.get("ENGLISH_REPLAY_HOURS", 3)))
    ap.add_argument("--english-speakers", type=int, default=20, help="target speaker count; sets the per-speaker cap (hours / N); more speakers are used if needed")
    ap.add_argument("--use-indicvoices-r", type=int, default=int(os.environ.get("USE_INDICVOICES_R", 0)))
    ap.add_argument("--indicvoices-hours", type=float, default=3.0)
    ap.add_argument("--asr-model", default=os.environ.get("ASR_FILTER_MODEL", "openai/whisper-large-v3-turbo"))
    ap.add_argument("--asr-batch", type=int, default=16)
    ap.add_argument("--skip-asr", action="store_true")
    ap.add_argument("--fleurs-n", type=int, default=300)
    ap.add_argument("--wiki-words", type=int, default=50000)
    ap.add_argument("--llm-max-words", type=int, default=int(os.environ.get("LLM_LEXICON_MAX_WORDS", 0)),
                    help="send up to this many new words to the LLM for pronunciation candidates (0 = off)")
    ap.add_argument("--g2p-epochs", type=int, default=40)
    ap.add_argument("--ood-urdu", type=int, default=8000)
    ap.add_argument("--ood-english", type=int, default=1000)
    ap.add_argument("--ood-mixed", type=int, default=1000)
    a = ap.parse_args()
    Prep(a.out, a).run(a.step)


if __name__ == "__main__":
    main()

"""Gradio demo for Kokoro-82M Urdu (runs on free CPU hardware). {MODEL_REPO} is filled by 99_upload.py."""
from __future__ import annotations

import os

import gradio as gr
import numpy as np

REPO = os.environ.get("LUGHAAT_TTS_REPO", "{MODEL_REPO}")
TOKEN = os.environ.get("HF_TOKEN")  # Space secret; needed only while the model repo is private

_tts = None


def get_tts():
    global _tts
    if _tts is None:
        from lughaat_tts import UrduPipeline
        _tts = UrduPipeline(voice="uf_rasa", device="cpu", repo_id=REPO, token=TOKEN)
    return _tts


EXAMPLES = [
    ["پاکستان ایک خوبصورت ملک ہے۔", "uf_rasa", 1.0, "auto", False],
    ["آج کی meeting کینسل ہو گئی ہے، please email check کریں۔", "um_rasa", 1.0, "auto", True],
    ["ٹرین صبح ۷:۴۵ پر روانہ ہوئی اور اس کی قیمت ۲۵۰۰ روپے تھی۔", "uf_rasa", 1.0, "auto", True],
    ["Good morning, how are you today?", "uf_rasa", 1.0, "native", False],
]


def speak(text: str, voice: str, speed: float, accent: str, show_phonemes: bool):
    text = (text or "").strip()
    if not text:
        return None, ""
    tts = get_tts()
    audio = tts(text, voice=voice, speed=float(speed), english_accent=accent)
    info = ""
    if show_phonemes:
        r = tts.phonemize(text, english_accent=accent)
        info = r.phonemes + "\n\n" + " ".join(f"{t.text}/{t.tag}" for t in r.tokens if t.tag != "punct")
    return (24000, (np.clip(audio, -1, 1) * 32767).astype(np.int16)), info


with gr.Blocks(title="Lughaat-TTS-82M") as demo:
    gr.Markdown(f"# Lughaat-TTS-82M\nUrdu, English and Urdu–English mixed text-to-speech (Kokoro-82M fine-tune), 82 M parameters, CPU. Model: `{REPO}`")
    with gr.Row():
        with gr.Column():
            text = gr.Textbox(label="Text (Urdu / English / mixed)", lines=4, rtl=True, value=EXAMPLES[0][0])
            voice = gr.Dropdown(["uf_rasa", "um_rasa"], value="uf_rasa", label="Voice")
            speed = gr.Slider(0.6, 1.5, value=1.0, step=0.05, label="Speed")
            accent = gr.Radio(["auto", "pakistani", "native"], value="auto", label="English accent")
            show = gr.Checkbox(label="Show phonemes and language tags", value=False)
            btn = gr.Button("Speak", variant="primary")
        with gr.Column():
            out = gr.Audio(label="Output (24 kHz)")
            ph = gr.Textbox(label="Phonemes", lines=4)
    btn.click(speak, [text, voice, speed, accent, show], [out, ph])
    gr.Examples(EXAMPLES, [text, voice, speed, accent, show])

if __name__ == "__main__":
    demo.queue().launch()

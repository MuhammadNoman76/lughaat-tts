"""listening_test.html: reference audio (when available) next to each candidate + a rating form (plan 9)."""
from __future__ import annotations

import html
import json
import os


def write_listening_test(out_dir: str, rows: list[dict], title: str = "Lughaat-TTS listening test",
                         reference_wavs: dict[str, str] | None = None) -> str:
    """rows: per-sentence dicts from evaluate.py (need text, voice, accent, wav, hyp, set)."""
    by_text: dict[str, list[dict]] = {}
    for r in rows:
        if r.get("wav"):
            by_text.setdefault(r["text"], []).append(r)
    parts = [f"""<!doctype html><html lang="ur"><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;background:#fafafa;color:#222}}
.card{{background:#fff;border:1px solid #ddd;border-radius:8px;padding:1rem;margin:1rem 0}} .ur{{font-size:1.4rem;direction:rtl;text-align:right}}
.row{{display:flex;flex-wrap:wrap;gap:1rem;align-items:center;margin:.5rem 0}} .hyp{{color:#666;font-size:.9rem;direction:rtl}}
select{{margin-left:.5rem}} .tag{{font-size:.8rem;background:#eee;border-radius:4px;padding:.1rem .4rem}}</style></head><body>
<h1>{html.escape(title)}</h1><p>Rate each clip 1 (bad) to 5 (excellent) for intelligibility and naturalness. Click <b>Export ratings</b> when done.</p>"""]
    for i, (text, items) in enumerate(by_text.items()):
        parts.append(f'<div class="card"><div class="ur">{html.escape(text)}</div>')
        if reference_wavs and text in reference_wavs:
            parts.append(f'<div class="row"><span class="tag">reference (real)</span><audio controls src="{html.escape(reference_wavs[text])}"></audio></div>')
        for j, r in enumerate(items):
            rel = os.path.relpath(r["wav"], out_dir).replace(os.sep, "/")
            parts.append(f'<div class="row"><span class="tag">{html.escape(r["voice"])} / {html.escape(r.get("accent",""))} / {html.escape(r["set"])}</span>'
                         f'<audio controls preload="none" src="{html.escape(rel)}"></audio>'
                         f'<label>intelligibility<select data-id="{i}-{j}-i"><option></option>{"".join(f"<option>{k}</option>" for k in range(1,6))}</select></label>'
                         f'<label>naturalness<select data-id="{i}-{j}-n"><option></option>{"".join(f"<option>{k}</option>" for k in range(1,6))}</select></label>'
                         f'<div class="hyp">ASR: {html.escape(r.get("hyp",""))}</div></div>')
        parts.append("</div>")
    parts.append("""<button onclick="exportRatings()">Export ratings</button><pre id="out"></pre>
<script>function exportRatings(){const o={};document.querySelectorAll('select').forEach(s=>{if(s.value)o[s.dataset.id]=s.value});
document.getElementById('out').textContent=JSON.stringify(o,null,1)}</script></body></html>""")
    path = os.path.join(out_dir, "listening_test.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    return path


__all__ = ["write_listening_test"]

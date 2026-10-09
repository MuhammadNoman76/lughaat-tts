"""Default Hugging Face repo that holds the weights, voices and G2P model.

``scripts/99_upload.py`` rewrites DEFAULT_REPO_ID with the real ``<HF_USERNAME>/lughaat-tts-82m``
before uploading the package to the model repo, so ``UrduPipeline()`` works with no
arguments after ``pip install git+https://huggingface.co/<HF_USERNAME>/lughaat-tts-82m``.
It can always be overridden with the ``LUGHAAT_TTS_REPO`` environment variable or the
``repo_id`` constructor argument.
"""
DEFAULT_REPO_ID = ""   # filled in by scripts/99_upload.py
MODEL_FILENAME = "lughaat-tts-82m.pth"
CONFIG_FILENAME = "config.json"
G2P_FILENAME = "lughaat_tts/data/g2p_model.pt"
VOICES = ("uf_rasa", "um_rasa")

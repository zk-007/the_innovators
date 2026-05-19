"""
Hugging Face Spaces entrypoint + local fallback.
Judges: open your Space URL in any browser (no install).
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

import app.app as gradio_app
from app.model import SignLanguageModel
from app.predictor import WebcamPredictor

print("Loading ASL model…")
gradio_app.model = SignLanguageModel()
gradio_app.predictor = WebcamPredictor(gradio_app.model)
print("Ready:", gradio_app.model.metadata())

demo = gradio_app.build_ui()

if __name__ == "__main__":
    demo.queue().launch()

"""
The Innovators — Phase 2 live demo
Live webcam ASL classifier + word spelling + two-way text bridge.
"""

from __future__ import annotations

import argparse

import gradio as gr
import numpy as np

from app.config import CONFIDENCE_THRESHOLD, ROOT, SMOOTHING_WINDOW
from app.model import SignLanguageModel
from app.spelling import SpellingState

model: SignLanguageModel | None = None
speller = SpellingState()
hearing_to_deaf = ""
deaf_to_hearing = ""


def _top3_bar(prob_map: dict[str, float]) -> str:
    top = sorted(prob_map.items(), key=lambda x: x[1], reverse=True)[:3]
    return " | ".join(f"{k}: {v:.0%}" for k, v in top)


def predict_frame(image: np.ndarray | None, threshold: float, enable_spelling: bool):
    global speller, deaf_to_hearing

    if image is None:
        return "No camera", "", speller.sentence, "Point webcam at your hand sign", deaf_to_hearing

    label, confidence, prob_map = model.predict(image)
    label = speller.smooth_label(label) if SMOOTHING_WINDOW else label

    pred_line = f"**{label}** — {confidence:.1%}\n\n{_top3_bar(prob_map)}"

    if enable_spelling:
        sentence, status = speller.update(label, confidence, threshold)
        deaf_to_hearing = sentence
        return pred_line, status, sentence, status, deaf_to_hearing

    return pred_line, "Spelling off — enable to build words", speller.sentence, "", deaf_to_hearing


def reset_spelling():
    speller.reset_sentence()
    return "", "Sentence cleared", ""


def hearing_user_types(text: str):
    global hearing_to_deaf
    hearing_to_deaf = text or ""
    return hearing_to_deaf


def build_ui() -> gr.Blocks:
    meta = model.metadata()
    title = "# The Silent Gap — The Innovators\n### Live ASL → Text (Phase 2)"

    with gr.Blocks(title="The Innovators — ASL Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown(title)
        gr.Markdown(
            f"**Model:** `{meta['model_name']}` · fold {meta['fold']} · "
            f"val acc {meta['best_val_acc']:.2%} · {meta['device']}"
        )

        with gr.Tabs():
            with gr.Tab("Live webcam + spelling"):
                gr.Markdown(
                    "Show a hand sign to the camera. **Hold a letter steady ~1.2s** to add it. "
                    "Sign **space** or **del** to add space / delete. This matches the hackathon "
                    "*word spelling interface* requirement."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        webcam = gr.Image(sources=["webcam"], type="numpy", label="Webcam")
                        threshold = gr.Slider(0.3, 0.95, value=CONFIDENCE_THRESHOLD, step=0.05, label="Confidence threshold")
                        spelling_on = gr.Checkbox(value=True, label="Enable hold-to-spell")
                        reset_btn = gr.Button("Clear sentence")
                    with gr.Column(scale=1):
                        prediction = gr.Markdown(label="Current sign")
                        status = gr.Textbox(label="Status", interactive=False)
                        sentence = gr.Textbox(label="Your message (deaf user → hearing user)", lines=4, interactive=False)

                reset_btn.click(reset_spelling, outputs=[sentence, status, sentence])

                webcam.stream(
                    fn=predict_frame,
                    inputs=[webcam, threshold, spelling_on],
                    outputs=[prediction, status, sentence, status, sentence],
                    time_limit=3600,
                    stream_every=0.25,
                )

            with gr.Tab("Two-way bridge"):
                gr.Markdown(
                    "**Hearing → Deaf:** type text below.\n\n"
                    "**Deaf → Hearing:** use the webcam tab; your spelled sentence appears here."
                )
                with gr.Row():
                    hearing_in = gr.Textbox(label="Hearing user types", lines=3, placeholder="Type a message for the deaf user to read…")
                    hearing_out = gr.Textbox(label="Message for deaf user", lines=3, interactive=False)
                    deaf_out = gr.Textbox(label="Deaf user signed text (from webcam tab)", lines=3, interactive=False)

                hearing_in.change(hearing_user_types, inputs=hearing_in, outputs=hearing_out)

                refresh = gr.Button("Refresh signed text")
                refresh.click(lambda: deaf_to_hearing, outputs=deaf_out)

        gr.Markdown(
            f"Team: **The Innovators** · [GitHub](https://github.com/zk-007/the_innovators) · "
            "Forman CS Club AI Hackathon 2026"
        )

    return demo


def main():
    global model
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="Create public Gradio link (+5% deployment bonus)")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    print("Loading model from", ROOT / "models")
    model = SignLanguageModel()
    print("Ready:", model.metadata())

    demo = build_ui()
    demo.queue().launch(server_name="0.0.0.0", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()

"""
The Innovators — Phase 2 live demo
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gradio as gr
import numpy as np

from app.config import (
    CONFIDENCE_THRESHOLD,
    DISPLAY_MIN_CONFIDENCE,
    MIN_MARGIN,
    REF_DIR,
    ROOT,
    STRONG_LETTERS,
)
from app.model import SignLanguageModel
from app.predictor import WebcamPredictor
from app.preprocess import MEDIAPIPE_OK
from app.spelling import SpellingState

model: SignLanguageModel | None = None
predictor: WebcamPredictor | None = None
speller = SpellingState()
deaf_to_hearing = ""

GUIDE_MD = """
### Training style (see grid)
**Dark hand silhouette** on **plain gray wall** — copy `ALL_CLASSES_GRID.png` poses.

### Best letters for live demo
**B, E, F, K, M, N, W, Y** — start here.

### Setup
- Plain light wall, **no face** in frame  
- Front light (not window behind you)  
- Hand fills **70–85%** of "Model sees" box  
- Hold letter **1 second** until it commits  
- Commit slider **0.70–0.80** for presentation  
"""


def _top5_md(prob_map: dict[str, float]) -> str:
    top = sorted(prob_map.items(), key=lambda x: x[1], reverse=True)[:5]
    return "\n".join(f"- **{k}**: {v:.1%}" for k, v in top)


def _ref_path(label: str) -> str | None:
    p = REF_DIR / f"{label}.jpg"
    return str(p) if p.exists() else None


def predict_frame(
    image: np.ndarray | None,
    threshold: float,
    enable_spelling: bool,
    mirror: bool,
    hand_crop: bool,
):
    global speller, deaf_to_hearing

    empty = None
    if image is None:
        return "No camera", "", speller.sentence, "Point webcam at hand", empty, empty, deaf_to_hearing

    try:
        label, confidence, prob_map, pil_crop, crop_note, margin = predictor.predict_from_webcam(
            image, mirror=mirror, use_hand_crop=hand_crop,
        )
        preview = np.array(pil_crop.resize((200, 200)))
        ref_img = _ref_path(label)
    except Exception as exc:
        return f"Error: {exc}", "", speller.sentence, "", empty, empty, deaf_to_hearing

    label = speller.smooth_label(label, confidence)
    uncertain = margin < MIN_MARGIN
    low_conf = confidence < DISPLAY_MIN_CONFIDENCE

    tag = ""
    if label in STRONG_LETTERS:
        tag = " ✓ reliable letter"
    if uncertain:
        tag += " ⚠️ unstable"
    if low_conf:
        tag += " ⚠️ low confidence"

    show = label if not low_conf else "nothing"
    pred_line = (
        f"## **{show}** ({confidence:.1%}){tag}\n\n"
        f"_margin {margin:.1%} · {crop_note} · smooth {predictor._prob_history.maxlen} frames_\n\n"
        f"**Top 5:**\n{_top5_md(prob_map)}"
    )
    if not low_conf and _ref_path(label):
        pred_line += f"\n\n_Copy pose from dataset reference: **{label}.jpg**_"

    label = show
    if enable_spelling:
        sentence, status = speller.update(label, confidence, threshold, margin=margin)
        deaf_to_hearing = sentence
        return pred_line, status, sentence, status, preview, ref_img, deaf_to_hearing

    deaf_to_hearing = speller.sentence
    return pred_line, "Spelling off", speller.sentence, "", preview, ref_img, deaf_to_hearing


def reset_spelling():
    speller.reset_sentence()
    predictor.reset()
    return "", "Cleared", None


def hearing_user_types(text: str):
    global hearing_to_deaf
    hearing_to_deaf = text or ""
    return hearing_to_deaf


def build_ui() -> gr.Blocks:
    meta = model.metadata()
    with gr.Blocks(title="The Innovators — ASL Demo") as demo:
        gr.Markdown("# The Silent Gap — The Innovators\n### Live ASL → Text")
        gr.Markdown(
            f"**{meta['model_name']}** · fold {meta['fold']} · "
            f"val {meta['best_val_acc']:.1%} · test ~61% · {meta['device']}"
        )

        with gr.Tabs():
            with gr.Tab("Live webcam + spelling"):
                gr.Markdown("**Model sees** should show **dark hand on gray only** — no room/door.")
                with gr.Row():
                    with gr.Column():
                        webcam = gr.Image(sources=["webcam"], type="numpy", label="Webcam")
                        mirror = gr.Checkbox(value=True, label="Mirror webcam")
                        hand_crop = gr.Checkbox(value=True, label="Hand mask + silhouette")
                        threshold = gr.Slider(
                            0.35, 0.95, value=CONFIDENCE_THRESHOLD, step=0.05,
                            label="Commit threshold (0.75 demo)",
                        )
                        spelling_on = gr.Checkbox(value=True, label="Hold-to-spell")
                        reset_btn = gr.Button("Clear sentence")
                    with gr.Column():
                        crop_preview = gr.Image(label="Model sees", type="numpy")
                        ref_preview = gr.Image(label="Dataset reference for predicted letter", type="filepath")
                        prediction = gr.Markdown()
                        status = gr.Textbox(label="Status", interactive=False)
                        sentence = gr.Textbox(label="Message", lines=3, interactive=False)

                reset_btn.click(reset_spelling, outputs=[sentence, status, ref_preview])

                webcam.stream(
                    fn=predict_frame,
                    inputs=[webcam, threshold, spelling_on, mirror, hand_crop],
                    outputs=[prediction, status, sentence, status, crop_preview, ref_preview, sentence],
                    time_limit=3600,
                    stream_every=0.22,
                )

            with gr.Tab("How to sign"):
                gr.Markdown(GUIDE_MD)
                grid = REF_DIR / "ALL_CLASSES_GRID.png"
                if grid.exists():
                    gr.Image(value=str(grid), label="All 29 classes — training dataset")

            with gr.Tab("Two-way bridge"):
                hearing_in = gr.Textbox(label="Hearing types", lines=2)
                hearing_out = gr.Textbox(label="For deaf user", lines=2, interactive=False)
                deaf_out = gr.Textbox(label="Signed text", lines=2, interactive=False)
                hearing_in.change(hearing_user_types, inputs=hearing_in, outputs=hearing_out)
                gr.Button("Refresh").click(lambda: deaf_to_hearing, outputs=deaf_out)

    return demo


def main():
    global model, predictor
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    if not MEDIAPIPE_OK:
        print("WARNING: pip install mediapipe==0.10.14")
    model = SignLanguageModel()
    predictor = WebcamPredictor(model)
    print("Ready:", model.metadata())

    demo = build_ui()
    demo.queue().launch(
        server_name="0.0.0.0", server_port=args.port, share=args.share, theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main()

---
title: The Innovators — ASL Live Demo
emoji: 🤟
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: true
license: mit
short_description: Live ASL sign language → text (webcam + spelling)
---

# The Innovators — The Silent Gap

**Forman CS Club · AI Hackathon 2026**

Live demo for judges: **open this Space in your browser**, allow webcam, and try signing.

## Judge link (after Space is running)

Your public URL will look like:

**https://huggingface.co/spaces/zk-007/the-innovators**

(Replace `zk-007` with your Hugging Face username if different.)

## How to use (30 seconds)

1. Open the Space link → wait for **Running** (first load may take 2–3 min).
2. Tab **Live webcam + spelling** → allow camera.
3. Plain wall behind you, one hand in frame.
4. Copy poses from **How to sign** tab (`ALL_CLASSES_GRID.png`).
5. Hold each letter ~1 second to spell (e.g. **B → E → E** for BEE).
6. Best letters: **B, E, F, K, M, N, W, Y**.

## Project links

- GitHub: [github.com/zk-007/the_innovators](https://github.com/zk-007/the_innovators)
- Kaggle: ConvNeXt Tiny, fold-0, ~61% hidden test accuracy

## Team

**The Innovators** — Phase 1 classifier + Phase 2 live spelling interface.

## Deploy this Space (owner only)

Full steps: [docs/DEPLOY_FOR_JUDGES.md](docs/DEPLOY_FOR_JUDGES.md)

```bash
huggingface-cli login
python scripts/upload_weights_to_hf.py --repo-id zk-007/the-innovators
```

Then create Space from GitHub repo at [huggingface.co/new-space](https://huggingface.co/new-space).

## Run locally

```bash
pip install -r requirements.txt
python run_demo.py
```

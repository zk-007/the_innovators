# The Innovators — The Silent Gap (ASL Sign Language)

**Forman Computer Science Club · AI Hackathon 2026**

Team **The Innovators** — 29-class ASL hand-sign classifier (A–Z, `space`, `del`, `nothing`) with a **live webcam demo** and **word-spelling interface** for Phase 2.

Repository: [github.com/zk-007/the_innovators](https://github.com/zk-007/the_innovators)

---

## Phase 1 — Model training (Kaggle)

| Item | Detail |
|------|--------|
| Model | `convnext_tiny.fb_in22k_ft_in1k` (timm) |
| Training | Stratified 3-fold CV; **fold 0** used for submission |
| Image size | 224×224 |
| Fold 0 val accuracy | **100%** (see `outputs/classification_report.txt`) |
| Test predictions | `submissions/submission.csv` (17,400 rows) |

**Notebook:** [`notebooks/the-innovators.ipynb`](notebooks/the-innovators.ipynb)  
**Checkpoint:** [`models/convnext_tiny_fb_in22k_ft_in1k_fold0_best.pth`](models/convnext_tiny_fb_in22k_ft_in1k_fold0_best.pth) (~111 MB, Git LFS)

### Submit to Kaggle

Upload `submissions/submission.csv` to the competition page (max 5/day).

---

## Phase 2 — Live demo (this repo)

Matches hackathon guide requirements:

| Feature | Implementation |
|---------|----------------|
| **Live webcam demo** | Real-time camera → model (no pre-recorded video) |
| **Word spelling** | Hold a letter ~1.2s to commit; `space` / `del` signs work |
| **Two-way bridge** | Tab: hearing user types text ↔ deaf user spells via webcam |

### Run locally

```bash
cd the_innovators
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Mac/Linux
pip install -r requirements.txt
python run_demo.py
```

Open **http://127.0.0.1:7860** and allow camera access.

### Public URL (deployment bonus +5%)

```bash
python run_demo.py --share
```

Gradio prints a public `*.gradio.live` link — use this for judges on Tuesday.

---

## Project structure

```
the_innovators/
├── app/
│   ├── app.py          # Gradio UI
│   ├── model.py        # Checkpoint loader + inference
│   ├── spelling.py     # Hold-to-commit word builder
│   ├── transforms.py   # Same eval transforms as training
│   └── config.py
├── models/             # fold 0 checkpoint (Git LFS)
├── notebooks/          # Kaggle training notebook
├── submissions/        # submission.csv
├── outputs/            # metrics & reports
├── run_demo.py
├── requirements.txt
└── README.md
```

---

## Model details (for presentation)

1. **Input:** RGB hand image (webcam frame or 200×200 dataset image).
2. **Preprocessing:** Resize 224, ImageNet normalize (same as validation in notebook).
3. **Backbone:** ConvNeXt Tiny (transfer learning from ImageNet-22k → ImageNet-1k).
4. **Head:** 29-way softmax classifier.
5. **Demo smoothing:** Majority vote over last 5 frames + confidence threshold.

### What we tried

- 3-fold training config in notebook (`strong_3fold`); submitted **fold 0** checkpoint within time limit.
- Heavy augmentations (random crop, rotation, color jitter) for generalization.

### Next steps

- Ensemble folds 1–2 for higher Kaggle LB.
- MediaPipe hand crop before classification for messier backgrounds.
- Fine-tune on user-specific lighting via short calibration session.

---

## Tuesday checklist (Top 10)

- [ ] GitHub repo public with this README
- [ ] `python run_demo.py --share` for live judge URL
- [ ] 5 min presentation: model → live demo → failures → future work
- [ ] `submissions/submission.csv` submitted before Monday 12:00 PM

---

## License

Hackathon project — Forman Christian College, AI Hackathon 2026.

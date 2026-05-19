# Deploy for judges (permanent browser link)

Goal: judges only open a URL — no Python, no install.

## One-time setup (~15 minutes)

### Step 1 — Hugging Face account

1. Go to [huggingface.co/join](https://huggingface.co/join)
2. Sign up (free)

### Step 2 — Upload model weights (111 MB)

The `.pth` file is too large for normal git without LFS.

**Option A — Model repo (recommended)**

1. [huggingface.co/new-model](https://huggingface.co/new-model)
2. Name: `the-innovators` (full id: `YOUR_USERNAME/the-innovators`)
3. Upload file: `models/convnext_tiny_fb_in22k_ft_in1k_fold0_best.pth`
4. In `app/model_loader.py`, set `HF_MODEL_REPO = "YOUR_USERNAME/the-innovators"`

**Option B — Space files**

Upload the same `.pth` into the Space repo under `models/` after Step 3.

### Step 3 — Create the Space

1. [huggingface.co/new-space](https://huggingface.co/new-space)
2. **Space name:** `the-innovators`
3. **SDK:** Gradio
4. **Hardware:** CPU basic (free) or GPU if you have quota
5. **Create from:** GitHub → `zk-007/the_innovators` (branch `main`)
6. Click **Create Space**

### Step 4 — Wait for build

- Status: **Building** → **Running** (5–15 min first time)
- Open: `https://huggingface.co/spaces/YOUR_USERNAME/the-innovators`

### Step 5 — Share with judges

Put this link in your presentation slide:

```
https://huggingface.co/spaces/YOUR_USERNAME/the-innovators
```

This URL stays public while the Space is **Running** (unlike `gradio.live` which expires when your laptop closes).

---

## Keep Space awake on demo day

- Open the Space once before presenting (warm start).
- In Space **Settings** → consider **Restart on error** ON.
- Free tier may sleep after inactivity — click the link 5 min before judges arrive.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Build failed | Check **Logs** tab; often missing `mediapipe==0.10.14` |
| Model not found | Upload `.pth` to HF model repo; check `HF_MODEL_REPO` in `app/model_loader.py` |
| Webcam blocked | Use HTTPS Space URL; browser must allow camera |
| Slow | Enable GPU in Space settings or use CPU and wait ~5s per frame |

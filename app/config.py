from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
DEFAULT_CHECKPOINT = MODELS_DIR / "convnext_tiny_fb_in22k_ft_in1k_fold0_best.pth"

CLASSES = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
    "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
    "space", "del", "nothing",
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Live demo tuning
CONFIDENCE_THRESHOLD = 0.55
HOLD_SECONDS = 1.2
SMOOTHING_WINDOW = 5
SPELLABLE_LABELS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

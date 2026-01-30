# gamma_batch_autorun.py
# ------------------------------------------------------------
# Recursively applies gamma correction to all images under:
#   test_images_2025/pond1/** and test_images_2025/pond2/**
# and writes results to a mirror directory tree under test_images_2025_gamma
# ------------------------------------------------------------

from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

# ------------ Settings ------------
INPUT_ROOT  = Path("test_images_2025")
OUTPUT_ROOT = Path("test_images_2025_gamma")
GAMMA       = 2.2       # typical range 1.6–2.4
SUFFIX      = "_gamma"  # added before extension
OVERWRITE   = False     # set True to overwrite if rerun

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

# ------------ Functions ------------

def adjust_gamma(img, gamma: float):
    """Apply gamma correction using LUT."""
    inv = 1.0 / gamma
    lut = ((np.arange(256) / 255.0) ** inv) * 255
    lut = np.clip(lut, 0, 255).astype("uint8")
    return cv2.LUT(img, lut)

def is_image(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_EXTS

def process_all():
    # collect pond1 + pond2 subfolders
    candidate_dirs = []
    for name in ("pond1", "pond2"):
        d = INPUT_ROOT / name
        if d.exists():
            candidate_dirs.append(d)
    if not candidate_dirs:
        candidate_dirs = [INPUT_ROOT]

    files = []
    for d in candidate_dirs:
        files.extend([p for p in d.rglob("*") if p.is_file() and is_image(p)])

    if not files:
        print("No images found.")
        return

    print(f"Found {len(files)} images. Output root: {OUTPUT_ROOT}")

    for src in tqdm(files, desc="Gamma correcting"):
        rel = src.relative_to(INPUT_ROOT)
        dst_dir = OUTPUT_ROOT / rel.parent
        dst_dir.mkdir(parents=True, exist_ok=True)

        dst = dst_dir / (src.stem + SUFFIX + src.suffix)

        if dst.exists() and not OVERWRITE:
            continue

        img = cv2.imread(str(src))
        if img is None:
            continue
        out = adjust_gamma(img, GAMMA)
        cv2.imwrite(str(dst), out)

    print("Done.")

# ------------ Run ------------
if __name__ == "__main__":
    process_all()

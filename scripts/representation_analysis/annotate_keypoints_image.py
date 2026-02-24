#!/usr/bin/env python3
"""
Annotate a single prawn image with 4 anatomical keypoints and two measurement lines.
Saves an annotated copy under outputs/annotated_images/ without modifying the original image.

Usage: run the script (it is pre-configured to annotate the image path supplied in the task).

Behavior:
- Attempts to find the corresponding YOLO pose label in datasets/train_on_2025_all/labels (train/ and val/).
- Supports YOLO-pose label formats with either normalized coordinates (0..1) or absolute pixel coordinates.
- Detects normalization mode by inspecting keypoint values.
- Draws small colored circles for keypoints and labels, draws CL (blue) and TL (red) lines between the requested points.
- Clamps out-of-bounds keypoints for visualization but prints warnings.

"""
from pathlib import Path
import sys
import math
from PIL import Image, ImageDraw, ImageFont
import warnings

# --- Configuration (the image path provided by the user) ---
WINDOWS_IMAGE_PATH = r"C:\Users\carmonta\Desktop\data-centric-cross-season-transfer\datasets\train_on_2025_all\images\frame_00004_jpg.rf.54b21877958724da379aee0e0e4684fa.jpg"
# Output file
OUT_DIR = Path("outputs/annotated_images")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Keypoint labels and anatomical mapping (YOLO pose order assumed):
KP_NAMES = ["Carapace", "Eyes", "Rostrum", "Tail"]
# Indices for lines
CL_FROM = 1  # Eyes
CL_TO = 0    # Carapace
TL_FROM = 2  # Rostrum
TL_TO = 3    # Tail

# Drawing style
CIRCLE_RADIUS = 6
CIRCLE_FILL = {
    "Rostrum": (255, 215, 0),   # golden
    "Eyes": (0, 255, 255),      # cyan
    "Carapace": (255, 0, 255),  # magenta
    "Tail": (0, 255, 0),        # lime
}
LINE_COLORS = {
    "CL": (0, 0, 255),  # blue
    "TL": (255, 0, 0),  # red
}
TEXT_COLOR = (255, 255, 255)
TEXT_BG = (0, 0, 0, 160)

# Small helper functions

def windows_to_posix(win_path: str) -> Path:
    """Convert a Windows absolute path like C:\Users\... to a POSIX-like path under the same user root."""
    p = win_path.replace("\\", "/")
    if ":" in p:
        # strip drive letter, keep leading slash
        _, rest = p.split(":", 1)
        rest = rest.lstrip("/")
        posix = Path("/") / Path(rest)
        return posix
    return Path(p)


def find_label_file(image_path: Path) -> Path:
    """Try several plausible locations to find the matching YOLO pose label.
    Returns Path or raises FileNotFoundError.
    """
    stem = image_path.name
    # label stem candidates: same name with .txt, also name before first ".rf." or other separators
    name_variants = [stem]
    if ".rf." in stem:
        name_variants.append(stem.split(".rf.")[0])
    if "_jpg" in stem:
        name_variants.append(stem.replace("_jpg", ""))
    # remove extension for label base
    label_bases = [Path(v).stem for v in name_variants]

    candidates = []
    # typical label locations
    repo_root = Path(__file__).resolve().parents[2]
    for base in label_bases:
        candidates += [
            repo_root / "datasets" / "train_on_2025_all" / "labels" / (base + ".txt"),
            repo_root / "datasets" / "train_on_2025_all" / "val" / "labels" / (base + ".txt"),
            repo_root / "datasets" / "train_on_2025_all" / "labels" / (base + ".TXT"),
        ]
    # also try labels folder next to images (train/val separation)
    img_parent = image_path.parent
    possible_labels = [
        img_parent.parent / "labels",
        img_parent / "../labels",
    ]
    for base in label_bases:
        for labroot in possible_labels:
            candidates.append(Path(labroot).resolve() / (base + ".txt"))

    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError("Could not find label file for image: {}. Tried {}".format(image_path, candidates))


def parse_yolo_pose_label(label_path: Path) -> list:
    """Parse the YOLO pose label and return list of floats after class and bbox.
    Returns list of keypoint (x,y) pairs as floats.
    """
    text = label_path.read_text().strip()
    if not text:
        raise ValueError(f"Empty label file: {label_path}")
    parts = text.split()
    # first 5 tokens: class cx cy w h (maybe)
    floats = []
    for t in parts[5:]:
        try:
            floats.append(float(t))
        except Exception:
            pass
    if len(floats) % 2 != 0:
        # maybe the file uses only keypoints (no bbox) - then remove first token if needed
        raise ValueError(f"Unexpected number of floats in label {label_path}: {len(floats)}")
    coords = [(floats[i], floats[i + 1]) for i in range(0, len(floats), 2)]
    return coords


def load_font(size=14):
    # try to find a common sans-serif font
    try:
        # PIL can load system fonts if provided path; fallback to default
        import matplotlib.font_manager as fm
        font_path = fm.findfont("DejaVu Sans")
        return ImageFont.truetype(font_path, size=size)
    except Exception:
        return ImageFont.load_default()


def clamp_point(x, y, w, h):
    cx = min(max(x, 0), w - 1)
    cy = min(max(y, 0), h - 1)
    return cx, cy


def main():
    # Resolve image path: try windows path and fallback to repository-local absolute
    win_p = WINDOWS_IMAGE_PATH
    posix_guess = windows_to_posix(win_p)
    repo_root = Path(__file__).resolve().parents[2]
    # Prefer repo-local absolute if exists
    candidate_paths = [Path(win_p), posix_guess, repo_root / Path("datasets") / "train_on_2025_all" / "images" / Path(win_p).name]
    image_path = None
    for p in candidate_paths:
        try:
            p_res = Path(p).resolve()
        except Exception:
            p_res = Path(p)
        if p_res.exists():
            image_path = p_res
            break
    if image_path is None:
        print("ERROR: could not find the image at any of:")
        for p in candidate_paths:
            print(" -", p)
        sys.exit(1)

    print("Using image:", image_path)

    # Load image
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    print(f"Image size: width={w}, height={h}")

    # Find and parse label
    try:
        label_path = find_label_file(image_path)
        print("Found label:", label_path)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    try:
        kps = parse_yolo_pose_label(label_path)
    except Exception as e:
        print("Failed to parse label:", e)
        sys.exit(1)

    num_kp = len(kps)
    print(f"Found {num_kp} keypoints in label.")
    if num_kp < 4:
        print("Warning: fewer than 4 keypoints found; aborting.")
        sys.exit(1)

    # Only first 4 keypoints are used (carapace, eyes, rostrum, tail)
    kps = kps[:4]

    # Detect normalized vs absolute by checking max coordinate value
    max_val = max(max(abs(x), abs(y)) for x, y in kps)
    normalized = max_val <= 1.0
    if normalized:
        print("Keypoints appear NORMALIZED (0..1). Converting to pixel coords.")
    else:
        print("Keypoints appear ABSOLUTE pixel coordinates.")
    print("Max keypoint value:", max_val)

    # Convert to pixel coords
    pixel_kps = []
    out_of_bounds = False
    for i, (kx, ky) in enumerate(kps):
        if normalized:
            px = kx * w
            py = ky * h
        else:
            px = kx
            py = ky
            # If absolute but image resolution mismatches a typical working size (e.g., 640), warn
            if px > w or py > h:
                warnings.warn(f"Label coordinates ({px:.1f},{py:.1f}) exceed image size ({w},{h}). They may belong to a different resolution.")
        # Clamp for drawing but remember if clamped
        cx, cy = clamp_point(px, py, w, h)
        if (cx != px) or (cy != py):
            out_of_bounds = True
            warnings.warn(f"Keypoint {i} clamped from ({px:.1f},{py:.1f}) to ({cx:.1f},{cy:.1f}) for visualization.")
        pixel_kps.append((cx, cy))

    # Prepare drawing
    draw = ImageDraw.Draw(img)
    font = load_font(size=14)

    # Draw circles and labels
    for idx, (x, y) in enumerate(pixel_kps):
        name = KP_NAMES[idx] if idx < len(KP_NAMES) else f"kp{idx}"
        color = CIRCLE_FILL.get(name, (255, 255, 255))
        x0 = x - CIRCLE_RADIUS
        y0 = y - CIRCLE_RADIUS
        x1 = x + CIRCLE_RADIUS
        y1 = y + CIRCLE_RADIUS
        draw.ellipse([x0, y0, x1, y1], fill=color, outline=(0, 0, 0))
        # label background rectangle
        text = name
        tw, th = draw.textsize(text, font=font)
        # place label offset a bit to the top-right
        tx = x + CIRCLE_RADIUS + 4
        ty = y - th / 2
        draw.rectangle([tx - 2, ty - 1, tx + tw + 2, ty + th + 1], fill=TEXT_BG)
        draw.text((tx, ty), text, fill=TEXT_COLOR, font=font)

    # Draw CL (Eyes (1) to Carapace (0)) - blue
    try:
        ex, ey = pixel_kps[CL_FROM]
        cx, cy = pixel_kps[CL_TO]
        draw.line([(ex, ey), (cx, cy)], fill=LINE_COLORS["CL"], width=3)
        # compute mid-point for label
        mx, my = (ex + cx) / 2, (ey + cy) / 2
        label = "CL"
        tw, th = draw.textsize(label, font=font)
        draw.rectangle([mx - tw/2 - 3, my - th/2 - 2, mx + tw/2 + 3, my + th/2 + 2], fill=(0, 0, 0, 160))
        draw.text((mx - tw/2, my - th/2), label, fill=LINE_COLORS["CL"], font=font)
    except Exception as e:
        warnings.warn(f"Failed to draw CL line: {e}")

    # Draw TL (Rostrum (2) to Tail (3)) - red
    try:
        rx, ry = pixel_kps[TL_FROM]
        txp, typ = pixel_kps[TL_TO]
        draw.line([(rx, ry), (txp, typ)], fill=LINE_COLORS["TL"], width=3)
        mx, my = (rx + txp) / 2, (ry + typ) / 2
        label = "TL"
        tw, th = draw.textsize(label, font=font)
        draw.rectangle([mx - tw/2 - 3, my - th/2 - 2, mx + tw/2 + 3, my + th/2 + 2], fill=(0, 0, 0, 160))
        draw.text((mx - tw/2, my - th/2), label, fill=LINE_COLORS["TL"], font=font)
    except Exception as e:
        warnings.warn(f"Failed to draw TL line: {e}")

    # Save annotated image
    out_name = image_path.stem + "_annotated.png"
    out_path = OUT_DIR / out_name
    img.save(out_path)
    print(f"Annotated image saved to: {out_path}")


if __name__ == "__main__":
    main()


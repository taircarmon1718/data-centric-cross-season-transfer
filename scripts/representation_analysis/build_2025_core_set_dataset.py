from pathlib import Path
import pandas as pd
import shutil

# ============================================================
# PATHS
# ============================================================

BASE = Path(r"C:\Users\carmonta\Desktop\data-centric-cross-season-transfer")

SRC_DATASET = BASE / "datasets" / "train_on_2025_all"
DST_DATASET = BASE / "datasets" / "train_on_2025_core_set"

CSV_PATH = (
    BASE
    / "outputs"
    / "rep_analysis"
    / "core_set_selection"
    / "core_set.csv"
)

# ============================================================
# CREATE DEST DIRS
# ============================================================

for split in ["train", "val"]:
    (DST_DATASET / "images" / split).mkdir(parents=True, exist_ok=True)
    (DST_DATASET / "labels" / split).mkdir(parents=True, exist_ok=True)

# ============================================================
# LOAD CSV
# ============================================================

df = pd.read_csv(CSV_PATH)

# ============================================================
# COPY FILES
# ============================================================

missing_images = 0
missing_labels = 0

for _, row in df.iterrows():
    img_rel = Path(row["image_id"])
    img_name = img_rel.name

    # -------- Determine split --------
    if "val/images" in str(img_rel).replace("\\", "/"):
        split = "val"
        src_img = SRC_DATASET / "val" / "images" / img_name
        src_lbl = SRC_DATASET / "val" / "labels" / img_name.replace(".jpg", ".txt")
    else:
        split = "train"
        src_img = SRC_DATASET / "images" / img_name
        src_lbl = SRC_DATASET / "labels" / img_name.replace(".jpg", ".txt")

    dst_img = DST_DATASET / "images" / split / img_name
    dst_lbl = DST_DATASET / "labels" / split / src_lbl.name

    if not src_img.exists():
        print(f"[WARN] Missing image: {src_img}")
        missing_images += 1
        continue

    if not src_lbl.exists():
        print(f"[WARN] Missing label: {src_lbl}")
        missing_labels += 1
        continue

    shutil.copy2(src_img, dst_img)
    shutil.copy2(src_lbl, dst_lbl)

# ============================================================
# CREATE data.yaml
# ============================================================

yaml_src = SRC_DATASET / "data.yaml"
yaml_dst = DST_DATASET / "data.yaml"

with open(yaml_src, "r", encoding="utf-8") as f:
    yaml_text = f.read()

yaml_text = yaml_text.replace(
    str(SRC_DATASET).replace("\\", "/"),
    str(DST_DATASET).replace("\\", "/")
)

with open(yaml_dst, "w", encoding="utf-8") as f:
    f.write(yaml_text)

# ============================================================
# SUMMARY
# ============================================================

print("✅ Core-set dataset created successfully")
print(f"📁 Source dataset: {SRC_DATASET}")
print(f"📁 Target dataset: {DST_DATASET}")
print(f"🖼️  Images copied: {len(df) - missing_images}")
print(f"🏷️  Labels missing: {missing_labels}")

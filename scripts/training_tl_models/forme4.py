from pathlib import Path
import shutil

ROOT = Path(r"C:\Users\carmonta\Desktop\data-centric-cross-season-transfer\models\TL_Optim_core_set\k_yolo_embeddings")
CLEAN_ROOT = ROOT / "_STAGE2_CLEAN"

def main():
    CLEAN_ROOT.mkdir(exist_ok=True)

    for k_dir in sorted(ROOT.glob("k*")):
        if not k_dir.is_dir():
            continue

        k_name = k_dir.name  # e.g. k01, k50

        # find only stage2 folders
        stage2_runs = list(k_dir.glob("*stage2"))

        if not stage2_runs:
            continue

        # take latest stage2 if multiple
        latest = sorted(stage2_runs, key=lambda p: p.stat().st_mtime)[-1]

        best_path = latest / "weights" / "best.pt"

        if not best_path.exists():
            print(f"[WARN] No best.pt in {latest}")
            continue

        out_dir = CLEAN_ROOT / k_name
        out_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy2(best_path, out_dir / "best.pt")

        print(f"[OK] {k_name} -> copied from {latest.name}")

    print("\nDone.")
    print(f"Clean folder created at:\n{CLEAN_ROOT}")

if __name__ == "__main__":
    main()

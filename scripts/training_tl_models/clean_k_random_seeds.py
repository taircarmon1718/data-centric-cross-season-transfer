"""
Clean TL-Optim k_random seed outputs into a compact structure.

Rules applied:
- Source base: models/TF/TL_Optim_core_set/k_random_seeds/
- Target base: models/TF/TL_Optim_core_set/k_random_cleaned/
- Seed folder name mapping:
    k_random         -> seed0
    k_random_seed1   -> seed1
    k_random_seed2   -> seed2
- Inside each seed/kXX/, keep only the run folder whose name ends with '2'.
- From that run copy only weights/best.pt to target: seedX/kYY/best.pt
- Fix k naming when needed: extracts kNN from runname (e.g., k012 -> k01)

Run this script from project root or via the project Python environment.
"""
from pathlib import Path
import shutil
import re
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_BASE = PROJECT_ROOT / 'models' / 'TF' / 'TL_Optim_core_set' / 'k_random_seeds'
TARGET_BASE = PROJECT_ROOT / 'models' / 'TF' / 'TL_Optim_core_set' / 'k_random_cleaned'

SEED_MAP = {
    'k_random': 'seed0',
    'k_random_seed1': 'seed1',
    'k_random_seed2': 'seed2',
}

RUN_SUFFIX = '2'


def extract_k_from_runname(runname: str) -> str:
    """Extract digits after 'k' and before trailing '2', return normalized k folder like 'k01'.

    Examples:
      TL_Optim_2024to2025_random_k012 -> k01
      TL_Optim_2024to2025_random_k502 -> k50
    """
    m = re.search(r'k(\d+)'+re.escape(RUN_SUFFIX)+r'$' , runname)
    if not m:
        # fallback: try find 'k' followed by digits anywhere
        m2 = re.search(r'k(\d+)', runname)
        if m2:
            num = int(m2.group(1))
            return f'k{num:02d}'
        return None
    digits = m.group(1)
    try:
        num = int(digits)
        return f'k{num:02d}'
    except Exception:
        return None


def process_seed_folder(seed_src: Path, seed_target_name: str, report: dict):
    """Process one seed folder: iterate k folders, pick run ending with '2', copy best.pt"""
    if not seed_src.exists() or not seed_src.is_dir():
        print(f"[WARN] seed source not found: {seed_src}")
        return
    for k_dir in sorted(seed_src.iterdir()):
        if not k_dir.is_dir():
            continue
        # inside k_dir there are run folders; we want the one whose name ends with '2'
        runs = [p for p in k_dir.iterdir() if p.is_dir()]
        chosen = [r for r in runs if r.name.endswith(RUN_SUFFIX)]
        if not chosen:
            print(f"[WARN] No run ending with '{RUN_SUFFIX}' in {k_dir}; skipping")
            report.setdefault('missing_run', []).append(str(k_dir))
            continue
        # choose the last one (if multiple) deterministically sorted
        chosen = sorted(chosen)
        run = chosen[-1]
        k_normal = extract_k_from_runname(run.name)
        if k_normal is None:
            print(f"[WARN] could not parse k from run name {run.name}; using k dir name {k_dir.name}")
            k_normal = k_dir.name
        target_k_dir = TARGET_BASE / seed_target_name / k_normal
        target_k_dir.mkdir(parents=True, exist_ok=True)
        src_best = run / 'weights' / 'best.pt'
        if not src_best.exists():
            print(f"[WARN] best.pt not found at {src_best}; skipping")
            report.setdefault('missing_best', []).append(str(run))
            continue
        dst_best = target_k_dir / 'best.pt'
        shutil.copy2(src_best, dst_best)
        print(f"COPIED: {src_best} -> {dst_best}")
        report.setdefault('copied', []).append(str(dst_best))


def main():
    print("Starting cleanup of k_random_seeds -> k_random_cleaned")
    print(f"SRC_BASE: {SRC_BASE}")
    print(f"TARGET_BASE: {TARGET_BASE}")
    if not SRC_BASE.exists():
        print(f"Source base does not exist: {SRC_BASE}")
        return 1
    # Clean target base if exists? We'll create fresh structure but do not delete existing to be safe
    TARGET_BASE.mkdir(parents=True, exist_ok=True)
    report = {}
    # iterate known seed mappings; also handle any unexpected seed folders by mapping names
    for src_name, tgt_name in SEED_MAP.items():
        seed_src = SRC_BASE / src_name
        if not seed_src.exists():
            print(f"[INFO] Expected seed folder not present: {seed_src} (skipping)")
            continue
        print(f"Processing seed folder: {seed_src} -> {tgt_name}")
        process_seed_folder(seed_src, tgt_name, report)
    # Also handle any other seed dirs present in source that were not in SEED_MAP
    for extra in sorted(SRC_BASE.iterdir()):
        if not extra.is_dir():
            continue
        if extra.name in SEED_MAP:
            continue
        # map extra name to a safe name (use original name)
        print(f"[INFO] Found extra seed folder: {extra.name}; mapping to same name")
        process_seed_folder(extra, extra.name, report)

    # Summary
    n_copied = len(report.get('copied', []))
    n_missing_best = len(report.get('missing_best', []))
    n_missing_run = len(report.get('missing_run', []))
    print("\nCleanup summary:")
    print(f"  copied best.pt files: {n_copied}")
    print(f"  runs missing best.pt: {n_missing_best}")
    print(f"  k dirs with no trailing-2 run: {n_missing_run}")
    if report.get('missing_run'):
        print("Missing runs:\n", "\n".join(report['missing_run']))
    if report.get('missing_best'):
        print("Runs missing best.pt:\n", "\n".join(report['missing_best']))
    return 0


if __name__ == '__main__':
    sys.exit(main())


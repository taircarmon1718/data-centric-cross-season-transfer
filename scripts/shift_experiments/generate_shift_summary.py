#!/usr/bin/env python3
"""
generate_shift_summary.py

Loads embedding_shift_metrics.json and summarizes shift magnitudes.
Saves a markdown summary to results/shift_summary.md
"""
from pathlib import Path
import json

PROJECT_ROOT = Path(r"C:\Users\carmonta\Desktop\data-centric-cross-season-transfer")
IN_JSON = PROJECT_ROOT / 'scripts' / 'shift_experiments' / 'results' / 'embedding_shift_metrics.json'
OUT_MD = PROJECT_ROOT / 'scripts' / 'shift_experiments' / 'results' / 'shift_summary.md'


def main():
    if not IN_JSON.exists():
        print('Metrics JSON not found:', IN_JSON)
        return
    with open(IN_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)
    cdist = data.get('centroid_distances', {})
    mmd = data.get('mmd', {})
    d24_clean = cdist.get('2024_vs_2025_clean')
    d24_shift = cdist.get('2024_vs_2025_shifted')
    ratio = None
    if d24_clean is not None and d24_shift is not None and d24_clean > 0:
        ratio = d24_shift / d24_clean
    mmd_clean = mmd.get('2024_vs_2025_clean')
    mmd_shift = mmd.get('2024_vs_2025_shifted')
    mmd_increase = None
    if mmd_clean is not None and mmd_shift is not None and mmd_clean > 0:
        mmd_increase = (mmd_shift - mmd_clean) / mmd_clean
    lines = []
    lines.append('# Shift summary')
    lines.append('')
    if d24_shift is None or d24_clean is None:
        lines.append('Insufficient centroid distance data to compare shifted vs clean.')
    else:
        lines.append(f'- Centroid distance 2024 vs clean: {d24_clean:.6f}')
        lines.append(f'- Centroid distance 2024 vs shifted: {d24_shift:.6f}')
        lines.append(f'- Ratio (shifted / clean): {ratio:.3f}' if ratio is not None else '')
    if mmd_clean is None or mmd_shift is None:
        lines.append('- Insufficient MMD data.')
    else:
        lines.append(f'- MMD 2024 vs clean: {mmd_clean:.6e}')
        lines.append(f'- MMD 2024 vs shifted: {mmd_shift:.6e}')
        lines.append(f'- Relative increase in MMD: {mmd_increase:.3f}' if mmd_increase is not None else '')
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_MD, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('Wrote shift summary to', OUT_MD)


if __name__ == '__main__':
    main()


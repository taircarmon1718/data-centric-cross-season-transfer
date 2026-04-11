#!/usr/bin/env python3
"""
run_all_k_evaluations.py

Run one existing evaluation script across all final K-sensitivity models:
  k_sensitivity_experiment/K_*/model/best.pt

This script:
- Does NOT modify any existing code.
- Calls the existing evaluator via subprocess.
- Writes all outputs into a NEW root:
    eval_k_sensitivity_results/
- Handles missing models and partial runs safely.
- Builds:
    eval_k_sensitivity_results/global_summary.csv

Usage example:
  python run_all_k_evaluations.py \
    --eval-script /Users/taircarmon/Desktop/data-centric-cross-season-transfer/scripts/eval/check_on_2025.py \
    --model-arg --model
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent
K_ROOT = PROJECT_ROOT / "adaptive_k_experiment"
OUT_ROOT = PROJECT_ROOT / "eval_k_sensitivity_results"
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "active_season_adaptive_uncertainty_pipeline" / "run_k_sensitivity_experiment.py"


def extract_k(folder_name: str) -> Optional[int]:
    m = re.fullmatch(r"K_(\d+)", folder_name)
    if not m:
        return None
    return int(m.group(1))


def _resolve_model_path(k_dir: Path, direction: str = "", k: Optional[int] = None) -> Optional[Path]:
    """
    Resolve final model checkpoint for one K directory.
    Priority:
      1) K_*/model/best.pt
      2) K_*/model/last.pt
      3) newest K_*/model/**/weights/best.pt
      4) newest K_*/model/**/weights/last.pt
      5) newest K_*/model/**/best.pt
      6) newest K_*/model/**/last.pt
    """
    direct = k_dir / "model" / "best.pt"
    if direct.exists():
        return direct
    direct_last = k_dir / "model" / "last.pt"
    if direct_last.exists():
        return direct_last

    model_root = k_dir / "model"
    if not model_root.exists():
        return None

    weighted = sorted(model_root.rglob("weights/best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if weighted:
        return weighted[0]
    weighted_last = sorted(model_root.rglob("weights/last.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if weighted_last:
        return weighted_last[0]

    any_best = sorted(model_root.rglob("best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if any_best:
        return any_best[0]
    any_last = sorted(model_root.rglob("last.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if any_last:
        return any_last[0]

    # Fallback: read model path from K_*/results.json
    res_json = k_dir / "results.json"
    if res_json.exists():
        try:
            payload = json.loads(res_json.read_text(encoding="utf-8"))
            best_model = payload.get("best_model")
            if isinstance(best_model, str) and best_model.strip():
                raw = Path(best_model)
                if raw.exists():
                    return raw

                # Cross-machine remap: if path contains repository name, rebuild under current PROJECT_ROOT
                marker = "data-centric-cross-season-transfer"
                low = best_model.replace("\\", "/")
                if marker in low:
                    rel = low.split(marker, 1)[1].lstrip("/")
                    remapped = PROJECT_ROOT / rel
                    if remapped.exists():
                        return remapped
        except Exception:
            pass
    # Last-resort project-wide search for best.pt files.
    # IMPORTANT: accept only candidates that match BOTH direction and K token,
    # otherwise we can accidentally reuse the same checkpoint for all K values.
    try:
        all_pts = sorted(PROJECT_ROOT.rglob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    except Exception:
        all_pts = []
    if not all_pts:
        return None

    k_token = f"K_{k}" if k is not None else ""
    direction_tokens = [t for t in str(direction).split("/") if t and t != "root"]

    def score(p: Path) -> int:
        s = str(p).replace("\\", "/").lower()
        sc = 0
        if k_token and k_token.lower() in s:
            sc += 3
        for tok in direction_tokens:
            if tok.lower() in s:
                sc += 2
        if "/adaptive_k_experiment/" in s:
            sc += 2
        if "/model/" in s:
            sc += 1
        return sc

    ranked = sorted(all_pts, key=lambda p: (score(p), p.stat().st_mtime), reverse=True)
    for cand in ranked:
        s = str(cand).replace("\\", "/").lower()
        has_k = (k_token.lower() in s) if k_token else True
        has_dir = all(tok.lower() in s for tok in direction_tokens) if direction_tokens else True
        if has_k and has_dir:
            return cand
    return None


def discover_models(k_root: Path) -> List[Tuple[str, int, Path, Optional[Path]]]:
    """
    Returns list of (direction, K, k_dir, best_model_path).
    Supports both layouts:
      - adaptive_k_experiment/<direction>/K_*/model/best.pt
      - adaptive_k_experiment/K_*/model/best.pt
    """
    rows = []
    if not k_root.exists():
        return rows
    for p in sorted(k_root.rglob("K_*"), key=lambda x: str(x)):
        if not p.is_dir():
            continue
        k = extract_k(p.name)
        if k is None:
            continue
        rel_parent = p.parent.relative_to(k_root)
        direction = str(rel_parent) if str(rel_parent) != "." else "root"
        best = _resolve_model_path(p, direction=direction, k=k)
        rows.append((direction, k, p, best))
    rows.sort(key=lambda x: (x[0], x[1]))
    return rows


def ensure_unique_dir(path: Path) -> Path:
    """If path exists, create sibling with suffix _runN."""
    if not path.exists():
        path.mkdir(parents=True, exist_ok=False)
        return path
    n = 2
    while True:
        cand = Path(f"{str(path)}_run{n}")
        if not cand.exists():
            cand.mkdir(parents=True, exist_ok=False)
            return cand
        n += 1


def ensure_fixed_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def is_k_eval_complete(k_out_dir: Path) -> bool:
    return (
        (k_out_dir / "summary.csv").exists()
        and (k_out_dir / "raw_outputs").exists()
        and (k_out_dir / "visualizations").exists()
    )


def find_first_csv(directory: Path) -> Optional[Path]:
    if not directory.exists():
        return None
    csvs = sorted(directory.rglob("*.csv"))
    return csvs[0] if csvs else None


def find_metric(summary_row: Dict[str, str], patterns: List[str]) -> Optional[float]:
    """
    Find first matching metric by fuzzy column-name patterns.
    """
    norm = {k.strip().lower(): v for k, v in summary_row.items()}
    for key, val in norm.items():
        if all(p in key for p in patterns):
            try:
                return float(val)
            except Exception:
                return None
    return None


def parse_summary_csv(summary_csv: Path) -> Dict[str, Optional[float]]:
    out = {
        "num_images": None,
        "det_rate_carapace": None,
        "det_rate_total": None,
        "mae_carapace": None,
        "mae_total": None,
        "mpe_carapace": None,
        "mpe_total": None,
    }
    if not summary_csv.exists():
        return out

    try:
        with open(summary_csv, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if not rows:
                return out
            # Prefer last row if evaluator appends aggregate row.
            row = rows[-1]
    except Exception:
        return out

    # number of samples
    for cand in ["num_images", "n_images", "num_samples", "n_samples", "count", "images"]:
        if cand in row:
            try:
                out["num_images"] = float(row[cand])
                break
            except Exception:
                pass

    # flexible metric extraction
    out["det_rate_carapace"] = find_metric(row, ["detection", "carapace"])
    out["det_rate_total"] = find_metric(row, ["detection", "total"])
    out["mae_carapace"] = find_metric(row, ["mae", "carapace"])
    out["mae_total"] = find_metric(row, ["mae", "total"])
    out["mpe_carapace"] = find_metric(row, ["mpe", "carapace"])
    out["mpe_total"] = find_metric(row, ["mpe", "total"])
    return out


def parse_num_samples_from_detail_csv(detail_csv: Path) -> Optional[float]:
    if not detail_csv.exists():
        return None
    try:
        with open(detail_csv, "r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return None
        # detail file from check_on_2025 has one row per GT record
        # and includes mode/status columns.
        return float(len(rows))
    except Exception:
        return None


def write_per_k_summary(k_out_dir: Path, k: int, metrics: Dict[str, Optional[float]]) -> Path:
    path = k_out_dir / "summary.csv"
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "K",
                "number_of_samples",
                "detection_rate_carapace",
                "detection_rate_total",
                "MAE_carapace",
                "MAE_total",
                "MPE_carapace",
                "MPE_total",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "K": k,
                "number_of_samples": metrics["num_images"],
                "detection_rate_carapace": metrics["det_rate_carapace"],
                "detection_rate_total": metrics["det_rate_total"],
                "MAE_carapace": metrics["mae_carapace"],
                "MAE_total": metrics["mae_total"],
                "MPE_carapace": metrics["mpe_carapace"],
                "MPE_total": metrics["mpe_total"],
            }
        )
    return path


def run_eval_subprocess(
    python_exe: str,
    eval_script: Path,
    model_path: Path,
    out_dir: Path,
    model_arg: str,
    passthrough: List[str],
) -> int:
    """
    Run evaluator in isolated working directory `out_dir`.
    We always pass model path; additional args are user-provided passthrough.
    """
    cmd = [python_exe, str(eval_script), model_arg, str(model_path)] + passthrough
    print("  Command:", " ".join(cmd))
    proc = subprocess.run(
        cmd,
        cwd=str(out_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    (out_dir / "run_stdout.log").write_text(proc.stdout or "", encoding="utf-8")
    (out_dir / "run_stderr.log").write_text(proc.stderr or "", encoding="utf-8")
    return proc.returncode


def run_eval_via_injected_module(
    python_exe: str,
    eval_script: Path,
    model_path: Path,
    out_dir: Path,
) -> int:
    """
    Fallback path for evaluators that do not expose a model CLI argument.
    Loads evaluator module in a subprocess, injects one model + output root, then runs main().
    """
    code = (
        "import importlib.util, pathlib, sys; "
        f"script = pathlib.Path(r'''{str(eval_script)}'''); "
        "spec = importlib.util.spec_from_file_location('eval_mod', str(script)); "
        "mod = importlib.util.module_from_spec(spec); "
        "spec.loader.exec_module(mod); "
        f"mod.OUT_ROOT = pathlib.Path(r'''{str(out_dir)}'''); "
        f"model_p = pathlib.Path(r'''{str(model_path)}'''); "
        "mod.discover_models = (lambda roots: {'best': model_p}); "
        "mod.main()"
    )
    cmd = [python_exe, "-c", code]
    print("  Fallback command: python -c <injected evaluator>")
    proc = subprocess.run(
        cmd,
        cwd=str(out_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    (out_dir / "run_stdout_fallback.log").write_text(proc.stdout or "", encoding="utf-8")
    (out_dir / "run_stderr_fallback.log").write_text(proc.stderr or "", encoding="utf-8")
    return proc.returncode


def build_global_summary(global_rows: List[Dict], out_root: Path) -> Path:
    out_path = out_root / "global_summary.csv"
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "direction",
                "K",
                "number_of_samples",
                "detection_rate_carapace",
                "detection_rate_total",
                "MAE_carapace",
                "MAE_total",
                "MPE_carapace",
                "MPE_total",
                "status",
            ],
        )
        writer.writeheader()
        for row in sorted(global_rows, key=lambda x: x["K"]):
            writer.writerow(row)
    return out_path


def write_resolved_models(discovered: List[Tuple[str, int, Path, Optional[Path]]], out_root: Path) -> Path:
    out_path = out_root / "resolved_models.csv"
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["direction", "K", "k_dir", "resolved_model_path", "exists"],
        )
        writer.writeheader()
        for direction, k, k_dir, mp in discovered:
            writer.writerow(
                {
                    "direction": direction,
                    "K": k,
                    "k_dir": str(k_dir),
                    "resolved_model_path": "" if mp is None else str(mp),
                    "exists": bool(mp is not None and mp.exists()),
                }
            )
    return out_path


def load_existing_per_k_summary(k_out_dir: Path) -> Optional[Dict]:
    p = k_out_dir / "summary.csv"
    if not p.exists():
        return None
    try:
        with open(p, "r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
            if not rows:
                return None
            r = rows[-1]
            return {
                "direction": r.get("direction", ""),
                "K": int(float(r.get("K", ""))),
                "number_of_samples": r.get("number_of_samples"),
                "detection_rate_carapace": r.get("detection_rate_carapace"),
                "detection_rate_total": r.get("detection_rate_total"),
                "MAE_carapace": r.get("MAE_carapace"),
                "MAE_total": r.get("MAE_total"),
                "MPE_carapace": r.get("MPE_carapace"),
                "MPE_total": r.get("MPE_total"),
                "status": "completed_cached",
            }
    except Exception:
        return None


def run_training_pipeline(python_exe: str, out_root: Path) -> int:
    """
    Try to generate missing adaptive_k_experiment checkpoints by running the
    K-sensitivity training pipeline once.
    """
    if not TRAIN_SCRIPT.exists():
        print(f"Training script not found, cannot auto-generate models: {TRAIN_SCRIPT}")
        return 1
    log_path = out_root / "auto_train_stdout.log"
    err_path = out_root / "auto_train_stderr.log"
    print(f"Attempting to auto-generate missing checkpoints via: {TRAIN_SCRIPT}")
    with open(log_path, "w", encoding="utf-8") as out_f, open(err_path, "w", encoding="utf-8") as err_f:
        proc = subprocess.run(
            [python_exe, str(TRAIN_SCRIPT)],
            cwd=str(PROJECT_ROOT),
            stdout=out_f,
            stderr=err_f,
            text=True,
        )
    print(f"Auto-train logs: {log_path} / {err_path}")
    return proc.returncode


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run evaluator for all K-sensitivity final models.")
    p.add_argument(
        "--eval-script",
        type=str,
        required=False,
        default=None,
        help="Path to existing evaluation script (e.g. /path/to/scripts/eval/check_on_2025.py). If omitted the script will try to auto-discover a suitable evaluator under scripts/eval/.",
    )
    p.add_argument(
        "--model-arg",
        type=str,
        default="--model",
        help="CLI argument name used by evaluator for model path (default: --model).",
    )
    p.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable to run evaluator.",
    )
    p.add_argument(
        "--out-root",
        type=str,
        default=str(OUT_ROOT),
        help="Output root for evaluation results (default: stable in-place folder, no _runN suffix).",
    )
    p.add_argument(
        "--unique-output",
        action="store_true",
        help="Create unique output folder with _runN suffix instead of reusing --out-root.",
    )
    p.add_argument(
        "--skip-completed",
        action="store_true",
        help="Skip K folders that already have completed summary/raw_outputs/visualizations.",
    )
    p.add_argument(
        "--no-auto-train",
        action="store_true",
        help="Disable automatic attempt to generate missing checkpoints.",
    )
    p.add_argument(
        "--single-model",
        type=str,
        default=None,
        help="Optional: path to a single model file (.pt). If provided, script will run evaluator once on this model and write summary into out-root/single_model.",
    )
    p.add_argument(
        "eval_args",
        nargs=argparse.REMAINDER,
        help="Extra args passed through to the evaluation script.",
    )
    return p.parse_args()


def auto_discover_eval_script(project_root: Path) -> Optional[Path]:
    """Try to find a sensible evaluation script under scripts/eval/.

    Priority:
      1) scripts/eval/check_on_2025.py
      2) scripts/eval/check_on_2024.py
      3) any script under scripts/eval containing 'check_on' in its name
      4) any script under scripts/eval containing 'eval' or 'batch' in its name

    Returns resolved Path or None if not found.
    """
    eval_dir = project_root / 'scripts' / 'eval'
    if not eval_dir.exists():
        return None
    preferred = ['check_on_2025.py', 'check_on_2024.py', 'check_on.py']
    for name in preferred:
        p = eval_dir / name
        if p.exists():
            return p.resolve()
    # fallback search
    py_files = sorted([p for p in eval_dir.iterdir() if p.suffix == '.py'])
    for p in py_files:
        n = p.name.lower()
        if 'check_on' in n:
            return p.resolve()
    for p in py_files:
        n = p.name.lower()
        if 'eval' in n or 'batch' in n:
            return p.resolve()
    return None


def main() -> None:
    args = parse_args()

    # handle eval_script: use provided path or try auto-discovery
    if args.eval_script:
        eval_script = Path(args.eval_script).resolve()
        if not eval_script.exists():
            raise FileNotFoundError(f"Evaluation script not found: {eval_script}")
    else:
        eval_script = auto_discover_eval_script(PROJECT_ROOT)
        if eval_script is None:
            raise FileNotFoundError(
                "Evaluation script not provided and auto-discovery failed. Please provide --eval-script pointing to your evaluator."
            )
        print(f"Auto-discovered evaluation script: {eval_script}")

    model_arg = args.model_arg.strip()
    if not model_arg.startswith("-"):
        raise ValueError("--model-arg must look like --model or -m")

    out_root_base = Path(args.out_root).resolve()
    out_root = ensure_unique_dir(out_root_base) if args.unique_output else ensure_fixed_dir(out_root_base)
    print(f"Output root: {out_root}")
    print(f"Evaluation script: {eval_script}")
    print(f"Model arg: {model_arg}")

    passthrough = list(args.eval_args or [])
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]

    # Single model mode takes priority: ignore K_ROOT, run once on this model.
    if args.single_model:
        single_model_path = Path(args.single_model).resolve()
        if not single_model_path.exists():
            raise FileNotFoundError(f"Single model file not found: {single_model_path}")
        print(f"Running single model evaluation on: {single_model_path}")
        single_model_out_root = ensure_unique_dir(out_root_base / "single_model")
        single_model_out_root.mkdir(parents=True, exist_ok=True)

        rc = run_eval_subprocess(
            python_exe=args.python,
            eval_script=eval_script,
            model_path=single_model_path,
            out_dir=single_model_out_root,
            model_arg=model_arg,
            passthrough=passthrough,
        )
        if rc != 0:
            stderr_txt = (single_model_out_root / "run_stderr.log").read_text(encoding="utf-8", errors="ignore")
            arg_error = ("unrecognized arguments" in stderr_txt.lower()) or ("error: " in stderr_txt.lower())
            if arg_error:
                print("  CLI argument mismatch detected, retrying with injected-module fallback...")
                rc = run_eval_via_injected_module(
                    python_exe=args.python,
                    eval_script=eval_script,
                    model_path=single_model_path,
                    out_dir=single_model_out_root,
                )

        # Collect and parse outputs for the single model run.
        generated_csv = sorted(single_model_out_root.rglob("*.csv"))
        generated_imgs = sorted(
            [p for p in single_model_out_root.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".pdf", ".svg"}]
        )

        # Copy discovered artifacts into stable single-model dir without overwrite.
        for src in generated_csv:
            dst = single_model_out_root / src.name
            if dst.exists():
                dst = single_model_out_root / f"{src.stem}_{int(time.time())}{src.suffix}"
            dst.write_bytes(src.read_bytes())
        for src in generated_imgs:
            dst = single_model_out_root / src.name
            if dst.exists():
                dst = single_model_out_root / f"{src.stem}_{int(time.time())}{src.suffix}"
            dst.write_bytes(src.read_bytes())

        summary_candidate = find_first_csv(single_model_out_root)
        metrics = parse_summary_csv(summary_candidate) if summary_candidate else {
            "num_images": None,
            "det_rate_carapace": None,
            "det_rate_total": None,
            "mae_carapace": None,
            "mae_total": None,
            "mpe_carapace": None,
            "mpe_total": None,
        }
        write_per_k_summary(single_model_out_root, 0, metrics)

        status = "ok" if rc == 0 else f"eval_failed_rc_{rc}"
        global_row = {
            "direction": "single_model",
            "K": 0,
            "number_of_samples": metrics["num_images"],
            "detection_rate_carapace": metrics["det_rate_carapace"],
            "detection_rate_total": metrics["det_rate_total"],
            "MAE_carapace": metrics["mae_carapace"],
            "MAE_total": metrics["mae_total"],
            "MPE_carapace": metrics["mpe_carapace"],
            "MPE_total": metrics["mpe_total"],
            "status": status,
        }
        global_csv = build_global_summary([global_row], out_root)
        print(f"\nDone. Global summary: {global_csv}")
        return

    discovered = discover_models(K_ROOT)
    if not discovered:
        print(f"No K_* folders found under: {K_ROOT}")
        return

    resolved_csv = write_resolved_models(discovered, out_root)
    print(f"Resolved checkpoint table: {resolved_csv}")
    for direction, k, _kdir, mp in discovered:
        k_name = f"K_{k}"
        if mp is not None and mp.exists():
            print(f"Resolved {direction}/{k_name} -> {mp}")
        else:
            print(f"Resolved {direction}/{k_name} -> MISSING")

    # If no checkpoints are resolvable, optionally try to generate them.
    resolved_count = sum(1 for _direction, _k, _kdir, mp in discovered if mp is not None and mp.exists())
    if resolved_count == 0 and not args.no_auto_train:
        rc_train = run_training_pipeline(args.python, out_root)
        if rc_train == 0:
            print("Auto-train completed, re-discovering checkpoints...")
            discovered = discover_models(K_ROOT)
            resolved_csv = write_resolved_models(discovered, out_root)
            print(f"Resolved checkpoint table (post-train): {resolved_csv}")
            for direction, k, _kdir, mp in discovered:
                k_name = f"K_{k}"
                if mp is not None and mp.exists():
                    print(f"Resolved {direction}/{k_name} -> {mp}")
                else:
                    print(f"Resolved {direction}/{k_name} -> MISSING")
            resolved_count = sum(1 for _direction, _k, _kdir, mp in discovered if mp is not None and mp.exists())
        else:
            print(f"Auto-train failed with code {rc_train}. Continuing with missing_model summary.")

    # If still no checkpoints are resolvable, emit informative summary.
    if resolved_count == 0:
        msg = (
            "No resolvable model checkpoints were found for adaptive_k_experiment.\n"
            "Expected one of:\n"
            "  - adaptive_k_experiment/<direction>/K_*/model/best.pt\n"
            "  - adaptive_k_experiment/<direction>/K_*/model/**/weights/best.pt\n"
            "  - valid local path in K_*/results.json -> best_model\n"
            "Current results.json files reference non-local Windows paths.\n"
            "Please copy/train .pt checkpoints on this machine, then rerun."
        )
        print("WARNING:", msg)
        global_rows = []
        for direction, k, _kdir, _mp in discovered:
            global_rows.append(
                {
                    "direction": direction,
                    "K": k,
                    "number_of_samples": "",
                    "detection_rate_carapace": "",
                    "detection_rate_total": "",
                    "MAE_carapace": "",
                    "MAE_total": "",
                    "MPE_carapace": "",
                    "MPE_total": "",
                    "status": "missing_model",
                }
            )
        global_csv = build_global_summary(global_rows, out_root)
        print(f"\nDone. Global summary: {global_csv}")
        return

    # Guardrail: detect accidental checkpoint reuse across multiple K in same direction.
    by_direction = {}
    for direction, k, _k_dir, model_path in discovered:
        if model_path is None:
            continue
        by_direction.setdefault(direction, {}).setdefault(str(model_path), []).append(k)
    for direction, model_map in by_direction.items():
        for mp, ks in model_map.items():
            if len(ks) > 1:
                print(
                    f"WARNING: same checkpoint resolved for multiple K in {direction}: "
                    f"{mp} -> K={sorted(ks)}"
                )

    global_rows: List[Dict] = []

    for direction, k, _k_dir, model_path in discovered:
        k_name = f"K_{k}"
        k_out_dir = out_root / direction / k_name
        raw_out = k_out_dir / "raw_outputs"
        viz_out = k_out_dir / "visualizations"

        if model_path is None or not model_path.exists():
            missing_hint = (_k_dir / "model" / "best.pt")
            print(f"Skipping {direction}/{k_name} (no model): {missing_hint}")
            global_rows.append(
                {
                    "direction": direction,
                    "K": k,
                    "number_of_samples": "",
                    "detection_rate_carapace": "",
                    "detection_rate_total": "",
                    "MAE_carapace": "",
                    "MAE_total": "",
                    "MPE_carapace": "",
                    "MPE_total": "",
                    "status": "missing_model",
                }
            )
            continue

        print(f"Running {direction}/{k_name}...")
        k_out_dir.mkdir(parents=True, exist_ok=True)
        raw_out.mkdir(parents=True, exist_ok=True)
        viz_out.mkdir(parents=True, exist_ok=True)

        if args.skip_completed and is_k_eval_complete(k_out_dir):
            cached = load_existing_per_k_summary(k_out_dir)
            if cached is not None:
                print(f"  Skipping {direction}/{k_name} (already complete)")
                global_rows.append(cached)
                continue

        run_dir = ensure_unique_dir(raw_out / "run")
        # check_on_2025/check_on_2024 do not expose model CLI and discover models internally.
        # Force injected single-model mode to evaluate exactly this K model.
        if eval_script.name in {"check_on_2025.py", "check_on_2024.py"}:
            rc = run_eval_via_injected_module(
                python_exe=args.python,
                eval_script=eval_script,
                model_path=model_path.resolve(),
                out_dir=run_dir,
            )
        else:
            rc = run_eval_subprocess(
                python_exe=args.python,
                eval_script=eval_script,
                model_path=model_path.resolve(),
                out_dir=run_dir,
                model_arg=model_arg,
                passthrough=passthrough,
            )
            if rc != 0:
                stderr_txt = (run_dir / "run_stderr.log").read_text(encoding="utf-8", errors="ignore")

                arg_error = ("unrecognized arguments" in stderr_txt.lower()) or ("error: " in stderr_txt.lower())
                if arg_error:
                    print("  CLI argument mismatch detected, retrying with injected-module fallback...")
                    rc = run_eval_via_injected_module(
                        python_exe=args.python,
                        eval_script=eval_script,
                        model_path=model_path.resolve(),
                        out_dir=run_dir,
                    )

        # Try to collect outputs generated by evaluator.
        # We do not assume fixed filenames; we gather discovered CSVs and images.
        generated_csv = sorted(run_dir.rglob("*.csv"))
        generated_imgs = sorted(
            [p for p in run_dir.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".pdf", ".svg"}]
        )

        # Copy discovered artifacts into stable per-K dirs without overwrite.
        for src in generated_csv:
            dst = raw_out / src.name
            if dst.exists():
                dst = raw_out / f"{src.stem}_{int(time.time())}{src.suffix}"
            dst.write_bytes(src.read_bytes())
        for src in generated_imgs:
            dst = viz_out / src.name
            if dst.exists():
                dst = viz_out / f"{src.stem}_{int(time.time())}{src.suffix}"
            dst.write_bytes(src.read_bytes())

        # Find a summary-like CSV to parse metrics.
        summary_candidate = None
        detail_candidate = None
        for c in generated_csv:
            if "summary" in c.name.lower():
                summary_candidate = c
            if c.name.lower() == "summary.csv":
                detail_candidate = c
        if summary_candidate is None:
            summary_candidate = find_first_csv(run_dir)
        # Prefer combined_summary.csv if available.
        for c in generated_csv:
            if c.name.lower() == "combined_summary.csv":
                summary_candidate = c
                break

        metrics = parse_summary_csv(summary_candidate) if summary_candidate else {
            "num_images": None,
            "det_rate_carapace": None,
            "det_rate_total": None,
            "mae_carapace": None,
            "mae_total": None,
            "mpe_carapace": None,
            "mpe_total": None,
        }
        if metrics["num_images"] is None and detail_candidate is not None:
            metrics["num_images"] = parse_num_samples_from_detail_csv(detail_candidate)
        write_per_k_summary(k_out_dir, k, metrics)

        status = "ok" if rc == 0 else f"eval_failed_rc_{rc}"
        global_rows.append(
            {
                "direction": direction,
                "K": k,
                "number_of_samples": metrics["num_images"],
                "detection_rate_carapace": metrics["det_rate_carapace"],
                "detection_rate_total": metrics["det_rate_total"],
                "MAE_carapace": metrics["mae_carapace"],
                "MAE_total": metrics["mae_total"],
                "MPE_carapace": metrics["mpe_carapace"],
                "MPE_total": metrics["mpe_total"],
                "status": status,
            }
        )
        print(f"  Finished {direction}/{k_name} with status: {status}")

    global_csv = build_global_summary(global_rows, out_root)
    print(f"\nDone. Global summary: {global_csv}")


if __name__ == "__main__":
    main()

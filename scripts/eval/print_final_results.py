#!/usr/bin/env python3

import pandas as pd
from pathlib import Path

# ============================================================
# PATHS
# ============================================================

ROOT = Path("/Users/taircarmon/Desktop/data-centric-cross-season-transfer")

# ----------- 2024 full model -----------
FULL_2024_TEST_2025 = ROOT / "scripts/eval/outputs/2024_Full_models/test_on_2025/all_models_combined_summaries.xlsx"
FULL_2024_TEST_2024 = ROOT / "scripts/eval/outputs/2024_Full_models/test_on_2024/all_models_pond_summaries.xlsx"
SHEET_2024 = "all_ponds_weights_best_pt"

# ----------- 2025 full model -----------
FULL_2025_TEST_2024 = ROOT / "scripts/eval/outputs/2025_Full_models/test_on_2024/all_models_pond_summaries.xlsx"
FULL_2025_TEST_2025 = ROOT / "scripts/eval/outputs/2025_Full_models/test_on_2025/all_models_combined_summaries.xlsx"
SHEET_2025 = "YOLOv11n_train_on_2025_all_pose"

# ----------- Full Transfer -----------
TF_FIXED_2024_TO_2025 = ROOT / "scripts/eval/outputs_TF/final/Fixed/test_on_2025/all_models_combined_summaries.xlsx"
TF_FIXED_2025_TO_2024 = ROOT / "scripts/eval/outputs_TF/final/Fixed/test_on_2024/all_models_pond_summaries.xlsx"

# ----------- Geometry Aware -----------
GA_2024_TO_2025 = ROOT / "scripts/eval/outputs_TF/TL_Optim_core_set/Adaptive_Shift/2024_to_2025/test_on_2025/all_models_combined_summaries.xlsx"
GA_2025_TO_2024 = ROOT / "scripts/eval/outputs_TF/TL_Optim_core_set/Adaptive_Shift/2025_to_2024/test_on_2024/all_models_pond_summaries.xlsx"

# ============================================================
# HELPERS
# ============================================================

def find_sheet_contains(path, keyword):
    xls = pd.ExcelFile(path)
    for s in xls.sheet_names:
        if keyword in s:
            return s
    raise ValueError(f"No sheet containing '{keyword}' found in {path}")

def get_first_non_all_sheet(path):
    xls = pd.ExcelFile(path)
    for s in xls.sheet_names:
        if s != "ALL_MODELS":
            return s
    raise ValueError(f"No valid sheet found in {path}")

def clean_percent(x):
    if isinstance(x, str):
        return float(x.replace("%", "").strip())
    return float(x)

def clean_mm(x):
    if isinstance(x, str):
        return float(x.replace("mm", "").strip())
    return float(x)

def parse_test_on_2025(path, sheet):
    df = pd.read_excel(path, sheet_name=sheet)
    row = df.iloc[0]

    return {
        "CL_Det": clean_percent(row["Detection Rate – carapace (%)"]),
        "TL_Det": clean_percent(row["Detection Rate – total (%)"]),
        "MAE_CL": clean_mm(row["MAE carapace (mm)"]),
        "MPE_CL": clean_percent(row["MPE carapace (%)"]),
        "MAE_TL": clean_mm(row["MAE total (mm)"]),
        "MPE_TL": clean_percent(row["MPE total (%)"]),
    }

def parse_test_on_2024(path, sheet):
    df = pd.read_excel(path, sheet_name=sheet, index_col=0)

    def clean_row(row):
        values = []
        for v in row.values:
            if isinstance(v, str):
                v = v.replace("%", "").replace("mm", "").strip()
            values.append(float(v))
        return sum(values) / len(values)

    return {
        "CL_Det": clean_row(df.loc["Detection Rate – carapace (%)"]),
        "TL_Det": clean_row(df.loc["Detection Rate – total (%)"]),
        "MAE_CL": clean_row(df.loc["MAE carapace (mm)"]),
        "MPE_CL": clean_row(df.loc["MPE carapace (%)"]),
        "MAE_TL": clean_row(df.loc["MAE total (mm)"]),
        "MPE_TL": clean_row(df.loc["MPE total (%)"]),
    }

def print_block(title, results_dict):
    print("\n" + "=" * 85)
    print(title)
    print("=" * 85)
    df = pd.DataFrame(results_dict).T
    print(df.round(3))


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    results_2425 = {}
    results_2524 = {}

    # Detect sheets dynamically
    sheet_tf_2425 = find_sheet_contains(TF_FIXED_2024_TO_2025, "2024all_to_2025all")
    sheet_tf_2524 = find_sheet_contains(TF_FIXED_2025_TO_2024, "2025all_to_2024all")

    sheet_ga_2425 = get_first_non_all_sheet(GA_2024_TO_2025)
    sheet_ga_2524 = get_first_non_all_sheet(GA_2025_TO_2024)

    # =========================
    # 2024 → 2025
    # =========================
    results_2425["No Adaptation (2024→2025)"] = parse_test_on_2025(
        FULL_2024_TEST_2025, SHEET_2024
    )

    results_2425["Full Transfer (2024→2025)"] = parse_test_on_2025(
        TF_FIXED_2024_TO_2025, sheet_tf_2425
    )

    results_2425["Geometry-Aware (2024→2025)"] = parse_test_on_2025(
        GA_2024_TO_2025, sheet_ga_2425
    )

    # =========================
    # 2025 → 2024
    # =========================
    results_2524["No Adaptation (2025→2024)"] = parse_test_on_2024(
        FULL_2025_TEST_2024, SHEET_2025
    )

    results_2524["Full Transfer (2025→2024)"] = parse_test_on_2024(
        TF_FIXED_2025_TO_2024, sheet_tf_2524
    )

    results_2524["Geometry-Aware (2025→2024)"] = parse_test_on_2024(
        GA_2025_TO_2024, sheet_ga_2524
    )

    print_block("RESULTS: 2024 → 2025", results_2425)
    print_block("RESULTS: 2025 → 2024", results_2524)
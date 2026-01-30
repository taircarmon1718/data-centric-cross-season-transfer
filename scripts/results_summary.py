from pathlib import Path
import pandas as pd

# ================== CONFIG ==================
# Two root directories where model folders and their summary tables are located:
DATASET_DIRS = [
    Path("outputs/dual_Sets_multi_models_test_on_2025Images"),
    Path("outputs/dual_Sets_multi_models_test_on_2024Images"),
]

# Output file names:
OUT_XLSX = Path("outputs/aggregated_pond_summaries.xlsx")
OUT_CSV_LONG = Path("outputs/aggregated_pond_summaries_long.csv")

# Possible pond columns in the summary tables (keep only those that actually exist)
TARGET_COLS_ORDER = ["Circular 1", "Circular 2", "Square", "Combined"]
METRIC_COL_NAME = "Metric"
MODEL_COL_NAME = "Model"
DATASET_COL_NAME = "Dataset"
# ============================================


def read_model_pond_table(model_dir: Path) -> pd.DataFrame | None:
    """
    Reads the POND_SUMMARY table of a given model:
      - First tries pond_summary.csv
      - If not found, looks for <model>_pond_summary.xlsx and reads sheet POND_SUMMARY
    Returns a DataFrame with index=Metric and columns=pond names.
    If nothing is found, returns None.
    """
    csv_path = model_dir / "pond_summary.csv"
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path, index_col=0)
            return df
        except Exception:
            pass

    # Look for Excel file (e.g., mymodel_pond_summary.xlsx)
    xlsx_candidates = list(model_dir.glob("*_pond_summary.xlsx"))
    for xp in xlsx_candidates:
        try:
            df = pd.read_excel(xp, sheet_name="POND_SUMMARY", index_col=0)
            return df
        except Exception:
            continue
    return None


def infer_dataset_name(dataset_root: Path) -> str:
    """
    Extracts '2025' or '2024' from the dataset folder name.
    """
    name = dataset_root.name
    if "2025" in name:
        return "2025"
    if "2024" in name:
        return "2024"
    return name  # fallback


def aggregate_all_tables(dataset_dirs: list[Path]) -> pd.DataFrame:
    """
    Aggregates all pond summary tables from all models and datasets into one long table:
      Dataset | Model | Metric | <Pond Columns...>
    Keeps only pond columns that actually exist in the data.
    """
    rows = []
    seen_any_cols = set()

    for root in dataset_dirs:
        if not root.exists():
            print(f"[WARN] Dataset dir not found: {root}")
            continue
        dataset_name = infer_dataset_name(root)

        # Each subdirectory is considered a model
        for model_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
            model_name = model_dir.name
            df = read_model_pond_table(model_dir)
            if df is None or df.empty:
                print(f"[INFO] No pond summary found for model: {model_name} under {root}")
                continue

            # Track which pond columns actually exist
            pond_cols = [c for c in df.columns if c in TARGET_COLS_ORDER]
            seen_any_cols.update(pond_cols)

            # Convert to long format: one row per metric
            for metric, row in df.iterrows():
                rec = {
                    DATASET_COL_NAME: dataset_name,
                    MODEL_COL_NAME: model_name,
                    METRIC_COL_NAME: metric,
                }
                # Keep only existing pond columns
                for c in pond_cols:
                    rec[c] = row.get(c)
                rows.append(rec)

    if not rows:
        return pd.DataFrame(columns=[DATASET_COL_NAME, MODEL_COL_NAME, METRIC_COL_NAME] + list(TARGET_COLS_ORDER))

    agg = pd.DataFrame(rows)

    # Column order: Dataset, Model, Metric, then pond columns in target order (only those present)
    pond_cols_present = [c for c in TARGET_COLS_ORDER if c in agg.columns]
    agg = agg[[DATASET_COL_NAME, MODEL_COL_NAME, METRIC_COL_NAME] + pond_cols_present]

    # Fill missing values with "N/A"
    agg = agg.fillna("N/A")
    return agg


def write_outputs(agg_long: pd.DataFrame, out_xlsx: Path, out_csv_long: Path):
    """
    Saves:
      1) Long CSV with all records
      2) XLSX file with:
         - Sheet 'ALL' (long table: Dataset, Model, Metric, ponds)
         - One sheet per Dataset
    """
    # 1) Save long CSV
    out_csv_long.parent.mkdir(parents=True, exist_ok=True)
    agg_long.to_csv(out_csv_long, index=False)
    print(f"[INFO] Wrote long CSV: {out_csv_long}")

    # 2) Save Excel workbook
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as writer:
        # Write ALL sheet
        agg_long.to_excel(writer, sheet_name="ALL", index=False)

        # Write one sheet per dataset
        if not agg_long.empty:
            for ds in sorted(agg_long[DATASET_COL_NAME].unique()):
                sub = agg_long[agg_long[DATASET_COL_NAME] == ds].copy()
                sub.to_excel(writer, sheet_name=f"DS_{ds}", index=False)

    print(f"[INFO] Wrote aggregated workbook: {out_xlsx}")


def main():
    agg_long = aggregate_all_tables(DATASET_DIRS)
    if agg_long.empty:
        print("[WARN] No tables found. Check your paths or files.")
        return
    write_outputs(agg_long, OUT_XLSX, OUT_CSV_LONG)
    print("[INFO] Done.")


if __name__ == "__main__":
    main()

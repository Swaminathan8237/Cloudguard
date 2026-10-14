"""
Script: src/revert_engineered_columns.py
Purpose: Reverts data/water_quality_dataset1.csv back to the original 5 columns:
         ['pH', 'TDS_ppm', 'Turbidity_NTU', 'Temperature_C', 'Label']
         using the clean backup from data/water_quality_dataset1_original.csv.
"""

from pathlib import Path
import shutil
import csv

DATA_PATH = Path("data/water_quality_dataset1.csv")
BACKUP_PATH = Path("data/water_quality_dataset1_original.csv")

def revert():
    if BACKUP_PATH.exists():
        shutil.copyfile(BACKUP_PATH, DATA_PATH)
        print(f"Successfully restored {DATA_PATH} from {BACKUP_PATH}")
    else:
        print(f"Error: {BACKUP_PATH} not found.")
        return

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        row_count = sum(1 for _ in reader)

    print(f"Dataset shape now: {row_count:,} rows × {len(header)} columns")
    print(f"Columns: {header}")

if __name__ == "__main__":
    revert()


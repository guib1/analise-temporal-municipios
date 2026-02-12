import pandas as pd
import sys
from pathlib import Path

def analyze_csv(file_path):
    print(f"Analyzing: {file_path}")
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    total_rows = len(df)
    print(f"Total Rows: {total_rows}")
    
    if total_rows == 0:
        print("CSV is empty.")
        return

    empty_cols = []
    non_empty_cols = []

    for col in df.columns:
        if df[col].isnull().all():
            empty_cols.append(col)
        else:
            non_empty_cols.append(col)

    print("\n--- EMPTY COLUMNS (Potential Crawler Failures) ---")
    if empty_cols:
        for col in sorted(empty_cols):
            print(f" - {col}")
    else:
        print("None! All columns have at least some data.")

    print("\n--- NON-EMPTY COLUMNS (Working Crawlers) ---")
    # Group by prefix for better readability
    from collections import defaultdict
    grouped = defaultdict(list)
    for col in non_empty_cols:
        prefix = col.split('_')[0] if '_' in col else 'other'
        grouped[prefix].append(col)
    
    for prefix, cols in sorted(grouped.items()):
        print(f"[{prefix.upper()}]: {', '.join(cols)}")

if __name__ == "__main__":
    target_dir = Path("data/output/final/by_municipio")
    files = list(target_dir.glob("*.csv"))
    if not files:
        print("No files found in target directory.")
    else:
        # Sort by modification time to get the latest
        latest_file = max(files, key=lambda f: f.stat().st_mtime)
        analyze_csv(latest_file)

#!/usr/bin/env python3
"""Convert CICIoT2023 per-attack CSV files into a single merged Parquet file.

Input layout:
    data/raw/CICIoT2023/CSV/<AttackType>/<file>.pcap.csv

Output:
    data/raw/CICIoT2023/ciciot2023.parquet

Label encoding:
    label (binary)    : 0 = benign, 1 = attack
    label_multiclass  : directory name (e.g. "DDoS-SYN_Flood", "Benign_Final")

Usage:
    uv run python scripts/convert_ciciot_csv_to_parquet.py
    uv run python scripts/convert_ciciot_csv_to_parquet.py \
        --input data/raw/CICIoT2023/CSV --output data/raw/CICIoT2023
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

BENIGN_LABEL = "Benign_Final"
SKIP_NAMES = {".DS_Store"}


def convert(input_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "ciciot2023.parquet"

    subdirs = sorted(
        p for p in input_dir.iterdir()
        if p.is_dir() and p.name not in SKIP_NAMES
    )
    if not subdirs:
        print(f"[error] No subdirectories found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    chunks: list[pd.DataFrame] = []
    total_rows = 0

    for subdir in subdirs:
        csv_files = sorted(subdir.glob("*.csv"))
        if not csv_files:
            continue
        label_multiclass = subdir.name
        label_binary = 0 if label_multiclass == BENIGN_LABEL else 1

        for csv_path in csv_files:
            try:
                df = pd.read_csv(csv_path, low_memory=False)
            except Exception as exc:
                print(f"[warn] Skipping {csv_path}: {exc}", file=sys.stderr)
                continue
            df["label"] = label_binary
            df["label_multiclass"] = label_multiclass
            chunks.append(df)
            total_rows += len(df)
            print(f"  [{label_multiclass}] {csv_path.name}: {len(df):,} rows")

    if not chunks:
        print("[error] No CSV files found.", file=sys.stderr)
        sys.exit(1)

    merged = pd.concat(chunks, ignore_index=True)
    # Normalize column names: strip whitespace
    merged.columns = [c.strip() for c in merged.columns]

    merged.to_parquet(out_path, index=False, compression="snappy")
    print(f"\n[ok] Wrote {total_rows:,} rows -> {out_path}")
    print(f"     Binary label distribution:\n{merged['label'].value_counts().to_string()}")
    print(f"     Multiclass distribution:\n{merged['label_multiclass'].value_counts().to_string()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert CICIoT2023 CSVs to Parquet.")
    parser.add_argument(
        "--input",
        default="data/raw/CICIoT2023/CSV",
        help="Path to CICIoT2023 CSV directory.",
    )
    parser.add_argument(
        "--output",
        default="data/raw/CICIoT2023",
        help="Output directory for Parquet file.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent
    input_dir = repo_root / args.input
    output_dir = repo_root / args.output

    if not input_dir.exists():
        print(f"[error] Input directory not found: {input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Converting: {input_dir} -> {output_dir}")
    convert(input_dir, output_dir)


if __name__ == "__main__":
    main()

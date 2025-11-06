"""
Basic cleaning step.

- Reads the input CSV (artifact name or local file).
- Filters rows by price range.
- Removes rows outside the NYC lat/lon bounding box.
- Saves the cleaned CSV as the specified output artifact.
- Also writes a copy to the project root so downstream steps can read it locally.

Run via MLflow entry point with parameters in MLproject.
"""

import argparse
import os
from pathlib import Path
import pandas as pd


# ---- NYC bounding box (approx) ----
LAT_MIN, LAT_MAX = 40.3, 41.2
LON_MIN, LON_MAX = -74.3, -73.5
# -----------------------------------


def _resolve_input_path(input_artifact: str) -> Path:
    """
    Resolve a readable CSV path from an artifact string or local file.

    Args:
        input_artifact (str): e.g., "sample1.csv:latest" or "sample.csv"

    Returns:
        Path: existing path to a CSV file
    """
    # 1) Direct filename from the artifact (strip possible :version)
    candidate = Path(input_artifact.split(":")[0])
    if candidate.exists():
        return candidate

    # 2) Common fallbacks when running offline / local
    here = Path(__file__).resolve().parent
    for p in [
        here / "sample.csv",
        Path.cwd() / "sample.csv",
        Path(__file__).resolve().parents[2] / "sample.csv",  # project root
    ]:
        if p.exists():
            return p

    raise FileNotFoundError(
        f"Could not locate input CSV. Tried: {candidate}, 'sample.csv' in working dirs."
    )


def go(
    input_artifact: str,
    output_artifact: str,
    output_type: str,
    output_description: str,
    min_price: float,
    max_price: float,
) -> None:
    """
    Clean the dataset.

    Args:
        input_artifact (str): Input CSV artifact name or local filename.
        output_artifact (str): Output CSV filename to write (e.g., clean_sample.csv).
        output_type (str): Artifact type (kept for compatibility/logging).
        output_description (str): Description (kept for compatibility/logging).
        min_price (float): Minimum allowed price (inclusive).
        max_price (float): Maximum allowed price (inclusive).
    """
    print(f"Reading input: {input_artifact}")
    input_path = _resolve_input_path(input_artifact)
    df = pd.read_csv(input_path)

    # ---- Price range filter ----
    before = len(df)
    if "price" in df.columns:
        df = df[df["price"].between(min_price, max_price, inclusive="both")]
    after = len(df)
    print(f"Price filter [{min_price}, {max_price}] removed {before - after} rows (kept {after}).")

    # ---- NYC boundary filter (new for v1.0.1) ----
    if {"latitude", "longitude"}.issubset(df.columns):
        before = len(df)
        df = df[
            df["latitude"].between(LAT_MIN, LAT_MAX, inclusive="both")
            & df["longitude"].between(LON_MIN, LON_MAX, inclusive="both")
        ]
        after = len(df)
        print(f"NYC boundary filter removed {before - after} rows (kept {after}).")
    else:
        print("Warning: 'latitude'/'longitude' columns not found; skipping NYC boundary filter.")

    # ---- Save outputs ----
    out_path = Path(output_artifact)
    df.to_csv(out_path, index=False)
    print(f"Wrote cleaned data to {out_path}")

    # Also write a copy to the project root so downstream steps (running in temp dirs)
    # can reliably read '../../clean_sample.csv' as used in data_split.
    proj_root_copy = Path(__file__).resolve().parents[2] / out_path.name
    try:
        df.to_csv(proj_root_copy, index=False)
        print(f"Copied cleaned data to project root: {proj_root_copy}")
    except Exception as e:
        print(f"Note: could not copy cleaned data to project root: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Basic data cleaning with NYC boundary filter.")
    parser.add_argument("--input_artifact", type=str, required=True, help="Input CSV (artifact or file)")
    parser.add_argument("--output_artifact", type=str, required=True, help="Output CSV filename")
    parser.add_argument("--output_type", type=str, required=True, help="Artifact type (kept for compatibility)")
    parser.add_argument("--output_description", type=str, required=True, help="Description (kept for compatibility)")
    parser.add_argument("--min_price", type=float, required=True, help="Minimum allowed price")
    parser.add_argument("--max_price", type=float, required=True, help="Maximum allowed price")
    args = parser.parse_args()

    go(
        input_artifact=args.input_artifact,
        output_artifact=args.output_artifact,
        output_type=args.output_type,
        output_description=args.output_description,
        min_price=args.min_price,
        max_price=args.max_price,
    )

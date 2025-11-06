#!/usr/bin/env python
"""
Basic cleaning: read input data, filter/clean, and (optionally) log a cleaned artifact.
When W&B is offline, we read local 'sample.csv' instead of using artifacts.
"""
import argparse
import logging
import os
import pandas as pd

# Only import wandb when available; we won't require it in offline mode
try:
    import wandb
except Exception:
    wandb = None

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def is_offline():
    # Treat option 3 selection and general env toggles as offline
    return (
        os.environ.get("WANDB_MODE", "").lower() == "offline"
        or os.environ.get("WANDB_DISABLED", "").lower() == "true"
        or os.environ.get("WANDB_DISABLE_CODE", "")  # some envs use this
    )


# DO NOT MODIFY signature used by mlflow
def go(args):
    offline = is_offline()

    # --- Input data ---
    if offline or wandb is None:
        logger.info("W&B offline detected: reading local 'sample.csv'")
        input_path = "sample.csv"
    else:
        logger.info("W&B online detected: using artifact %s", args.input_artifact)
        run = wandb.init(project="nyc_airbnb", job_type="basic_cleaning", group="cleaning", save_code=True)
        input_path = run.use_artifact(args.input_artifact).file()

    df = pd.read_csv(input_path)

    # --- Cleaning ---
    # Price range
    idx = df["price"].between(args.min_price, args.max_price)
    df = df[idx].copy()

    # Dates
    if "last_review" in df.columns:
        df["last_review"] = pd.to_datetime(df["last_review"])

    # NYC bounds
    if "longitude" in df.columns and "latitude" in df.columns:
        idx = df["longitude"].between(-74.25, -73.50) & df["latitude"].between(40.5, 41.2)
        df = df[idx].copy()

    # Save cleaned CSV
    output_csv = "clean_sample.csv"
    df.to_csv(output_csv, index=False)
    logger.info("Wrote cleaned data to %s", output_csv)

    # --- Optional artifact logging ---
    if not offline and wandb is not None:
        logger.info("Logging cleaned artifact to W&B")
        run = wandb.run or wandb.init(project="nyc_airbnb", job_type="basic_cleaning")
        artifact = wandb.Artifact(
            args.output_artifact,
            type=args.output_type,
            description=args.output_description,
        )
        artifact.add_file(output_csv)
        run.log_artifact(artifact)
        # Do NOT call artifact.wait() in offline/online mixed contexts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A very basic data cleaning")

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Name of the input artifact to download from Weights & Biases (ignored in offline mode)",
        required=True,
    )
    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Name of the cleaned dataset artifact to be created",
        required=True,
    )
    parser.add_argument(
        "--output_type",
        type=str,
        help="Type of the output artifact (e.g., 'clean_data')",
        required=True,
    )
    parser.add_argument(
        "--output_description",
        type=str,
        help="Description of the output artifact contents",
        required=True,
    )
    parser.add_argument(
        "--min_price",
        type=float,
        help="Minimum price threshold for valid listings",
        required=True,
    )
    parser.add_argument(
        "--max_price",
        type=float,
        help="Maximum price threshold for valid listings",
        required=True,
    )

    args = parser.parse_args()
    go(args)

import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
import os

def go(args):
    # Load cleaned data
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../../clean_sample.csv"))



    # Split data
    train_df, temp_df = train_test_split(
        df,
        test_size=args.test_size + args.val_size,
        random_state=args.random_seed,
        stratify=df[args.stratify_by] if args.stratify_by in df.columns else None,
    )
    relative_val_size = args.val_size / (args.test_size + args.val_size)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_val_size,
        random_state=args.random_seed,
        stratify=temp_df[args.stratify_by] if args.stratify_by in df.columns else None,
    )

    # Save splits
    os.makedirs("outputs", exist_ok=True)
    train_df.to_csv("outputs/train.csv", index=False)
    val_df.to_csv("outputs/val.csv", index=False)
    test_df.to_csv("outputs/test.csv", index=False)

    print("✅ Data successfully split and saved in outputs/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_artifact", type=str, required=True)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--val_size", type=float, default=0.2)
    parser.add_argument("--stratify_by", type=str, default="neighbourhood_group")
    parser.add_argument("--random_seed", type=int, default=42)
    args = parser.parse_args()

    go(args)

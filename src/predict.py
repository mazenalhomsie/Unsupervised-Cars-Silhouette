"""Classify vehicles (bus / car / van) from silhouette features.

Usage (from the project root):
    python src/predict.py path/to/new_vehicles.csv [-o predictions.csv]

The input CSV needs the 18 feature columns of data/vehicle.csv.
A 'class' column, if present, is ignored. Missing values are imputed.
"""
import argparse
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "svm_pipeline.joblib"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="CSV with the 18 silhouette features")
    parser.add_argument("-o", "--output", type=Path, default=Path("predictions.csv"))
    args = parser.parse_args()

    if not MODEL.exists():
        raise SystemExit(f"Model not found at {MODEL}. Run 'python src/train.py' first.")
    bundle = joblib.load(MODEL)
    model, features = bundle["model"], bundle["features"]

    df = pd.read_csv(args.input)
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise SystemExit(f"Input is missing required columns: {missing}")

    X = df[features]
    proba = model.predict_proba(X)
    out = df.copy()
    out["predicted_class"] = model.predict(X)
    out["confidence"] = proba.max(axis=1).round(3)
    out.to_csv(args.output, index=False)

    print(out["predicted_class"].value_counts().to_string())
    print(f"\n{len(out)} predictions written to {args.output}")


if __name__ == "__main__":
    main()

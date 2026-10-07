"""Predict RUL from a CSV using the matching fitted preprocessing/model pipeline."""

import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent

def predict_frame(frame, model):
    features = list(model.feature_names_in_)
    missing = [name for name in features if name not in frame.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    # Preserve the training order and reject invalid input instead of silently filling it.
    values = frame[features].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError("Features must contain finite numeric values without missing readings.")

    output = frame.copy()
    output["Predicted_RUL"] = model.predict(values)
    return output

def main():
    parser = argparse.ArgumentParser(description="Predict turbofan RUL in operating cycles.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", type=Path, default=ROOT / "results/final_model_pipeline.joblib")
    args = parser.parse_args()

    model = joblib.load(args.model)
    predictions = predict_frame(pd.read_csv(args.input), model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output, index=False)
    print(f"Saved {len(predictions)} predictions to {args.output}")

if __name__ == "__main__":
    main()

"""Predict TARGET from a CSV with the same feature columns (TARGET optional).

Example: python XGBoost/evaluate.py paths.input=path/to/unseen.csv
This script does not calculate the undisclosed evaluation-period weighted score.
"""

from __future__ import annotations

import json

import hydra
import pandas as pd
import xgboost as xgb
from omegaconf import DictConfig

from train import metrics, read_data, repo_path, transform


@hydra.main(version_base="1.3", config_path="conf", config_name="evaluate")
def main(cfg: DictConfig) -> None:
    input_path = repo_path(str(cfg.paths.input))
    model_dir = repo_path(str(cfg.paths.model_dir))
    type_path = repo_path(str(cfg.paths.feature_types))
    spec = json.loads((model_dir / "feature_schema.json").read_text(encoding="utf-8"))
    decision = json.loads((model_dir / "decision.json").read_text(encoding="utf-8"))
    frame, _ = read_data(input_path, type_path, require_target=False)
    features = transform(frame, spec)
    model = xgb.XGBClassifier()
    model.load_model(str(model_dir / "model.json"))
    probability = model.predict_proba(features)[:, 1]
    prediction = (probability >= decision["threshold"]).astype("int8")
    output = pd.DataFrame({"LNMON": frame.LNMON, "TARGET_probability": probability,
                           "TARGET_prediction": prediction})
    output_path = repo_path(str(cfg.paths.output)) if cfg.paths.output is not None else model_dir / "predictions.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    print(f"Wrote {len(output):,} predictions to {output_path}")
    if "TARGET" in frame:
        result = metrics(frame.TARGET.to_numpy(), probability, decision["threshold"])
        print(f"Unweighted misclassification rate: {result['misclassification_rate']:.6f}")


if __name__ == "__main__":
    main()

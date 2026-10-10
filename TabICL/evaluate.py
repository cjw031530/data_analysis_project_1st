"""Predict TARGET with the selected TabICL configuration and labeled source context."""

from __future__ import annotations

import json

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from train import (file_sha256, fit_classifier, metrics, predict_probabilities,
                   read_data, repo_path, sample_context)


@hydra.main(version_base="1.3", config_path="conf", config_name="evaluate")
def main(cfg: DictConfig) -> None:
    input_path = repo_path(str(cfg.paths.input))
    data_path = repo_path(str(cfg.paths.data))
    type_path = repo_path(str(cfg.paths.feature_types))
    model_dir = repo_path(str(cfg.paths.model_dir))
    decision = json.loads((model_dir / "decision.json").read_text(encoding="utf-8"))
    if file_sha256(data_path) != decision["data_sha256"]:
        raise ValueError("Source CSV changed since training; the saved context cannot be reproduced")
    if file_sha256(type_path) != decision["feature_types_sha256"]:
        raise ValueError("Feature type map changed since training")
    source, schema = read_data(data_path, type_path)
    incoming, _ = read_data(input_path, type_path, require_target=False)
    context = sample_context(source, int(decision["context_max_rows"]), int(decision["seed"]))
    if len(context) != decision["final_context_rows"]:
        raise ValueError("Reconstructed context has a different row count")
    model = fit_classifier(context, schema, decision["features"], decision["model"], int(decision["seed"]))
    probability = predict_probabilities(model, incoming, schema, decision["features"],
                                        int(decision["prediction_batch_rows"]))
    prediction = (probability >= decision["threshold"]).astype(np.int8)
    output = pd.DataFrame({"LNMON": incoming.LNMON, "TARGET_probability": probability,
                           "TARGET_prediction": prediction})
    output_path = repo_path(str(cfg.paths.output)) if cfg.paths.output is not None else model_dir / "predictions.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    print(f"Wrote {len(output):,} predictions to {output_path}", flush=True)
    if "TARGET" in incoming:
        result = metrics(incoming.TARGET.to_numpy(), probability, decision["threshold"])
        print(f"Unweighted misclassification rate: {result['misclassification_rate']:.6f}", flush=True)


if __name__ == "__main__":
    main()

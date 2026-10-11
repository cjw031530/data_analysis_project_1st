"""Rebuild the saved TabPFN context and predict a labeled or unlabeled CSV."""

from __future__ import annotations

import csv
import json

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from TabICL.train import file_sha256, metrics, predict_probabilities, repo_path, sample_context
from TabPFN.train import fit_classifier, read_subset


@hydra.main(version_base="1.3", config_path="conf", config_name="evaluate")
def main(cfg: DictConfig) -> None:
    data_path = repo_path(str(cfg.paths.data))
    type_path = repo_path(str(cfg.paths.feature_types))
    model_dir = repo_path(str(cfg.paths.model_dir))
    decision = json.loads((model_dir / "decision.json").read_text(encoding="utf-8"))
    if file_sha256(data_path) != decision["data_sha256"]:
        raise ValueError("Source CSV changed since training")
    if file_sha256(type_path) != decision["feature_types_sha256"]:
        raise ValueError("Feature type map changed since training")
    context_end = str(cfg.context_end)
    if context_end not in {decision["test_context_end"], decision["final_context_end"]}:
        raise ValueError("context_end must match a recorded training decision")
    read_columns = ["LNMON", "TARGET"] + [name for name in decision["features"] if name != "month_index"]
    source, schema = read_subset(data_path, type_path, read_columns)
    incoming_path = repo_path(str(cfg.paths.input))
    with incoming_path.open(encoding="utf-8-sig", newline="") as input_file:
        input_header = next(csv.reader(input_file))
    incoming_columns = [name for name in read_columns if name in input_header]
    incoming, _ = read_subset(incoming_path, type_path, incoming_columns, require_target=False)
    if cfg.input_start is not None:
        incoming = incoming.loc[incoming.LNMON.ge(str(cfg.input_start))]
    if cfg.input_end is not None:
        incoming = incoming.loc[incoming.LNMON.le(str(cfg.input_end))]
    if incoming.empty:
        raise ValueError("No rows remain after the input month filter")
    if "TARGET" in incoming and incoming.LNMON.le(context_end).any():
        raise ValueError("Labeled evaluation rows must be later than the context period")
    context_source = source.loc[source.LNMON.le(context_end)]
    context = sample_context(context_source, int(decision["context_max_rows"]), int(decision["seed"]))
    expected_rows = decision["test_context_rows"] if context_end == decision["test_context_end"] else decision["final_context_rows"]
    if len(context) != expected_rows:
        raise ValueError("Reconstructed context has a different row count")
    model = fit_classifier(context, schema, decision["features"], decision["model"], int(decision["seed"]))
    probability = predict_probabilities(model, incoming, schema, decision["features"],
                                        int(decision["prediction_batch_rows"]))
    prediction = (probability >= decision["threshold"]).astype(np.int8)
    # Preserve exact float32 probabilities when readers parse the CSV as float64.
    output = pd.DataFrame({"LNMON": incoming.LNMON, "TARGET_probability": probability.astype(np.float64),
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

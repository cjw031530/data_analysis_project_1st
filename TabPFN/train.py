"""Time-ordered TabPFN classification on the repository's labeled KCB CSV."""

from __future__ import annotations

import csv
import platform
import random
from importlib.metadata import version
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion

from TabICL.train import (
    MISSING, MONTHS, best_threshold, file_sha256, make_features, metrics,
    monthly_metrics, predict_probabilities, read_type_map, repo_path,
    sample_context, select_features, variant_columns, write_json,
)


def read_subset(path: Path, type_path: Path, columns: list[str] | None = None,
                row_indices: set[int] | None = None, require_target: bool = True):
    """Read validated source columns while keeping wide CSV memory bounded."""
    with path.open(encoding="utf-8-sig", newline="") as source:
        header = next(csv.reader(source))
    schema = read_type_map(type_path, header, require_target)
    if require_target and "TARGET" not in header:
        raise ValueError("Training data must include TARGET")
    wanted = set(header if columns is None else columns)
    if not wanted.issubset(header):
        raise ValueError(f"Missing columns: {wanted - set(header)}")
    dtypes = {name: "float32" for name in wanted if schema[name] == "numeric"}
    dtypes.update({name: "string" for name in wanted if schema[name] in {"categorical", "boolean", "month_index", "target"}})
    skiprows = None if row_indices is None else lambda line: line > 0 and line - 1 not in row_indices
    frame = pd.read_csv(path, usecols=[name for name in header if name in wanted],
                        skiprows=skiprows, dtype=dtypes,
                        keep_default_na=False, na_values=MISSING, encoding="utf-8-sig",
                        low_memory=False)
    if row_indices is not None and len(frame) != len(row_indices):
        raise ValueError("Selected CSV row count differs from the requested row indices")
    frame["LNMON"] = frame.LNMON.str.strip()
    month = frame.LNMON
    valid = month.str.fullmatch(r"\d{6}").fillna(False)
    valid &= month.str[-2:].astype("Int64").between(1, 12).fillna(False)
    if not valid.all():
        raise ValueError("LNMON must contain valid YYYYMM values")
    if "TARGET" in frame:
        if not frame.TARGET.isin(["0", "1"]).all():
            raise ValueError("TARGET must contain only 0 and 1")
        frame["TARGET"] = frame.TARGET.astype("int8")
    return frame, schema


def fit_classifier(context, schema, columns, config: dict, seed: int) -> TabPFNClassifier:
    model_version = ModelVersion(config["version"])
    model = TabPFNClassifier.create_default_for_version(
        model_version,
        n_estimators=int(config["n_estimators"]),
        device=config["device"],
        n_preprocessing_jobs=int(config["preprocessing_jobs"]),
        fit_mode=config["fit_mode"],
        random_state=seed,
    )
    model.fit(make_features(context, schema, columns), context.TARGET.to_numpy())
    return model


@hydra.main(version_base="1.3", config_path="conf", config_name="train")
def main(cfg: DictConfig) -> None:
    seed = int(cfg.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(int(cfg.features.selector_jobs))
    output = repo_path(str(cfg.paths.output_dir))
    output.mkdir(parents=True, exist_ok=True)
    data_path = repo_path(str(cfg.paths.data))
    type_path = repo_path(str(cfg.paths.feature_types))
    labels, schema = read_subset(data_path, type_path, ["LNMON", "TARGET"])
    if sorted(labels.LNMON.unique().tolist()) != MONTHS:
        raise ValueError("Training CSV must contain exactly 202306 through 202405")
    train_labels = labels.loc[labels.LNMON.between("202306", "202401")]
    if cfg.variant not in {"no_month", "month_index"}:
        raise ValueError("variant must be no_month or month_index")
    selector_indices = set(sample_context(train_labels, int(cfg.features.selector_rows), seed).index)
    print(f"Selecting features from {len(selector_indices):,} training rows", flush=True)
    selector_context, _ = read_subset(data_path, type_path, row_indices=selector_indices)
    selected = select_features(selector_context, schema, cfg)
    del selector_context
    columns = variant_columns(selected, str(cfg.variant), int(cfg.features.top_k))
    frame, _ = read_subset(data_path, type_path, ["LNMON", "TARGET"] + [name for name in columns if name != "month_index"])
    del labels, train_labels
    train = frame.loc[frame.LNMON.between("202306", "202401")]
    validation = frame.loc[frame.LNMON.between("202402", "202403")]
    test = frame.loc[frame.LNMON.between("202404", "202405")]
    model_config = OmegaConf.to_container(cfg.model, resolve=True)
    context = sample_context(train, int(cfg.context.max_rows), seed)
    print(f"Validating with {len(context):,} context rows and {len(columns)} features", flush=True)
    model = fit_classifier(context, schema, columns, model_config, seed)
    probability = predict_probabilities(model, validation, schema, columns,
                                        int(cfg.prediction.batch_rows))
    threshold = best_threshold(validation.TARGET.to_numpy(), probability)
    validation_metrics = metrics(validation.TARGET.to_numpy(), probability, threshold)
    del model
    pretest = frame.loc[frame.LNMON.between("202306", "202403")]
    test_context = sample_context(pretest, int(cfg.context.max_rows), seed)
    print(f"Testing with {len(test_context):,} context rows", flush=True)
    model = fit_classifier(test_context, schema, columns, model_config, seed)
    test_probability = predict_probabilities(model, test, schema, columns,
                                             int(cfg.prediction.batch_rows))
    test_metrics = metrics(test.TARGET.to_numpy(), test_probability, threshold)
    del model
    final_context = sample_context(frame, int(cfg.context.max_rows), seed)
    decision = {
        "variant": str(cfg.variant), "features": columns, "threshold": threshold,
        "seed": seed, "context_max_rows": int(cfg.context.max_rows),
        "test_context_end": "202403", "test_context_rows": len(test_context),
        "final_context_end": "202405", "final_context_rows": len(final_context),
        "model": model_config, "prediction_batch_rows": int(cfg.prediction.batch_rows),
        "data_sha256": file_sha256(data_path),
        "feature_types_sha256": file_sha256(type_path),
    }
    write_json(output / "decision.json", decision)
    write_json(output / "metrics.json", {
        "split": {"train": "202306-202401", "validation": "202402-202403",
                  "test": "202404-202405"},
        "run_config": OmegaConf.to_container(cfg, resolve=True),
        "selector": "XGBoost gain fitted on stratified 202306-202401 rows only",
        "selector_rows": len(selector_indices),
        "validation": validation_metrics,
        "validation_by_month": monthly_metrics(validation, probability, threshold),
        "test": test_metrics,
        "test_by_month": monthly_metrics(test, test_probability, threshold),
        "environment": {"python": platform.python_version(), "tabpfn": version("tabpfn"),
                        "torch": torch.__version__},
        "note": "Test context ends 202403. Future predictions reconstruct a context through 202405.",
    })
    print(f"Test misclassification rate: {test_metrics['misclassification_rate']:.6f}", flush=True)


if __name__ == "__main__":
    main()

"""Reproducible time-ordered TabICLv2 classification for the KCB table."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import random
from importlib.metadata import version
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import sklearn
import torch
import xgboost as xgb
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from tabicl import TabICLClassifier


ROOT = Path(__file__).resolve().parents[1]
MISSING = ["", "NA", "N/A", "NAN", "NULL", "NONE", "<NA>"]
MISSING_CATEGORY = "__TABICL_MISSING__"
BOOLEAN_VALUES = {"0": "0", "1": "1", "0.0": "0", "1.0": "1", "N": "0", "Y": "1"}
MONTHS = [f"{2023 + (5 + i) // 12:04d}{(5 + i) % 12 + 1:02d}" for i in range(12)]


def repo_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def file_sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalize_text(series: pd.Series) -> pd.Series:
    stripped = series.astype("string").str.strip()
    return stripped.mask(stripped.str.upper().isin(MISSING))


def read_type_map(path: Path, header: list[str], require_target: bool) -> dict[str, str]:
    table = pd.read_csv(path, dtype="string", keep_default_na=False)
    if table.columns.tolist() != ["variable", "type"] or table.variable.duplicated().any():
        raise ValueError("feature_types.csv needs unique variable,type rows")
    if len(header) != len(set(header)):
        raise ValueError("The CSV has duplicate column names")
    schema = dict(zip(table.variable, table.type))
    if schema.get("LNMON") != "month_index" or schema.get("TARGET") != "target":
        raise ValueError("LNMON and TARGET must have their declared roles")
    if set(schema.values()) != {"numeric", "categorical", "boolean", "month_index", "target"}:
        raise ValueError("The type map contains unknown or missing roles")
    expected = set(schema) if require_target or "TARGET" in header else set(schema) - {"TARGET"}
    if set(header) != expected:
        raise ValueError(f"CSV/type-map mismatch: missing={expected - set(header)}, extra={set(header) - expected}")
    return schema


def read_data(path: Path, type_path: Path, require_target: bool = True) -> tuple[pd.DataFrame, dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        header = next(csv.reader(source))
    if require_target and "TARGET" not in header:
        raise ValueError("Training data must include TARGET")
    schema = read_type_map(type_path, header, require_target)
    dtypes = {name: "float32" for name, kind in schema.items() if kind == "numeric"}
    dtypes.update({name: "string" for name, kind in schema.items() if kind in {"categorical", "boolean"}})
    dtypes["LNMON"] = "string"
    if "TARGET" in header:
        dtypes["TARGET"] = "string"
    frame = pd.read_csv(path, dtype=dtypes, keep_default_na=False, na_values=MISSING,
                        encoding="utf-8-sig", low_memory=False)
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


def month_index(series: pd.Series) -> pd.Series:
    year = series.str[:4].astype("int32")
    month = series.str[4:].astype("int32")
    return ((year - 2023) * 12 + month - 5).astype("float32")


def make_features(frame: pd.DataFrame, schema: dict[str, str], columns: list[str]) -> pd.DataFrame:
    result = {}
    for name in columns:
        if name == "month_index":
            result[name] = month_index(frame.LNMON)
        elif schema.get(name) == "numeric":
            result[name] = frame[name].astype("float32")
        elif schema.get(name) == "categorical":
            values = normalize_text(frame[name])
            if values.eq(MISSING_CATEGORY).any():
                raise ValueError(f"Reserved missing token occurs in {name}")
            result[name] = values.fillna(MISSING_CATEGORY).astype("category")
        elif schema.get(name) == "boolean":
            values = normalize_text(frame[name]).str.upper()
            unknown = values.notna() & ~values.isin(BOOLEAN_VALUES)
            if unknown.any():
                raise ValueError(f"Unexpected boolean values in {name}: {values[unknown].unique().tolist()[:5]}")
            result[name] = values.map(BOOLEAN_VALUES).fillna(MISSING_CATEGORY).astype("category")
        else:
            raise ValueError(f"Unknown predictor: {name}")
    return pd.DataFrame(result, index=frame.index)


def select_features(train: pd.DataFrame, schema: dict[str, str], cfg: DictConfig) -> list[str]:
    predictors = [name for name in train.columns if schema.get(name) in {"numeric", "categorical", "boolean"}]
    top_k = int(cfg.features.top_k)
    if not 2 <= top_k <= len(predictors):
        raise ValueError(f"features.top_k must be between 2 and {len(predictors)}")
    model = xgb.XGBClassifier(
        objective="binary:logistic", tree_method="hist", enable_categorical=True,
        n_estimators=int(cfg.features.selector_trees), max_depth=int(cfg.features.selector_depth),
        learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        max_cat_to_onehot=4, importance_type="gain", n_jobs=int(cfg.features.selector_jobs),
        random_state=int(cfg.seed),
    )
    model.fit(make_features(train, schema, predictors), train.TARGET.to_numpy())
    ranks = sorted(range(len(predictors)), key=lambda i: (-float(model.feature_importances_[i]), i))
    selected = [predictors[i] for i in ranks[:top_k]]
    del model
    return selected


def variant_columns(selected: list[str], variant: str, top_k: int) -> list[str]:
    if variant == "no_month":
        return selected[:top_k]
    if variant == "month_index":
        return selected[:top_k - 1] + ["month_index"]
    raise ValueError(f"Unknown variant: {variant}")


def sample_context(frame: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    if max_rows < 2:
        raise ValueError("context.max_rows must be at least 2")
    if len(frame) <= max_rows:
        return frame
    strata = frame.LNMON.astype(str) + "_" + frame.TARGET.astype(str)
    split = StratifiedShuffleSplit(n_splits=1, train_size=max_rows, random_state=seed)
    indices, _ = next(split.split(np.zeros(len(frame)), strata))
    return frame.iloc[np.sort(indices)]


def make_classifier(config: dict, seed: int) -> TabICLClassifier:
    model_path = config.get("model_path")
    return TabICLClassifier(
        checkpoint_version=config["checkpoint_version"],
        model_path=str(repo_path(model_path)) if model_path else None,
        device=config.get("device"), n_estimators=int(config["n_estimators"]),
        batch_size=int(config["batch_size"]), kv_cache=config["kv_cache"],
        offload_mode=config["offload_mode"], n_jobs=int(config["n_jobs"]),
        random_state=seed,
    )


def predict_probabilities(model: TabICLClassifier, frame: pd.DataFrame,
                          schema: dict[str, str], columns: list[str], batch_rows: int) -> np.ndarray:
    if batch_rows < 1:
        raise ValueError("prediction.batch_rows must be positive")
    positive_index = list(model.classes_).index(1)
    parts = []
    for start in range(0, len(frame), batch_rows):
        batch = make_features(frame.iloc[start:start + batch_rows], schema, columns)
        parts.append(model.predict_proba(batch)[:, positive_index])
        print(f"Predicted {min(start + batch_rows, len(frame)):,}/{len(frame):,} rows", flush=True)
    return np.concatenate(parts) if parts else np.empty(0, dtype=float)


def fit_classifier(context: pd.DataFrame, schema: dict[str, str], columns: list[str],
                   config: dict, seed: int) -> TabICLClassifier:
    model = make_classifier(config, seed)
    model.fit(make_features(context, schema, columns), context.TARGET.to_numpy())
    return model


def best_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """Threshold minimizing ordinary 0/1 error, as in XGBoost/train.py."""
    order = np.argsort(-probabilities, kind="mergesort")
    y = labels[order].astype(np.int8)
    p = probabilities[order]
    errors = int(y.sum())
    best = (errors, 0.0, 1.0 + np.finfo(float).eps)
    previous = -1
    for end in np.r_[np.flatnonzero(np.diff(p)), len(p) - 1]:
        group = y[previous + 1:end + 1]
        errors += len(group) - 2 * int(group.sum())
        threshold = float(p[end])
        candidate = (errors, abs(threshold - 0.5), threshold)
        if candidate < best:
            best = candidate
        previous = end
    return best[2]


def metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    predicted = (probabilities >= threshold).astype(np.int8)
    tp = int(((labels == 1) & (predicted == 1)).sum())
    fp = int(((labels == 0) & (predicted == 1)).sum())
    fn = int(((labels == 1) & (predicted == 0)).sum())
    tn = int(((labels == 0) & (predicted == 0)).sum())
    return {
        "rows": len(labels), "positive_rate": float(labels.mean()),
        "threshold": float(threshold), "misclassification_rate": float((predicted != labels).mean()),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "roc_auc": float(roc_auc_score(labels, probabilities)) if len(np.unique(labels)) == 2 else None,
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def monthly_metrics(frame: pd.DataFrame, probabilities: np.ndarray, threshold: float) -> dict:
    months = frame.LNMON.to_numpy()
    labels = frame.TARGET.to_numpy()
    return {month: metrics(labels[months == month], probabilities[months == month], threshold)
            for month in sorted(set(months))}


@hydra.main(version_base="1.3", config_path="conf", config_name="train")
def main(cfg: DictConfig) -> None:
    seed = int(cfg.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    output = repo_path(str(cfg.paths.output_dir))
    output.mkdir(parents=True, exist_ok=True)
    data_path = repo_path(str(cfg.paths.data))
    type_path = repo_path(str(cfg.paths.feature_types))
    frame, schema = read_data(data_path, type_path)
    if sorted(frame.LNMON.unique().tolist()) != MONTHS:
        raise ValueError("Training CSV must contain exactly 202306 through 202405")
    train = frame.loc[frame.LNMON.between("202306", "202401")]
    valid = frame.loc[frame.LNMON.between("202402", "202403")]
    test = frame.loc[frame.LNMON.between("202404", "202405")]
    variants = list(cfg.variants)
    if len(variants) != len(set(variants)) or not variants or set(variants) - {"no_month", "month_index"}:
        raise ValueError("variants must be a nonempty unique subset of no_month, month_index")
    context_rows = int(cfg.context.max_rows)
    batch_rows = int(cfg.prediction.batch_rows)
    model_config = OmegaConf.to_container(cfg.model, resolve=True)
    selected = select_features(train, schema, cfg)
    context = sample_context(train, context_rows, seed)
    candidates = {}
    for variant in variants:
        columns = variant_columns(selected, variant, int(cfg.features.top_k))
        print(f"Validating {variant}: {len(context):,} context rows, {len(columns)} features", flush=True)
        model = fit_classifier(context, schema, columns, model_config, seed)
        probability = predict_probabilities(model, valid, schema, columns, batch_rows)
        threshold = best_threshold(valid.TARGET.to_numpy(), probability)
        candidates[variant] = {
            "features": columns, "context_rows": len(context), "threshold": threshold,
            "validation": metrics(valid.TARGET.to_numpy(), probability, threshold),
            "validation_by_month": monthly_metrics(valid, probability, threshold),
        }
        del model
    chosen = min(candidates, key=lambda name: (
        candidates[name]["validation"]["misclassification_rate"],
        -candidates[name]["validation"]["average_precision"],
    ))
    selected_candidate = candidates[chosen]
    print(f"Testing {chosen} on 202404-202405", flush=True)
    model = fit_classifier(context, schema, selected_candidate["features"], model_config, seed)
    test_probability = predict_probabilities(model, test, schema, selected_candidate["features"], batch_rows)
    test_metrics = metrics(test.TARGET.to_numpy(), test_probability, selected_candidate["threshold"])
    del model
    print(f"Final context through 202405; test error={test_metrics['misclassification_rate']:.6f}", flush=True)
    final_selected = select_features(frame, schema, cfg)
    final_columns = variant_columns(final_selected, chosen, int(cfg.features.top_k))
    final_context = sample_context(frame, context_rows, seed)
    data_hash = file_sha256(data_path)
    type_hash = file_sha256(type_path)
    write_json(output / "decision.json", {
        "variant": chosen, "threshold": selected_candidate["threshold"],
        "features": final_columns, "seed": seed, "context_max_rows": context_rows,
        "model": model_config, "prediction_batch_rows": batch_rows,
        "data_sha256": data_hash, "feature_types_sha256": type_hash,
        "final_context_rows": len(final_context),
    })
    write_json(output / "metrics.json", {
        "split": {"train": "202306-202401", "validation": "202402-202403",
                  "test": "202404-202405", "final_context": "202306-202405"},
        "run_config": OmegaConf.to_container(cfg, resolve=True),
        "selector": "XGBoost gain fitted only on the available training period",
        "candidates": candidates, "chosen": chosen,
        "test_before_final_context": test_metrics,
        "test_by_month": monthly_metrics(test, test_probability, selected_candidate["threshold"]),
        "final_features": final_columns, "final_context_rows": len(final_context),
        "environment": {"python": platform.python_version(), "tabicl": version("tabicl"),
                        "torch": torch.__version__, "sklearn": sklearn.__version__, "xgboost": xgb.__version__,
                        "data_sha256": data_hash, "feature_types_sha256": type_hash},
        "note": "Test metrics use the pre-final context. Inference re-creates the final context from the source CSV.",
    })


if __name__ == "__main__":
    main()

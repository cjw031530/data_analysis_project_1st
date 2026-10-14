"""Seven preprocessing comparisons with rolling F7-F9 and a weighted F10-F12 test.

Run from the repository root: python -m XGBoost.preprocessing_experiment
All preprocessing is fitted inside each training window. Test labels are used
only after all method, parameter, iteration, and threshold choices are frozen.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import pickle
import platform
import time
from pathlib import Path

import hydra
import numpy as np
import optuna
import pandas as pd
import sklearn
import xgboost as xgb
from omegaconf import DictConfig, OmegaConf
from sklearn.preprocessing import OneHotEncoder

from XGBoost.train import (
    MISSING_CATEGORY, best_threshold, fit_preprocessor, metrics, normalize_text,
    read_data, repo_path, suggest_params, transform,
)


METHODS = {
    "raw_minimal": "1. 원형 · 최소 변환",
    "current": "2. 현재 전처리",
    "drop_missing": "3. 결측 50% 초과 열 삭제",
    "one_hot": "4. 현재 + One-hot",
    "drop_missing_keep_flags": "5. 삭제 + 결측 여부 유지",
    "drop_categorical": "6. 범주형 변수 제외",
    "numeric_categories": "7. 숫자 범주를 수치형으로",
}


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def file_hash(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def missing_mask(frame: pd.DataFrame, name: str, schema: dict) -> pd.Series:
    return frame[name].isna() if name in schema["numeric"] else normalize_text(frame[name]).isna()


def numeric_tokens(series: pd.Series) -> pd.Series:
    return pd.to_numeric(normalize_text(series), errors="coerce").astype("float32")


def all_numeric(series: pd.Series) -> bool:
    text = normalize_text(series)
    observed = text.dropna()
    return bool(len(observed) and pd.to_numeric(observed, errors="coerce").notna().all())


class ExperimentPreprocessor:
    """Fit category vocabularies and column decisions using training rows only."""

    def __init__(self, method: str, schema: dict, missing_cutoff: float = 0.5):
        if method not in METHODS:
            raise ValueError(f"Unknown method: {method}")
        self.method = method
        self.schema = schema
        self.missing_cutoff = missing_cutoff

    def fit(self, train: pd.DataFrame):
        self.spec = fit_preprocessor(train, self.schema, include_month=False)
        self.original_columns = list(self.spec["column_order"])
        self.missing_rates = {name: float(missing_mask(train, name, self.schema).mean())
                              for name in self.original_columns}
        self.dropped = []
        if self.method in {"drop_missing", "drop_missing_keep_flags"}:
            self.dropped = [name for name in self.original_columns
                            if self.missing_rates[name] > self.missing_cutoff]
        elif self.method == "drop_categorical":
            self.dropped = list(self.schema["categorical"])
        self.columns = [name for name in self.original_columns if name not in self.dropped]
        self.spec["column_order"] = self.columns
        self.numeric_category_columns = []
        self.raw_numeric = []
        self.raw_codes = {}
        if self.method == "raw_minimal":
            for name in self.columns:
                if name in self.schema["numeric"] or all_numeric(train[name]):
                    self.raw_numeric.append(name)
                else:
                    levels = sorted(normalize_text(train[name]).dropna().unique().tolist())
                    self.raw_codes[name] = {value: index for index, value in enumerate(levels)}
        elif self.method == "numeric_categories":
            self.numeric_category_columns = [name for name in self.schema["categorical"]
                                             if all_numeric(train[name])]
        self.encoder = None
        if self.method == "one_hot":
            self.onehot_columns = [name for name in self.columns if name in self.schema["categorical"]]
            self.other_columns = [name for name in self.columns if name not in self.onehot_columns]
            self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
            self.encoder.fit(self._category_frame(train))
        sample = self.transform(train.iloc[:min(8, len(train))])
        self.feature_count = int(sample.shape[1])
        if self.encoder is not None:
            self.feature_names = self.other_columns + self.encoder.get_feature_names_out(self.onehot_columns).tolist()
        else:
            self.feature_names = sample.columns.tolist()
        if set(self.feature_names) & {"TARGET", "LNMON", "month_index"}:
            raise AssertionError("A target or calendar month entered the predictor list")
        return self

    def _category_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({name: normalize_text(frame[name]).fillna(MISSING_CATEGORY)
                             for name in self.onehot_columns}, index=frame.index)

    def transform(self, frame: pd.DataFrame):
        if self.method == "raw_minimal":
            columns = {}
            for name in self.columns:
                if name in self.schema["numeric"]:
                    columns[name] = frame[name].astype("float32")
                elif name in self.raw_numeric:
                    columns[name] = numeric_tokens(frame[name])
                else:
                    columns[name] = normalize_text(frame[name]).map(self.raw_codes[name]).astype("float32")
            return pd.DataFrame(columns, index=frame.index)
        out = transform(frame, self.spec)
        if self.method == "numeric_categories":
            for name in self.numeric_category_columns:
                out[name] = numeric_tokens(frame[name])
        elif self.method == "drop_missing_keep_flags":
            for name in self.dropped:
                out[f"missing__{name}"] = missing_mask(frame, name, self.schema).astype("float32")
        elif self.method == "one_hot":
            # Dense float32 preserves true numeric zeros. CSR implicit zeros are
            # treated as missing by XGBoost, which would confound this experiment.
            numeric = out[self.other_columns].to_numpy(dtype=np.float32)
            encoded = self.encoder.transform(self._category_frame(frame))
            return np.concatenate((numeric, encoded), axis=1)
        return out

    def description(self) -> dict:
        return {
            "method": self.method, "label": METHODS[self.method],
            "input_features": len(self.original_columns), "output_features": self.feature_count,
            "dropped_columns": self.dropped,
            "added_missing_indicators": len(self.dropped) if self.method == "drop_missing_keep_flags" else 0,
            "numeric_category_columns": self.numeric_category_columns,
            "raw_numeric_columns": self.raw_numeric if self.method == "raw_minimal" else [],
            "raw_text_columns": list(self.raw_codes),
            "missing_cutoff": self.missing_cutoff,
            "training_missing_rates": self.missing_rates,
        }


def weighted_error(frame: pd.DataFrame, probability: np.ndarray, threshold: float, weights: dict) -> float:
    month_weights = {str(month): float(weight) for month, weight in weights.items()}
    row_weights = frame.LNMON.map(month_weights)
    if row_weights.isna().any() or len(frame) != len(probability) or frame.empty:
        raise ValueError("Invalid prediction rows or missing month weights")
    wrong = (probability >= threshold).astype(np.int8) != frame.TARGET.to_numpy()
    return float(np.dot(row_weights.to_numpy(dtype=float), wrong) / len(frame))


def prepare_folds(frame: pd.DataFrame, schema: dict, method: str, cfg: DictConfig) -> list[dict]:
    prepared = []
    for fold in cfg.folds:
        train = frame.loc[frame.LNMON.between("202306", str(fold.train_end))]
        valid = frame.loc[frame.LNMON.eq(str(fold.valid_month))]
        if train.empty or valid.empty or train.LNMON.max() >= valid.LNMON.min():
            raise ValueError(f"Invalid chronological split: {fold.name}")
        prep = ExperimentPreprocessor(method, schema, float(cfg.missing_cutoff)).fit(train)
        prepared.append({
            "name": str(fold.name), "month": str(fold.valid_month),
            "x_train": prep.transform(train), "y_train": train.TARGET.to_numpy(),
            "x_valid": prep.transform(valid), "y_valid": valid.TARGET.to_numpy(),
            "indices": valid.index.to_numpy(), "preprocessing": prep.description(),
        })
        print(f"  {fold.name}: train={len(train):,}, valid={len(valid):,}, features={prep.feature_count}", flush=True)
    return prepared


def make_experiment_model(params: dict, rounds: int, cfg: DictConfig, early_stopping: bool):
    return xgb.XGBClassifier(
        objective="binary:logistic", tree_method="hist", device=str(cfg.device),
        enable_categorical=True, n_estimators=rounds, n_jobs=int(cfg.cpu_jobs),
        random_state=int(cfg.seed), max_bin=int(cfg.model.max_bin), eval_metric="logloss",
        early_stopping_rounds=int(cfg.model.early_stopping) if early_stopping else None,
        **params,
    )


def predict(model, values, rounds: int) -> np.ndarray:
    # DMatrix prediction avoids inplace_predict device mismatch warnings while
    # supporting both native categorical DataFrames and dense one-hot arrays.
    matrix = xgb.DMatrix(values, enable_categorical=True)
    return model.get_booster().predict(matrix, iteration_range=(0, rounds))


def evaluate_cv(prepared: list[dict], params: dict, cfg: DictConfig, cache: Path, signature: str) -> dict:
    cache.mkdir(parents=True, exist_ok=True)
    params_hash = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()
    probabilities, labels, records = [], [], []
    for fold in prepared:
        meta_path = cache / f"{fold['name']}.json"
        array_path = cache / f"{fold['name']}.npz"
        if meta_path.exists() and array_path.exists():
            record = json.loads(meta_path.read_text(encoding="utf-8"))
            if record["signature"] != signature or record["params_hash"] != params_hash:
                raise ValueError(f"Cached fold configuration changed: {meta_path}")
            with np.load(array_path) as saved:
                if not np.array_equal(saved["indices"], fold["indices"]):
                    raise ValueError("Cached validation rows changed")
                probability = saved["probability"].copy()
        else:
            start = time.perf_counter()
            model = make_experiment_model(params, int(cfg.model.max_rounds), cfg, early_stopping=True)
            model.fit(fold["x_train"], fold["y_train"],
                      eval_set=[(fold["x_valid"], fold["y_valid"])], verbose=False)
            rounds = int(model.best_iteration) + 1
            probability = predict(model, fold["x_valid"], rounds)
            record = {
                "name": fold["name"], "month": fold["month"], "best_rounds": rounds,
                "train_rows": len(fold["y_train"]), "valid_rows": len(fold["y_valid"]),
                "fit_seconds": time.perf_counter() - start,
                "preprocessing": fold["preprocessing"], "signature": signature,
                "params_hash": params_hash,
            }
            np.savez_compressed(array_path, probability=probability, indices=fold["indices"])
            write_json(meta_path, record)
            del model
            gc.collect()
        labels.append(fold["y_valid"])
        probabilities.append(probability)
        records.append(record)
        print(f"  {fold['name']}: rounds={record['best_rounds']}, fit={record['fit_seconds']:.1f}s", flush=True)
    y = np.concatenate(labels)
    p = np.concatenate(probabilities)
    threshold, _ = best_threshold(y, p)
    for fold, probability, record in zip(prepared, probabilities, records):
        record["metrics"] = metrics(fold["y_valid"], probability, threshold)
    return {
        "threshold": threshold, "rounds": max(1, int(np.median([r["best_rounds"] for r in records]))),
        "pooled": metrics(y, p, threshold), "folds": records,
        "fit_seconds": float(sum(record["fit_seconds"] for record in records)),
    }


def tune_method(frame, schema, method, cfg, output, signature, baseline_result) -> dict:
    print(f"TPE: {METHODS[method]}, {cfg.tuning.trials} trials", flush=True)
    prepared = prepare_folds(frame, schema, method, cfg)
    storage_dir = output / "checkpoints" / method / "tuning"
    storage_dir.mkdir(parents=True, exist_ok=True)
    sampler_path = storage_dir / "sampler.pkl"
    if sampler_path.exists():
        with sampler_path.open("rb") as source:
            sampler = pickle.load(source)
    else:
        sampler = optuna.samplers.TPESampler(seed=int(cfg.seed), n_startup_trials=int(cfg.tuning.startup_trials))
    study = optuna.create_study(direction="minimize", study_name=method,
                               storage=f"sqlite:///{(storage_dir / 'study.sqlite3').as_posix()}",
                               sampler=sampler, load_if_exists=True)
    if study.user_attrs.get("signature", signature) != signature:
        raise ValueError("Tuning study configuration changed")
    study.set_user_attr("signature", signature)
    search = OmegaConf.to_container(cfg.tuning.search, resolve=True)
    baseline = OmegaConf.to_container(cfg.model.baseline, resolve=True)
    if not study.trials:
        study.enqueue_trial({key: baseline[key] for key in search})
    completed = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    while len(completed) < int(cfg.tuning.trials):
        running = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.RUNNING]
        if len(running) > 1:
            raise ValueError("More than one unfinished trial; run one process per output directory")
        trial = optuna.trial.Trial(study, running[0]._trial_id) if running else study.ask()
        params = {**baseline, **suggest_params(trial, search)}
        if trial.number == 0 and params == baseline:
            result = baseline_result
        else:
            result = evaluate_cv(prepared, params, cfg, storage_dir / f"trial_{trial.number:03d}", signature)
        trial.set_user_attr("params_full", params)
        trial.set_user_attr("result", result)
        study.tell(trial, result["pooled"]["misclassification_rate"])
        with sampler_path.open("wb") as destination:
            pickle.dump(study.sampler, destination)
        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        print(f"TPE {method}: {len(completed)}/{cfg.tuning.trials}, best={study.best_value:.6f}", flush=True)
        write_json(storage_dir / "trials.json", {
            "trials": [{"number": t.number, "value": t.value, "params": t.params} for t in completed]
        })
    best = study.best_trial
    del prepared
    gc.collect()
    return {"params": best.user_attrs["params_full"], "cv": best.user_attrs["result"],
            "trials": int(cfg.tuning.trials), "best_trial": best.number}


def evaluate_holdout(frame, schema, method, phase, params, cv, cfg, output, signature) -> dict:
    path = output / "checkpoints" / method / phase
    path.mkdir(parents=True, exist_ok=True)
    result_path = path / "holdout.json"
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result["signature"] != signature or result["params"] != params or result["cv"] != cv:
            raise ValueError("Saved holdout configuration changed")
        return result
    train = frame.loc[frame.LNMON.le(str(cfg.test.train_end))]
    weights = {str(month): float(weight) for month, weight in cfg.test.weights.items()}
    test = frame.loc[frame.LNMON.isin(weights)]
    if set(test.LNMON) != set(weights) or train.LNMON.max() >= test.LNMON.min():
        raise ValueError("Invalid holdout months")
    prep = ExperimentPreprocessor(method, schema, float(cfg.missing_cutoff)).fit(train)
    x_train, x_test = prep.transform(train), prep.transform(test)
    start = time.perf_counter()
    model = make_experiment_model(params, int(cv["rounds"]), cfg, early_stopping=False)
    model.fit(x_train, train.TARGET.to_numpy(), verbose=False)
    fit_seconds = time.perf_counter() - start
    probability = predict(model, x_test, int(cv["rounds"]))
    threshold = float(cv["threshold"])
    test_metrics = metrics(test.TARGET.to_numpy(), probability, threshold)
    test_metrics["weighted_score"] = weighted_error(test, probability, threshold, weights)
    test_metrics["all_zero_weighted_score"] = float(
        np.dot(test.LNMON.map(weights).to_numpy(), test.TARGET.to_numpy()) / len(test))
    monthly = {}
    for month, weight in weights.items():
        mask = test.LNMON.eq(month).to_numpy()
        monthly[month] = {**metrics(test.TARGET.to_numpy()[mask], probability[mask], threshold), "weight": weight}
    model.save_model(str(path / "model.ubj"))
    with (path / "preprocessor.pkl").open("wb") as destination:
        pickle.dump(prep, destination)
    np.savez_compressed(path / "predictions.npz", indices=test.index.to_numpy(),
                        probability=probability, labels=test.TARGET.to_numpy(), months=test.LNMON.to_numpy(dtype=str))
    result = {
        "method": method, "label": METHODS[method], "phase": phase,
        "params": params, "cv": cv, "test": test_metrics, "test_by_month": monthly,
        "preprocessing": prep.description(), "fit_seconds": fit_seconds,
        "signature": signature,
    }
    write_json(result_path, result)
    print(f"{phase} {method}: weighted={test_metrics['weighted_score']:.6f}, error={test_metrics['misclassification_rate']:.6f}", flush=True)
    del model, x_train, x_test
    gc.collect()
    return result


def export_tables(report: dict, output: Path) -> None:
    comparison, fold_rows, monthly_rows = [], [], []
    for result in report["results"]:
        shared = {"method": result["method"], "label": result["label"], "phase": result["phase"]}
        comparison.append({
            **shared, "cv_error": result["cv"]["pooled"]["misclassification_rate"],
            "cv_auc": result["cv"]["pooled"]["roc_auc"], "threshold": result["cv"]["threshold"],
            "rounds": result["cv"]["rounds"], "test_weighted_score": result["test"]["weighted_score"],
            "test_error": result["test"]["misclassification_rate"], "test_auc": result["test"]["roc_auc"],
            "test_recall": result["test"]["recall"], "test_log_loss": result["test"]["log_loss"],
            "features": result["preprocessing"]["output_features"],
            "dropped_columns": len(result["preprocessing"]["dropped_columns"]),
            "fit_seconds": result["fit_seconds"],
        })
        for fold in result["cv"]["folds"]:
            fold_rows.append({**shared, "fold": fold["name"], "month": fold["month"],
                              "features": fold["preprocessing"]["output_features"], **fold["metrics"]})
        for month, values in result["test_by_month"].items():
            monthly_rows.append({**shared, "month": month, **values})
    for name, rows in (("comparison", comparison), ("fold_metrics", fold_rows), ("monthly_metrics", monthly_rows)):
        pd.DataFrame(rows).to_csv(output / f"{name}.csv", index=False, encoding="utf-8-sig")


@hydra.main(version_base="1.3", config_path="conf", config_name="preprocessing_experiment")
def main(cfg: DictConfig) -> None:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    output = repo_path(str(cfg.paths.output_dir))
    output.mkdir(parents=True, exist_ok=True)
    if not 0 < float(cfg.missing_cutoff) < 1:
        raise ValueError("missing_cutoff must be between zero and one")
    if not 0 < int(cfg.tuning.startup_trials) < int(cfg.tuning.trials):
        raise ValueError("startup_trials must be smaller than total trials")
    if set(cfg.methods) != set(METHODS) or len(cfg.methods) != len(METHODS):
        raise ValueError("This experiment requires the seven specified methods")
    data_path, type_path = repo_path(str(cfg.paths.data)), repo_path(str(cfg.paths.feature_types))
    config = OmegaConf.to_container(cfg, resolve=True)
    signature_info = {"config": config, "data_sha256": file_hash(data_path),
                      "types_sha256": file_hash(type_path), "code_sha256": file_hash(Path(__file__)),
                      "xgboost": xgb.__version__, "sklearn": sklearn.__version__}
    signature = hashlib.sha256(json.dumps(signature_info, sort_keys=True).encode()).hexdigest()
    signature_path = output / "checkpoints" / "run_signature.json"
    if signature_path.exists() and json.loads(signature_path.read_text())["signature"] != signature:
        raise ValueError("Experiment configuration changed; choose a new output directory")
    write_json(signature_path, {"signature": signature, **signature_info})
    OmegaConf.save(cfg, output / "run_config.yaml")
    frame, schema = read_data(data_path, type_path)
    expected_months = {"202306", "202307", "202308", "202309", "202310", "202311", "202312",
                       "202401", "202402", "202403", "202404", "202405"}
    if set(frame.LNMON) != expected_months:
        raise ValueError("Source must contain exactly 202306 through 202405")
    baseline_params = OmegaConf.to_container(cfg.model.baseline, resolve=True)
    baselines = {}
    for method in cfg.methods:
        print(f"Screening {METHODS[method]}", flush=True)
        prepared = prepare_folds(frame, schema, method, cfg)
        result = evaluate_cv(prepared, baseline_params, cfg,
                             output / "checkpoints" / method / "screening", signature)
        baselines[method] = result
        write_json(output / "checkpoints" / "screening_progress.json", baselines)
        print(f"Screening {method}: pooled error={result['pooled']['misclassification_rate']:.6f}", flush=True)
        del prepared
        gc.collect()
    finalists = sorted(baselines, key=lambda method: (
        baselines[method]["pooled"]["misclassification_rate"], baselines[method]["pooled"]["log_loss"]
    ))[:int(cfg.tuning.finalists)]
    print(f"TPE finalists selected only from F7-F9: {finalists}", flush=True)
    tuned = {method: tune_method(frame, schema, method, cfg, output, signature, baselines[method])
             for method in finalists}
    write_json(output / "checkpoints" / "frozen_choices.json", {"baselines": baselines, "tuned": tuned})
    results = []
    for method in cfg.methods:
        results.append(evaluate_holdout(frame, schema, method, "screening", baseline_params,
                                       baselines[method], cfg, output, signature))
    for method, choice in tuned.items():
        results.append(evaluate_holdout(frame, schema, method, "tuned", choice["params"],
                                       choice["cv"], cfg, output, signature))
    candidates = [result for result in results if result["phase"] == "tuned"] or results
    selected = min(candidates, key=lambda result: result["cv"]["pooled"]["misclassification_rate"])
    report = {
        "protocol": {
            "folds": config["folds"], "test_train_end": str(cfg.test.train_end),
            "test_weights": config["test"]["weights"],
            "weighted_score_formula": "sum(row_weight * error) / total_test_rows",
            "threshold_selection": "One threshold minimizing pooled F7-F9 OOF errors, frozen before F10-F12",
            "rounds_selection": "Median early-stopping best rounds from F7-F9; no test early stopping",
            "calendar_feature": "excluded from all methods",
            "raw_minimal": "Original numeric values stay numeric; text values receive training-only integer codes; missing values stay NaN. No custom type classification or feature deletion.",
        },
        "tuning_finalists": finalists, "tuning": tuned,
        "selected_by_cv": {"method": selected["method"], "phase": selected["phase"]},
        "results": results,
        "environment": {"python": platform.python_version(), "xgboost": xgb.__version__,
                        "sklearn": sklearn.__version__, "optuna": optuna.__version__,
                        "data_sha256": signature_info["data_sha256"], "types_sha256": signature_info["types_sha256"]},
        "limitations": "F7-F9 metrics select preprocessing, hyperparameters, rounds, and thresholds, so they are selection scores. Earlier project work already inspected some F10-F12 months; this comparison is exploratory. Preprocessing feature roles come from the existing unsupervised type map. It is not a measurement of the unavailable June-November 2024 target data.",
    }
    write_json(output / "metrics.json", report)
    export_tables(report, output)
    from XGBoost.visualize_preprocessing import create_visualizations
    create_visualizations(output)
    print(f"Completed seven methods and {len(tuned)} tuned finalists: {output}", flush=True)


if __name__ == "__main__":
    main()

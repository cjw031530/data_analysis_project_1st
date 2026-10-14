"""Run three preprocessing methods x two calendar modes, then a small TPE search.

python -X utf8 -m XGBoost.calendar_experiment
Existing comparable no-calendar results are reused without copying checkpoints.
All choices use F7-F9 before evaluating weighted F10-F12.
"""

from __future__ import annotations

import copy
import gc
import hashlib
import json
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

from XGBoost.calendar_preprocessing import (
    BASE_METHODS, CALENDAR_MODES, CalendarPreprocessor, variant_key, variant_label,
)
from XGBoost.preprocessing_experiment import (
    evaluate_cv, export_tables, file_hash, make_experiment_model, predict, weighted_error, write_json,
)
from XGBoost.train import ROOT, metrics, read_data, repo_path, suggest_params


def preprocess(train, schema, method, mode, cfg):
    return CalendarPreprocessor(method, mode, float(cfg.missing_cutoff), schema).fit(train)


def prepare_folds(frame, schema, method, mode, cfg):
    prepared = []
    for fold in cfg.folds:
        train = frame.loc[frame.LNMON.between("202306", str(fold.train_end))]
        valid = frame.loc[frame.LNMON.eq(str(fold.valid_month))]
        if train.empty or valid.empty or train.LNMON.max() >= valid.LNMON.min():
            raise ValueError(f"Invalid chronological fold: {fold.name}")
        prep = preprocess(train, schema, method, mode, cfg)
        prepared.append({
            "name": str(fold.name), "month": str(fold.valid_month),
            "x_train": prep.transform(train), "y_train": train.TARGET.to_numpy(),
            "x_valid": prep.transform(valid), "y_valid": valid.TARGET.to_numpy(),
            "indices": valid.index.to_numpy(), "preprocessing": prep.description(),
        })
        print(f"  {fold.name}: train={len(train):,}, valid={len(valid):,}, features={prep.feature_count}, "
              f"converted={len(prep.converted_by_cardinality)}", flush=True)
    return prepared


def load_reuse(cfg, signature_info):
    if not cfg.reuse.enabled:
        return None
    source = repo_path(str(cfg.reuse.output_dir))
    if not (source / "metrics.json").exists() or not (source / "run_config.yaml").exists():
        print("Comparable prior results unavailable; train all six conditions.", flush=True)
        return None
    report = json.loads((source / "metrics.json").read_text(encoding="utf-8"))
    old_config = OmegaConf.to_container(OmegaConf.load(source / "run_config.yaml"), resolve=True)
    config = OmegaConf.to_container(cfg, resolve=True)
    for key in ("seed", "device", "cpu_jobs", "missing_cutoff", "folds", "test", "model"):
        if old_config[key] != config[key]:
            raise ValueError(f"Prior results differ in {key}; disable reuse or restore matching settings")
    for key in ("data_sha256", "types_sha256", "xgboost", "sklearn", "optuna", "python"):
        if report["environment"][key] != signature_info[key]:
            raise ValueError(f"Prior results differ in {key}; set reuse.enabled=false to retrain")
    return {"report": report, "config": old_config, "directory": source}


def source_record(reuse, method, mode, phase):
    if reuse is None or mode != "no_month" or method == "cardinality5":
        return None
    return next((r for r in reuse["report"]["results"]
                 if r["method"] == method and r["phase"] == phase), None)


def tune_variant(frame, schema, method, mode, cfg, output, signature, baseline_cv, reuse):
    key = variant_key(method, mode)
    record = source_record(reuse, method, mode, "tuned")
    if record is not None and reuse["config"]["tuning"] == OmegaConf.to_container(cfg.tuning, resolve=True):
        choice = copy.deepcopy(reuse["report"]["tuning"][method])
        choice["reused"] = True
        print(f"Reuse identical {cfg.tuning.trials}-trial TPE: {key}", flush=True)
        return choice
    prepared = prepare_folds(frame, schema, method, mode, cfg)
    storage_dir = output / "checkpoints" / key / "tuning"
    storage_dir.mkdir(parents=True, exist_ok=True)
    sampler_path = storage_dir / "sampler.pkl"
    if sampler_path.exists():
        with sampler_path.open("rb") as source:
            sampler = pickle.load(source)
    else:
        sampler = optuna.samplers.TPESampler(seed=int(cfg.seed), n_startup_trials=int(cfg.tuning.startup_trials))
    study = optuna.create_study(direction="minimize", study_name=key, sampler=sampler,
        storage=f"sqlite:///{(storage_dir / 'study.sqlite3').as_posix()}", load_if_exists=True)
    if study.user_attrs.get("signature", signature) != signature:
        raise ValueError("TPE configuration changed")
    study.set_user_attr("signature", signature)
    search = OmegaConf.to_container(cfg.tuning.search, resolve=True)
    baseline = OmegaConf.to_container(cfg.model.baseline, resolve=True)
    if not study.trials:
        study.enqueue_trial({name: baseline[name] for name in search})
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    while len(completed) < int(cfg.tuning.trials):
        running = [t for t in study.trials if t.state == optuna.trial.TrialState.RUNNING]
        if len(running) > 1:
            raise ValueError("Run one training process per output directory")
        trial = optuna.trial.Trial(study, running[0]._trial_id) if running else study.ask()
        params = {**baseline, **suggest_params(trial, search)}
        cv = baseline_cv if trial.number == 0 and params == baseline else evaluate_cv(
            prepared, params, cfg, storage_dir / f"trial_{trial.number:03d}", signature)
        trial.set_user_attr("params_full", params)
        trial.set_user_attr("result", cv)
        study.tell(trial, cv["pooled"]["misclassification_rate"])
        with sampler_path.open("wb") as target:
            pickle.dump(study.sampler, target)
        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        write_json(storage_dir / "trials.json", {"trials": [
            {"number": t.number, "value": t.value, "params": t.params} for t in completed]})
        print(f"TPE {key}: {len(completed)}/{cfg.tuning.trials}, best={study.best_value:.6f}", flush=True)
    best = study.best_trial
    del prepared
    gc.collect()
    return {"params": best.user_attrs["params_full"], "cv": best.user_attrs["result"],
            "trials": int(cfg.tuning.trials), "best_trial": best.number, "reused": False}


def evaluate_holdout(frame, schema, method, mode, phase, params, cv, cfg, output, signature, reuse):
    key = variant_key(method, mode)
    path = output / "checkpoints" / key / phase
    result_path = path / "holdout.json"
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result["signature"] != signature or result["params"] != params or result["cv"] != cv:
            raise ValueError("Saved holdout configuration changed")
        return result
    old = source_record(reuse, method, mode, phase)
    if old is not None and old["params"] == params and old["cv"] == cv:
        result = copy.deepcopy(old)
        result.update({"method": key, "label": variant_label(method, mode), "base_method": method,
            "calendar_mode": mode, "include_month": False, "reused": True, "signature": signature,
            "checkpoint_path": (reuse["directory"] / "checkpoints" / method / phase).relative_to(ROOT).as_posix(),
            "reuse_source": (reuse["directory"] / "metrics.json").relative_to(ROOT).as_posix()})
        path.mkdir(parents=True, exist_ok=True)
        write_json(result_path, result)
        print(f"Reuse holdout: {key}/{phase}, weighted={result['test']['weighted_score']:.6f}", flush=True)
        return result
    train = frame.loc[frame.LNMON.le(str(cfg.test.train_end))]
    weights = {str(month): float(weight) for month, weight in cfg.test.weights.items()}
    test = frame.loc[frame.LNMON.isin(weights)]
    if set(test.LNMON) != set(weights) or train.LNMON.max() >= test.LNMON.min():
        raise ValueError("Invalid holdout months")
    prep = preprocess(train, schema, method, mode, cfg)
    x_train, x_test = prep.transform(train), prep.transform(test)
    started = time.perf_counter()
    model = make_experiment_model(params, int(cv["rounds"]), cfg, early_stopping=False)
    model.fit(x_train, train.TARGET.to_numpy(), verbose=False)
    seconds = time.perf_counter() - started
    probability = predict(model, x_test, int(cv["rounds"]))
    threshold = float(cv["threshold"])
    scores = metrics(test.TARGET.to_numpy(), probability, threshold)
    scores["weighted_score"] = weighted_error(test, probability, threshold, weights)
    scores["all_zero_weighted_score"] = float(np.dot(
        test.LNMON.map(weights).to_numpy(), test.TARGET.to_numpy()) / len(test))
    monthly = {}
    for month, weight in weights.items():
        mask = test.LNMON.eq(month).to_numpy()
        monthly[month] = {**metrics(test.TARGET.to_numpy()[mask], probability[mask], threshold), "weight": weight}
    path.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path / "model.ubj"))
    with (path / "preprocessor.pkl").open("wb") as target:
        pickle.dump(prep, target)
    np.savez_compressed(path / "predictions.npz", indices=test.index.to_numpy(), probability=probability,
                        labels=test.TARGET.to_numpy(), months=test.LNMON.to_numpy(dtype=str))
    result = {"method": key, "label": variant_label(method, mode), "base_method": method,
        "calendar_mode": mode, "include_month": mode == "month_index", "phase": phase,
        "params": params, "cv": cv, "test": scores, "test_by_month": monthly,
        "preprocessing": prep.description(), "fit_seconds": seconds, "signature": signature,
        "reused": False, "checkpoint_path": path.relative_to(ROOT).as_posix()}
    write_json(result_path, result)
    print(f"{phase} {key}: weighted={scores['weighted_score']:.6f}, error={scores['misclassification_rate']:.6f}", flush=True)
    del model, x_train, x_test
    gc.collect()
    return result


def export_calendar_tables(report, output):
    export_tables(report, output)
    for name in ("comparison", "fold_metrics", "monthly_metrics"):
        table = pd.read_csv(output / f"{name}.csv")
        attributes = pd.DataFrame([{name: r[name] for name in (
            "method", "phase", "base_method", "calendar_mode", "include_month", "reused")} for r in report["results"]])
        table.merge(attributes, on=["method", "phase"], validate="many_to_one").to_csv(
            output / f"{name}.csv", index=False, encoding="utf-8-sig")
    pairs = []
    for method in report["protocol"]["methods"]:
        records = {r["calendar_mode"]: r for r in report["results"]
                   if r["phase"] == "screening" and r["base_method"] == method}
        no, yes = records["no_month"], records["month_index"]
        pairs.append({"base_method": method, "label": BASE_METHODS[method],
            "weighted_no_month": no["test"]["weighted_score"], "weighted_with_month": yes["test"]["weighted_score"],
            "weighted_delta": yes["test"]["weighted_score"] - no["test"]["weighted_score"],
            "cv_delta": yes["cv"]["pooled"]["misclassification_rate"] - no["cv"]["pooled"]["misclassification_rate"],
            "test_error_delta": yes["test"]["misclassification_rate"] - no["test"]["misclassification_rate"],
            "test_auc_delta": yes["test"]["roc_auc"] - no["test"]["roc_auc"]})
    pd.DataFrame(pairs).to_csv(output / "paired_comparison.csv", index=False, encoding="utf-8-sig")
    changes = []
    for result in report["results"]:
        if result["base_method"] != "cardinality5":
            continue
        for window, prep in [(f["name"], f["preprocessing"]) for f in result["cv"]["folds"]] + [("final_fit", result["preprocessing"])]:
            for variable in prep["converted_by_cardinality"]:
                changes.append({"method": result["method"], "phase": result["phase"], "window": window,
                    "variable": variable, "training_unique_values": prep["training_cardinality"][variable],
                    "original_type": "categorical", "new_type": "numeric"})
    pd.DataFrame(changes).to_csv(output / "type_changes.csv", index=False, encoding="utf-8-sig")


@hydra.main(version_base="1.3", config_path="conf", config_name="calendar_experiment")
def main(cfg: DictConfig):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    if list(cfg.methods) != list(BASE_METHODS) or list(cfg.calendar_modes) != list(CALENDAR_MODES):
        raise ValueError("This comparison requires the specified 3 x 2 conditions")
    if not 0 < int(cfg.tuning.startup_trials) < int(cfg.tuning.trials) or not 0 < int(cfg.tuning.finalists) <= 6:
        raise ValueError("Invalid small-TPE budget")
    output = repo_path(str(cfg.paths.output_dir))
    output.mkdir(parents=True, exist_ok=True)
    data_path, type_path = repo_path(str(cfg.paths.data)), repo_path(str(cfg.paths.feature_types))
    config = OmegaConf.to_container(cfg, resolve=True)
    code_paths = [Path(__file__), Path(__file__).with_name("calendar_preprocessing.py"),
                  Path(__file__).with_name("preprocessing_experiment.py"), Path(__file__).with_name("train.py")]
    prior = repo_path(str(cfg.reuse.output_dir))
    signature_info = {"config": config, "data_sha256": file_hash(data_path), "types_sha256": file_hash(type_path),
        "code_sha256": {p.name: file_hash(p) for p in code_paths}, "python": platform.python_version(),
        "xgboost": xgb.__version__, "sklearn": sklearn.__version__, "optuna": optuna.__version__,
        "reuse_sha256": {name: file_hash(prior / name) for name in ("metrics.json", "run_config.yaml")
                         if cfg.reuse.enabled and (prior / name).exists()}}
    signature = hashlib.sha256(json.dumps(signature_info, sort_keys=True).encode()).hexdigest()
    signature_path = output / "checkpoints" / "run_signature.json"
    if signature_path.exists() and json.loads(signature_path.read_text())["signature"] != signature:
        raise ValueError("Experiment configuration changed; choose a new output directory")
    write_json(signature_path, {"signature": signature, **signature_info})
    OmegaConf.save(cfg, output / "run_config.yaml")
    reuse = load_reuse(cfg, signature_info)
    frame, schema = read_data(data_path, type_path)
    if set(frame.LNMON) != set(pd.date_range("2023-06-01", "2024-05-01", freq="MS").strftime("%Y%m")):
        raise ValueError("Labeled source must contain 202306 through 202405")
    variants = [(m, c) for m in cfg.methods for c in cfg.calendar_modes]
    baseline_params = OmegaConf.to_container(cfg.model.baseline, resolve=True)
    baselines = {}
    for method, mode in variants:
        key = variant_key(method, mode)
        print(f"Screening: {variant_label(method, mode)}", flush=True)
        old = source_record(reuse, method, mode, "screening")
        if old is not None:
            if old["params"] != baseline_params:
                raise ValueError("Prior baseline hyperparameters changed")
            cv = copy.deepcopy(old["cv"])
            print(f"  Reuse validated prior CV: {cv['pooled']['misclassification_rate']:.6f}", flush=True)
        else:
            prepared = prepare_folds(frame, schema, method, mode, cfg)
            cv = evaluate_cv(prepared, baseline_params, cfg, output / "checkpoints" / key / "screening", signature)
            del prepared
            gc.collect()
        baselines[key] = cv
        write_json(output / "checkpoints" / "screening_progress.json", baselines)
    finalists = sorted(baselines, key=lambda k: (
        baselines[k]["pooled"]["misclassification_rate"], baselines[k]["pooled"]["log_loss"]))[:int(cfg.tuning.finalists)]
    print(f"TPE finalists selected using CV only: {finalists}", flush=True)
    tuned = {}
    for key in finalists:
        method, mode = key.split("__")
        tuned[key] = tune_variant(frame, schema, method, mode, cfg, output, signature, baselines[key], reuse)
    write_json(output / "checkpoints" / "frozen_choices.json", {"baselines": baselines, "tuned": tuned})
    results = [evaluate_holdout(frame, schema, m, c, "screening", baseline_params,
                               baselines[variant_key(m, c)], cfg, output, signature, reuse) for m, c in variants]
    for key, choice in tuned.items():
        method, mode = key.split("__")
        results.append(evaluate_holdout(frame, schema, method, mode, "tuned", choice["params"],
                                       choice["cv"], cfg, output, signature, reuse))
    selected = min(results, key=lambda r: r["cv"]["pooled"]["misclassification_rate"])
    report = {"protocol": {
        "methods": list(cfg.methods), "calendar_modes": list(cfg.calendar_modes), "folds": config["folds"],
        "test_train_end": str(cfg.test.train_end), "test_weights": config["test"]["weights"],
        "weighted_score_formula": "sum(row_weight * error) / total_test_rows",
        "calendar_encoding": "202306=1,...,202411=18; fixed origin; numerical input only when enabled",
        "cardinality5": "Existing integer/spacing type rules are retained; numeric-coded categories with >5 train-only distinct numeric levels become numeric. Text categories and boolean roles are unchanged.",
        "threshold_selection": "One threshold minimizing pooled F7-F9 OOF errors; frozen before F10-F12",
        "rounds_selection": "Median best rounds from F7-F9; no test early stopping",
        "selection": "Six common-parameter conditions first; only CV top two receive 12-trial TPE",
    }, "tuning_finalists": finalists, "tuning": tuned,
        "selected_by_cv": {"method": selected["method"], "phase": selected["phase"]},
        "results": results, "environment": signature_info,
        "limitations": "One seed and historically inspected evaluation months; exploratory comparison. Shared unsupervised type map predates this experiment. Actual June-November 2024 labeled data is unavailable. Tree splits do not linearly extrapolate an increasing calendar index."}
    write_json(output / "metrics.json", report)
    export_calendar_tables(report, output)
    from XGBoost.visualize_calendar import create_visualizations
    create_visualizations(output)
    print(f"Completed 3 x 2 conditions plus {len(tuned)} TPE finalists: {output}", flush=True)


if __name__ == "__main__":
    main()

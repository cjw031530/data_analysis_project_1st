"""Time ordered XGBoost tuning, holdout evaluation, and final training.

Run from any directory: python XGBoost/train.py
All learned preprocessing rules are fitted on the training rows of each split.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path
from types import SimpleNamespace

import hydra
import numpy as np
import optuna
import pandas as pd
import sklearn
import xgboost as xgb
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
MISSING = ["", "NA", "N/A", "NAN", "NULL", "NONE", "<NA>"]
MISSING_CATEGORY = "__KCB_MISSING__"
BOOLEAN_VALUES = {"0": 0.0, "1": 1.0, "0.0": 0.0, "1.0": 1.0, "N": 0.0, "Y": 1.0}
FOLDS = [
    ("202309", "202310"),
    ("202310", "202311"),
    ("202311", "202312"),
    ("202312", "202401"),
]
def month_index(month: str) -> int:
    text = str(month)
    if len(text) != 6 or not text.isdecimal() or not 1 <= int(text[4:]) <= 12:
        raise ValueError(f"Invalid LNMON: {month!r}")
    return 12 * (int(text[:4]) - 2023) + int(text[4:]) - 5


def repo_path(value: str) -> Path:
    """Interpret configured relative paths from the repository root."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def normalize_text(series: pd.Series) -> pd.Series:
    stripped = series.str.strip()
    return stripped.mask(stripped.str.upper().isin(MISSING))


def read_type_map(path: Path, header: list[str]) -> dict[str, list[str]]:
    table = pd.read_csv(path, dtype="string", keep_default_na=False)
    if table.columns.tolist() != ["variable", "type"]:
        raise ValueError("feature_types.csv must have variable,type columns")
    if table.variable.duplicated().any() or len(header) != len(set(header)):
        raise ValueError("Duplicate variable names in data or type map")
    if set(table.variable) != set(header):
        raise ValueError(f"Type map mismatch: missing={set(header)-set(table.variable)}, extra={set(table.variable)-set(header)}")
    allowed = {"numeric", "categorical", "boolean", "month_index", "target"}
    if not set(table.type).issubset(allowed):
        raise ValueError(f"Unexpected types: {set(table.type)-allowed}")
    types = dict(zip(table.variable, table.type))
    if types.get("LNMON") != "month_index" or types.get("TARGET") != "target":
        raise ValueError("LNMON and TARGET must have their declared roles")
    return {kind: [name for name in header if types[name] == kind] for kind in allowed}


def read_data(path: Path, type_path: Path, require_target: bool = True) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        header = next(csv.reader(source))
    expected = pd.read_csv(type_path, nrows=0).columns.tolist()
    if expected != ["variable", "type"]:
        raise ValueError("Invalid feature type file")
    # Prediction files may omit TARGET; every predictor must still be present.
    full_header = header if "TARGET" in header else header + ["TARGET"]
    schema = read_type_map(type_path, full_header)
    if require_target and "TARGET" not in header:
        raise ValueError("Training data must include TARGET")
    dtypes = {name: "float32" for name in schema["numeric"]}
    dtypes.update({name: "string" for name in schema["categorical"] + schema["boolean"]})
    dtypes["LNMON"] = "string"
    if "TARGET" in header:
        dtypes["TARGET"] = "string"
    frame = pd.read_csv(
        path, dtype=dtypes, keep_default_na=False, na_values=MISSING,
        encoding="utf-8-sig", low_memory=False,
    )
    frame["LNMON"] = frame.LNMON.str.strip()
    frame["month_index"] = frame.LNMON.map(month_index).astype("float32")
    if frame.LNMON.isna().any():
        raise ValueError("LNMON contains missing values")
    if "TARGET" in frame:
        if not frame.TARGET.isin(["0", "1"]).all():
            raise ValueError("TARGET must contain only 0 and 1")
        frame["TARGET"] = frame.TARGET.astype("int8")
    return frame, schema


def fit_preprocessor(train: pd.DataFrame, schema: dict[str, list[str]], include_month: bool) -> dict:
    categories = {}
    for name in schema["categorical"]:
        raw = normalize_text(train[name])
        if raw.eq(MISSING_CATEGORY).any():
            raise ValueError(f"Reserved missing token occurs in {name}")
        categories[name] = sorted(raw.dropna().unique().tolist())
        if raw.isna().any():
            categories[name].append(MISSING_CATEGORY)
    predictors = set(schema["numeric"] + schema["categorical"] + schema["boolean"])
    order = [c for c in train.columns if c in predictors]
    # Use the order in the original CSV, then append the optional month index.
    if include_month:
        order.append("month_index")
    return {"numeric": schema["numeric"], "categorical": schema["categorical"],
            "boolean": schema["boolean"], "categories": categories,
            "column_order": order, "include_month": include_month}


def transform(frame: pd.DataFrame, spec: dict) -> pd.DataFrame:
    columns = {}
    for name in spec["column_order"]:
        if name in spec["numeric"] or name == "month_index":
            columns[name] = frame[name].astype("float32")
        elif name in spec["boolean"]:
            raw = normalize_text(frame[name]).str.upper()
            unknown = raw.notna() & ~raw.isin(BOOLEAN_VALUES)
            if unknown.any():
                raise ValueError(f"Unexpected boolean values in {name}: {raw[unknown].unique().tolist()[:5]}")
            columns[name] = raw.map(BOOLEAN_VALUES).astype("float32")
        else:
            raw = normalize_text(frame[name]).fillna(MISSING_CATEGORY)
            dtype = pd.CategoricalDtype(categories=spec["categories"][name])
            columns[name] = raw.astype(dtype)
    out = pd.DataFrame(columns, index=frame.index)
    if out.columns.tolist() != spec["column_order"]:
        raise AssertionError("Feature order changed")
    return out


def best_threshold(labels: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    """Find the exact threshold minimizing ordinary 0/1 errors."""
    order = np.argsort(-probabilities, kind="mergesort")
    y = labels[order].astype(np.int8)
    p = probabilities[order]
    errors = int(y.sum())  # Predict zero for all rows.
    best = (errors, 0.0, 1.0 + np.finfo(float).eps)
    ends = np.r_[np.flatnonzero(np.diff(p)), len(p) - 1]
    previous = -1
    for end in ends:
        group = y[previous + 1:end + 1]
        errors += len(group) - 2 * int(group.sum())
        threshold = float(p[end])
        candidate = (errors, abs(threshold - 0.5), threshold)
        if candidate < best:
            best = candidate
        previous = end
    return best[2], best[0] / len(y)


def metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    pred = (probabilities >= threshold).astype(np.int8)
    tp = int(((labels == 1) & (pred == 1)).sum())
    fp = int(((labels == 0) & (pred == 1)).sum())
    fn = int(((labels == 1) & (pred == 0)).sum())
    tn = int(((labels == 0) & (pred == 0)).sum())
    return {
        "rows": len(labels), "positive_rate": float(labels.mean()),
        "threshold": float(threshold), "misclassification_rate": float((pred != labels).mean()),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
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


def make_model(params: dict, rounds: int, jobs: int, seed: int, early_stopping: int | None = None):
    return xgb.XGBClassifier(
        objective="binary:logistic", tree_method="hist", device="cpu",
        enable_categorical=True, n_estimators=rounds, n_jobs=jobs,
        random_state=seed, eval_metric="logloss",
        early_stopping_rounds=early_stopping, **params,
    )


def suggest_params(trial: optuna.Trial, search: dict) -> dict:
    params = {}
    for name, rule in search.items():
        if "choices" in rule:
            params[name] = trial.suggest_categorical(name, rule["choices"])
        elif name == "max_depth":
            params[name] = trial.suggest_int(name, **rule)
        else:
            params[name] = trial.suggest_float(name, **rule)
    return params


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def write_json_atomic(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    write_json(temporary, value)
    os.replace(temporary, path)


def checkpointed_trial(prepared: list, params: dict, number: int, args,
                       checkpoint_root: Path, config_sha256: str, progress: tqdm,
                       expected_error: float | None = None) -> tuple[float, dict]:
    """Keep every completed fold model so an interrupted trial can resume."""
    trial_dir = checkpoint_root / f"trial_{number:05d}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = trial_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("params") != params or manifest.get("config_sha256") != config_sha256:
            raise ValueError(f"Checkpoint does not match the current trial: {trial_dir}")
    else:
        manifest = {"trial_number": number, "params": params,
                    "config_sha256": config_sha256, "folds": {}}
        write_json_atomic(manifest_path, manifest)

    labels, scores, fold_records, best_rounds = [], [], [], []
    for fold_number, (x_train, y_train, x_valid, y_valid, month) in enumerate(prepared, start=1):
        progress.set_postfix_str(f"trial={number + 1} fold={fold_number}/{len(prepared)}")
        key = str(fold_number)
        model_path = trial_dir / f"fold_{fold_number:02d}.ubj"
        saved = manifest["folds"].get(key)
        if saved is not None:
            if saved["month"] != month or not model_path.is_file():
                raise ValueError(f"Incomplete checkpoint metadata: {model_path}")
            with model_path.open("rb") as source:
                digest = hashlib.file_digest(source, "sha256").hexdigest()
            if digest != saved["model_sha256"]:
                raise ValueError(f"Checkpoint hash mismatch: {model_path}")
            model = xgb.XGBClassifier()
            model.load_model(str(model_path))
            best_round = int(saved["best_round"])
        else:
            model = make_model(params, args.rounds, args.jobs, args.seed, args.early_stopping)
            model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=False)
            best_round = int(model.best_iteration) + 1
            temporary = model_path.with_name(model_path.stem + ".tmp.ubj")
            model.save_model(str(temporary))
            os.replace(temporary, model_path)
            with model_path.open("rb") as source:
                digest = hashlib.file_digest(source, "sha256").hexdigest()
            manifest["folds"][key] = {"month": month, "best_round": best_round,
                                      "model_sha256": digest}
            write_json_atomic(manifest_path, manifest)

        probability = model.predict_proba(x_valid, iteration_range=(0, best_round))[:, 1]
        labels.append(y_valid)
        scores.append(probability)
        best_rounds.append(best_round)
        fold_records.append({"month": month, "best_round": best_round,
                             "log_loss": float(log_loss(y_valid, probability))})

    labels_all = np.concatenate(labels)
    scores_all = np.concatenate(scores)
    threshold, error = best_threshold(labels_all, scores_all)
    if expected_error is not None and not np.isclose(error, expected_error, rtol=0, atol=1e-6):
        raise ValueError(f"Replayed trial {number} differs from its stored Optuna score")
    attrs = {"threshold": threshold, "best_rounds": best_rounds, "folds": fold_records,
             "oof_ap": float(average_precision_score(labels_all, scores_all))}
    manifest["result"] = {"error": error, **attrs}
    write_json_atomic(manifest_path, manifest)
    return error, attrs


def has_complete_checkpoint(root: Path, number: int, fold_count: int) -> bool:
    trial_dir = root / f"trial_{number:05d}"
    manifest_path = trial_dir / "manifest.json"
    if not manifest_path.is_file():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return ("result" in manifest and len(manifest.get("folds", {})) == fold_count
            and all((trial_dir / f"fold_{fold:02d}.ubj").is_file()
                    for fold in range(1, fold_count + 1)))


def tune(frame: pd.DataFrame, schema: dict, include_month: bool, args, output: Path) -> tuple[dict, int, dict]:
    variant = "month_index" if include_month else "no_month"
    folds = FOLDS[-args.folds:]
    prepared = []
    for train_end, valid_month in tqdm(folds, desc=f"Prepare {variant}", unit="fold", leave=False):
        train = frame.loc[frame.LNMON.between("202306", train_end)]
        valid = frame.loc[frame.LNMON.eq(valid_month)]
        spec = fit_preprocessor(train, schema, include_month)
        prepared.append((transform(train, spec), train.TARGET.to_numpy(),
                         transform(valid, spec), valid.TARGET.to_numpy(), valid_month))
    config = {"variant": variant, "folds": [list(pair) for pair in folds], "rounds": args.rounds,
              "early_stopping": args.early_stopping, "seed": args.seed,
              "startup_trials": args.startup_trials,
              "data_sha256": args.data_sha256, "types_sha256": args.types_sha256,
              "xgboost": xgb.__version__, "optuna": optuna.__version__,
              "sklearn": sklearn.__version__, "pandas": pd.__version__,
              "train_sha256": args.train_sha256, "jobs": args.jobs,
              "search": args.search, "baseline": args.baseline}
    # The code hash changes when checkpoint support is added. All settings that
    # affect model results still have to match the existing study exactly.
    comparable_config = {key: value for key, value in config.items() if key != "train_sha256"}
    config_sha256 = hashlib.sha256(json.dumps(comparable_config, sort_keys=True).encode()).hexdigest()
    checkpoint_root = output / "checkpoints" / variant
    study = optuna.create_study(
        direction="minimize", study_name=variant,
        storage=f"sqlite:///{(output / (variant + '.sqlite3')).as_posix()}",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=args.seed, n_startup_trials=args.startup_trials),
    )
    stored_config = study.user_attrs.get("config")
    if stored_config is not None and {key: value for key, value in stored_config.items()
                                      if key != "train_sha256"} != comparable_config:
        raise ValueError(f"Existing Optuna study has a different configuration: {variant}")
    if stored_config != config:
        study.set_user_attr("config", config)
    if not study.trials:
        study.enqueue_trial(args.baseline)
    # study.optimize() marked interrupted trials as FAIL in older runs. Repeat
    # their exact parameter sets before sampling anything new.
    retried = set(study.user_attrs.get("retried_failed_trials", []))
    for frozen in study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.FAIL,)):
        if frozen.number not in retried:
            if set(frozen.params) != set(args.search):
                raise ValueError(f"Failed trial {frozen.number} lacks parameters needed for recovery")
            study.enqueue_trial(frozen.params, user_attrs={"retry_of": frozen.number})
            retried.add(frozen.number)
            study.set_user_attr("retried_failed_trials", sorted(retried))

    completed = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    with tqdm(total=args.trials, initial=min(completed, args.trials),
              desc=f"Optuna {variant}", unit="trial", dynamic_ncols=True) as progress:
        if completed:
            progress.set_postfix_str(f"best_error={study.best_value:.5f}")
        # Older studies have metrics but no model files. Refit those trials once
        # so every completed trial has the same recoverable model artifacts.
        for frozen in study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,)):
            if not has_complete_checkpoint(checkpoint_root, frozen.number, len(prepared)):
                print(f"Saving models for completed {variant} trial {frozen.number + 1}", flush=True)
                checkpointed_trial(prepared, frozen.params, frozen.number, args,
                                   checkpoint_root, config_sha256, progress,
                                   expected_error=frozen.value)

        while completed < args.trials:
            running = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.RUNNING,))
            if len(running) > 1:
                raise ValueError(f"Multiple unfinished trials in {variant}; run only one trainer per output directory")
            trial = optuna.trial.Trial(study, running[0]._trial_id) if running else study.ask()
            params = suggest_params(trial, args.search)
            error, attrs = checkpointed_trial(prepared, params, trial.number, args,
                                               checkpoint_root, config_sha256, progress)
            for key, value in attrs.items():
                trial.set_user_attr(key, value)
            study.tell(trial, error)
            completed += 1
            progress.update(1)
            progress.set_postfix_str(f"best_error={study.best_value:.5f}")
    best = study.best_trial
    rounds = max(1, int(np.median(best.user_attrs["best_rounds"])))
    return best.params, rounds, {"cv_error": best.value, "cv_threshold": best.user_attrs["threshold"],
                                 "best_rounds": best.user_attrs["best_rounds"],
                                 "oof_ap": best.user_attrs["oof_ap"],
                                 "trials": len(study.trials)}


def configured_args(cfg: DictConfig) -> SimpleNamespace:
    jobs = int(cfg.cpu.jobs)
    if jobs == 0:
        jobs = os.cpu_count() or 1
    args = SimpleNamespace(
        data=repo_path(str(cfg.paths.data)),
        types=repo_path(str(cfg.paths.feature_types)),
        output_dir=repo_path(str(cfg.paths.output_dir)),
        trials=int(cfg.tuning.trials),
        startup_trials=int(cfg.tuning.startup_trials),
        folds=int(cfg.tuning.folds),
        rounds=int(cfg.tuning.rounds),
        early_stopping=int(cfg.tuning.early_stopping),
        jobs=jobs,
        seed=int(cfg.seed),
        variants=list(cfg.variants),
        search=OmegaConf.to_container(cfg.tuning.search, resolve=True),
        baseline=OmegaConf.to_container(cfg.tuning.baseline, resolve=True),
        hydra_config=OmegaConf.to_container(cfg, resolve=True),
    )
    if min(args.trials, args.startup_trials, args.rounds, args.early_stopping, args.jobs) < 1:
        raise ValueError("trials, startup_trials, rounds, early_stopping, and jobs must be positive")
    if not 1 <= args.folds <= len(FOLDS):
        raise ValueError(f"tuning.folds must be between 1 and {len(FOLDS)}")
    if not args.variants or len(args.variants) != len(set(args.variants)) or not set(args.variants) <= {"no_month", "month_index"}:
        raise ValueError("variants must contain no_month and/or month_index without duplicates")
    if set(args.baseline) != set(args.search):
        raise ValueError("tuning.baseline and tuning.search must have the same parameter names")
    for name, rule in args.search.items():
        value = args.baseline[name]
        if "choices" in rule:
            if value not in rule["choices"]:
                raise ValueError(f"Baseline {name} is not among its search choices")
        elif not rule["low"] <= value <= rule["high"]:
            raise ValueError(f"Baseline {name} is outside its search interval")
    return args


@hydra.main(version_base="1.3", config_path="conf", config_name="train")
def main(cfg: DictConfig) -> None:
    args = configured_args(cfg)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.data.open("rb") as source:
        args.data_sha256 = hashlib.file_digest(source, "sha256").hexdigest()
    with args.types.open("rb") as source:
        args.types_sha256 = hashlib.file_digest(source, "sha256").hexdigest()
    args.train_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    frame, schema = read_data(args.data, args.types)
    if sorted(frame.LNMON.unique().tolist()) != [f"{2023 + (5 + i) // 12:04d}{(5 + i) % 12 + 1:02d}" for i in range(12)]:
        raise ValueError("Training CSV must contain exactly 202306 through 202405")
    candidates = {}
    for variant in args.variants:
        include_month = variant == "month_index"
        print(f"Tuning {variant}: {args.trials} total trials, {args.folds} time folds, {args.jobs} CPU threads", flush=True)
        params, cv_rounds, cv = tune(frame, schema, include_month, args, args.output_dir)
        train = frame.loc[frame.LNMON.between("202306", "202401")]
        valid = frame.loc[frame.LNMON.between("202402", "202403")]
        print(f"Validating {variant} on 202402-202403", flush=True)
        spec = fit_preprocessor(train, schema, include_month)
        x_valid = transform(valid, spec)
        y_valid = valid.TARGET.to_numpy()
        model = make_model(params, args.rounds, args.jobs, args.seed, args.early_stopping)
        model.fit(transform(train, spec), train.TARGET.to_numpy(),
                  eval_set=[(x_valid, y_valid)], verbose=False)
        rounds = int(model.best_iteration) + 1
        probability = model.predict_proba(x_valid)[:, 1]
        threshold, _ = best_threshold(valid.TARGET.to_numpy(), probability)
        result = metrics(valid.TARGET.to_numpy(), probability, threshold)
        candidates[variant] = {"params": params, "rounds": rounds, "cv_rounds_median": cv_rounds, "cv": cv,
                               "validation": result, "validation_by_month": monthly_metrics(valid, probability, threshold),
                               "threshold": threshold,
                               "model": model, "spec": spec}
        print(f"{variant}: CV error={cv['cv_error']:.5f}, validation error={result['misclassification_rate']:.5f}", flush=True)
    chosen = min(candidates, key=lambda name: (
        candidates[name]["validation"]["misclassification_rate"],
        -candidates[name]["validation"]["average_precision"],
    ))
    selected = candidates[chosen]
    test = frame.loc[frame.LNMON.between("202404", "202405")]
    print(f"Testing {chosen} on 202404-202405", flush=True)
    test_probability = selected["model"].predict_proba(transform(test, selected["spec"]))[:, 1]
    test_result = metrics(test.TARGET.to_numpy(), test_probability, selected["threshold"])
    print(f"Chosen={chosen}; untouched April-May test error={test_result['misclassification_rate']:.5f}", flush=True)
    # The held-out test above evaluates the model trained only through January.
    # Refit on all labeled months for the undisclosed June-November inputs.
    print(f"Refitting {chosen} on all labeled months through 202405", flush=True)
    final_spec = fit_preprocessor(frame, schema, chosen == "month_index")
    final_model = make_model(selected["params"], selected["rounds"], args.jobs, args.seed)
    final_model.fit(transform(frame, final_spec), frame.TARGET.to_numpy(), verbose=False)
    final_model.save_model(str(args.output_dir / "model.json"))
    write_json(args.output_dir / "feature_schema.json", final_spec)
    write_json(args.output_dir / "decision.json", {
        "variant": chosen, "threshold": selected["threshold"],
        "params": selected["params"], "rounds": selected["rounds"],
    })
    report = {
        "split": {"tune": "202306-202401", "validation": "202402-202403",
                  "test": "202404-202405", "final_fit": "202306-202405"},
        "run_config": {"trials_per_variant": args.trials,
                       "random_startup_trials": args.startup_trials,
                       "folds": [list(pair) for pair in FOLDS[-args.folds:]],
                       "max_rounds": args.rounds, "early_stopping_rounds": args.early_stopping,
                       "seed": args.seed, "cpu_threads": args.jobs,
                       "hydra": args.hydra_config},
        "candidates": {name: {k: v for k, v in record.items() if k not in {"model", "spec"}}
                       for name, record in candidates.items()},
        "chosen": chosen, "test_before_refit": test_result,
        "test_by_month": monthly_metrics(test, test_probability, selected["threshold"]),
        "note": "Test metrics are for the selected pre-refit model; the final model uses all months through 202405.",
        "environment": {"python": platform.python_version(), "xgboost": xgb.__version__,
                        "optuna": optuna.__version__, "hydra": hydra.__version__,
                        "sklearn": sklearn.__version__,
                        "pandas": pd.__version__, "cpu_threads": args.jobs,
                        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                                     capture_output=True, text=True, check=False).stdout.strip(),
                        "data_sha256": args.data_sha256, "types_sha256": args.types_sha256,
                        "train_sha256": args.train_sha256,
                        "evaluate_sha256": hashlib.sha256((Path(__file__).parent / "evaluate.py").read_bytes()).hexdigest()},
    }
    write_json(args.output_dir / "metrics.json", report)


if __name__ == "__main__":
    main()

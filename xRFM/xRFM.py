"""Colab/GPU-ready xRFM baseline with leakage-safe rolling-origin evaluation.

The official ``xrfm`` package requires validation data in ``fit``.  For every
rolling training window, this script takes a stratified validation subset only
from that historical training window.  The OOF month and final test months are
never passed to ``fit``.

The default command runs F7/F8/F9 OOF only. In Google Colab:
    !pip install -q "xrfm[cu12]==0.4.5" pandas scikit-learn threadpoolctl
    !python xRFM.py

Run the untouched final test only after all choices are fixed:
    py xRFM.py --data kcb_202306_202405_undersampled_1to4.csv --run-final-test

xRFM 0.4.x requires Python 3.11 or newer.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from threadpoolctl import threadpool_limits
except ImportError:  # pragma: no cover
    threadpool_limits = None


SEED = 2026
DATA_FILENAME = "kcb_202306_202405_undersampled_1to4.csv"
COLAB_PROJECT_DIR = Path("/content/drive/MyDrive/kcb_project")
DEV_MONTHS = tuple(range(202306, 202313)) + (202401, 202402)
TEST_MONTHS = (202403, 202404, 202405)
ROLLING_FOLDS = (
    ("F7", tuple(range(202306, 202312)), 202312),
    ("F8", tuple(range(202306, 202313)), 202401),
    ("F9", tuple(range(202306, 202313)) + (202401,), 202402),
)
TEST_WEIGHTS = {202403: 0.1, 202404: 0.3, 202405: 0.6}


@dataclass(frozen=True)
class DatasetSpec:
    frame: pd.DataFrame
    feature_columns: list[str]
    date_column: str
    target_column: str


def default_data_path() -> Path:
    """Use the mounted Colab Drive dataset when available, else a local file."""
    data_dir = COLAB_PROJECT_DIR / "data"
    if data_dir.is_dir():
        preferred = (
            data_dir / DATA_FILENAME,
            data_dir / "kcb_202306_202405_undersampled_1to4(2).csv",
        )
        for path in preferred:
            if path.exists():
                return path
        matches = sorted(data_dir.glob("kcb_202306_202405_undersample*.csv"))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            choices = "\n".join(f"  - {path}" for path in matches)
            raise FileNotFoundError(
                "Multiple KCB CSV files were found. Pass the intended one with --data:\n"
                + choices
            )
    return Path(__file__).with_name(DATA_FILENAME)


def default_output_dir() -> Path:
    if COLAB_PROJECT_DIR.is_dir():
        return COLAB_PROJECT_DIR / "outputs_xrfm"
    return Path(__file__).with_name("outputs_xrfm")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a fixed GPU/CPU xRFM baseline.")
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--date-col", default="LNMON")
    parser.add_argument("--target-col", default="TARGET")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--n-jobs", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="auto uses CUDA when available and otherwise falls back to CPU.",
    )
    parser.add_argument(
        "--internal-validation-fraction",
        type=float,
        default=0.10,
        help="Historical training rows reserved for xRFM's required fit validation.",
    )
    parser.add_argument(
        "--time-limit-minutes",
        type=float,
        default=0.0,
        help="Per fit time limit. Zero means no limit.",
    )
    parser.add_argument("--run-final-test", action="store_true")
    args = parser.parse_args()
    if args.data is None:
        args.data = default_data_path()
    if args.output_dir is None:
        args.output_dir = default_output_dir()
    return args


def read_csv_robust(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found: {path}\n"
            f"Put {DATA_FILENAME} beside this script or pass --data PATH."
        )
    errors: list[str] = []
    for encoding in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False)
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")
    raise UnicodeError("Could not decode CSV. " + " | ".join(errors))


def resolve_column(df: pd.DataFrame, requested: str, candidates: tuple[str, ...], kind: str) -> str:
    if requested in df.columns:
        return requested
    lower_to_original = {str(column).lower(): str(column) for column in df.columns}
    if requested.lower() in lower_to_original:
        return lower_to_original[requested.lower()]
    matches = [lower_to_original[name.lower()] for name in candidates if name.lower() in lower_to_original]
    if len(matches) == 1:
        warnings.warn(f"Using detected {kind} column: {matches[0]!r}")
        return matches[0]
    preview = ", ".join(map(str, df.columns[:30]))
    option = "--date-col" if kind == "date" else "--target-col"
    raise ValueError(
        f"Could not identify the {kind} column. Use {option}. First columns: {preview}"
    )


def normalize_month(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip()
    digits = text.str.replace(r"[^0-9]", "", regex=True).str[:6]
    numeric = pd.to_numeric(digits, errors="coerce")
    result = numeric.where(numeric.between(190001, 210012))
    unresolved = result.isna()
    if unresolved.any():
        parsed = pd.to_datetime(series[unresolved], errors="coerce")
        result.loc[unresolved] = parsed.dt.year * 100 + parsed.dt.month
    if result.isna().any():
        raise ValueError(
            "Unparseable date values: " + repr(series[result.isna()].head(10).tolist())
        )
    result = result.astype(int)
    if not (result % 100).between(1, 12).all():
        raise ValueError("Date column contains invalid YYYYMM values.")
    return result


def normalize_binary_target(series: pd.Series) -> tuple[pd.Series, dict[str, int]]:
    if series.isna().any():
        raise ValueError("Target contains missing values.")
    unique = list(pd.unique(series))
    if len(unique) != 2:
        raise ValueError(f"Expected a binary target; got {unique!r}")
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all() and set(numeric.astype(int).unique()) == {0, 1}:
        return numeric.astype(np.int64), {"0": 0, "1": 1}
    ordered = sorted(unique, key=lambda value: str(value))
    mapping = {ordered[0]: 0, ordered[1]: 1}
    printable = {str(key): value for key, value in mapping.items()}
    warnings.warn(f"Target mapping: {printable}")
    return series.map(mapping).astype(np.int64), printable


def load_dataset(path: Path, date_col: str, target_col: str) -> tuple[DatasetSpec, dict[str, int]]:
    df = read_csv_robust(path)
    unnamed = [column for column in df.columns if str(column).startswith("Unnamed:")]
    if unnamed:
        df = df.drop(columns=unnamed)
    date_col = resolve_column(
        df, date_col, ("period", "month", "date", "base_ym", "ym", "기준년월"), "date"
    )
    target_col = resolve_column(
        df, target_col, ("target", "label", "y", "bad", "bad_flag", "default"), "target"
    )
    df = df.copy()
    df[date_col] = normalize_month(df[date_col])
    df[target_col], mapping = normalize_binary_target(df[target_col])
    required = set(DEV_MONTHS + TEST_MONTHS)
    df = df[df[date_col].isin(required)].copy()
    counts = df[date_col].value_counts().sort_index()
    missing = sorted(required.difference(counts.index))
    if missing:
        raise ValueError(f"Required months are missing: {missing}")
    features = [column for column in df.columns if column not in {date_col, target_col}]
    if not features:
        raise ValueError("No feature columns were found.")
    print(f"Loaded {len(df):,} rows, {len(features):,} features")
    print(counts.to_string())
    return DatasetSpec(df, features, date_col, target_col), mapping


def make_one_hot_encoder() -> OneHotEncoder:
    kwargs: dict[str, Any] = {"handle_unknown": "ignore", "dtype": np.float32}
    try:
        return OneHotEncoder(sparse_output=False, **kwargs)
    except TypeError:  # scikit-learn < 1.2
        return OneHotEncoder(sparse=False, **kwargs)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical = list(X.select_dtypes(include=["object", "category", "bool", "string"]).columns)
    numerical = [column for column in X.columns if column not in categorical]
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numerical:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numerical,
            )
        )
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", make_one_hot_encoder()),
                    ]
                ),
                categorical,
            )
        )
    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.0)


def split_internal_validation(
    X: np.ndarray, y: np.ndarray, fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not 0.01 <= fraction <= 0.40:
        raise ValueError("--internal-validation-fraction must be between 0.01 and 0.40")
    return train_test_split(
        X,
        y,
        test_size=fraction,
        random_state=seed,
        stratify=y,
    )


def resolve_device(requested: str):
    import torch

    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but PyTorch cannot see a GPU. "
            "In Colab select Runtime > Change runtime type > T4 GPU and restart."
        )
    return torch.device(requested)


def build_xrfm(
    seed: int,
    n_jobs: int,
    time_limit_minutes: float,
    device_name: str,
):
    if sys.version_info < (3, 11):
        raise RuntimeError("xRFM 0.4.x requires Python 3.11 or newer.")
    try:
        import torch
        from xrfm import xRFM
    except ImportError as exc:
        raise ImportError(
            'Install the official implementation with: py -m pip install "xrfm==0.4.5"'
        ) from exc
    torch.set_num_threads(n_jobs)
    device = torch.device(device_name)
    return xRFM(
        device=device,
        n_trees=1,
        n_tree_iters=0,
        tuning_metric="accuracy",
        use_temperature_tuning=False,
        split_temperature=None,
        n_threads=n_jobs,
        random_state=seed,
        time_limit_s=None if time_limit_minutes <= 0 else time_limit_minutes * 60.0,
        verbose=True,
    )


def predict_xrfm(model: Any, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(model.predict(X), dtype=np.int64).reshape(-1)
    proba = np.asarray(model.predict_proba(X), dtype=float)
    if proba.ndim == 2 and proba.shape[1] >= 2:
        score = proba[:, 1]
    else:
        score = proba.reshape(-1)
    return pred, score


def metrics_row(
    split: str,
    y_true: np.ndarray,
    pred: np.ndarray,
    score: np.ndarray,
    fit_seconds: float,
    predict_seconds: float,
) -> dict[str, Any]:
    error = float(np.mean(pred != y_true))
    try:
        auc: float | None = float(roc_auc_score(y_true, score))
    except ValueError:
        auc = None
    try:
        loss: float | None = float(
            log_loss(y_true, np.clip(score, 1e-7, 1 - 1e-7), labels=[0, 1])
        )
    except ValueError:
        loss = None
    return {
        "model": "xRFM",
        "split": split,
        "n": int(len(y_true)),
        "misclassification_rate": error,
        "accuracy": float(accuracy_score(y_true, pred)),
        "roc_auc": auc,
        "log_loss": loss,
        "fit_seconds": float(fit_seconds),
        "predict_seconds": float(predict_seconds),
    }


def save_results(
    output_dir: Path,
    metrics: list[dict[str, Any]],
    predictions: list[pd.DataFrame],
    metadata: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics).to_csv(output_dir / "metrics.csv", index=False)
    if predictions:
        pd.concat(predictions, ignore_index=True).to_csv(
            output_dir / "predictions.csv", index=False
        )
    (output_dir / "results.json").write_text(
        json.dumps({"metadata": metadata, "metrics": metrics}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def fit_and_score(
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    spec: DatasetSpec,
    split_name: str,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], pd.DataFrame]:
    X_train_df = train[spec.feature_columns]
    X_eval_df = evaluation[spec.feature_columns]
    y_train_all = train[spec.target_column].to_numpy(dtype=np.int64)
    y_eval = evaluation[spec.target_column].to_numpy(dtype=np.int64)
    preprocessor = build_preprocessor(X_train_df)
    prep_started = time.perf_counter()
    X_train_all = np.asarray(preprocessor.fit_transform(X_train_df), dtype=np.float32)
    X_eval = np.asarray(preprocessor.transform(X_eval_df), dtype=np.float32)
    print(
        f"Preprocessed train={X_train_all.shape}, eval={X_eval.shape}, "
        f"time={time.perf_counter() - prep_started:.1f}s"
    )
    X_fit, X_internal_val, y_fit, y_internal_val = split_internal_validation(
        X_train_all, y_train_all, args.internal_validation_fraction, args.seed
    )
    model = build_xrfm(
        args.seed,
        args.n_jobs,
        args.time_limit_minutes,
        args.resolved_device,
    )
    started = time.perf_counter()
    model.fit(X_fit, y_fit, X_internal_val, y_internal_val)
    fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    pred, score = predict_xrfm(model, X_eval)
    predict_seconds = time.perf_counter() - started
    row = metrics_row(split_name, y_eval, pred, score, fit_seconds, predict_seconds)
    table = pd.DataFrame(
        {
            "model": "xRFM",
            "split": split_name,
            "row_index": evaluation.index.to_numpy(),
            "month": evaluation[spec.date_column].to_numpy(),
            "y_true": y_eval,
            "score": score,
            "prediction": pred,
        }
    )
    return row, table


def run_experiment(args: argparse.Namespace) -> None:
    np.random.seed(args.seed)
    device = resolve_device(args.device)
    args.resolved_device = str(device)
    if device.type == "cuda":
        import torch

        print(f"Device: cuda ({torch.cuda.get_device_name(device)})")
    else:
        warnings.warn("CUDA is unavailable; xRFM will run on CPU.")
        print("Device: cpu")
    print(f"Data: {args.data.resolve()}")
    print(f"Output: {args.output_dir.resolve()}")
    spec, label_mapping = load_dataset(args.data.resolve(), args.date_col, args.target_col)
    try:
        xrfm_version = importlib.metadata.version("xrfm")
    except importlib.metadata.PackageNotFoundError:
        xrfm_version = "not installed at file creation/inspection time"
    metadata = {
        "data": str(args.data.resolve()),
        "date_column": spec.date_column,
        "target_column": spec.target_column,
        "label_mapping": label_mapping,
        "feature_count": len(spec.feature_columns),
        "seed": args.seed,
        "n_jobs": args.n_jobs,
        "device": str(device),
        "xrfm_version": xrfm_version,
        "xrfm_config": {
            "n_trees": 1,
            "n_tree_iters": 0,
            "tuning_metric": "accuracy",
            "temperature_tuning": False,
            "class_weight": None,
        },
        "internal_validation_fraction": args.internal_validation_fraction,
        "test_weights": {str(k): v for k, v in TEST_WEIGHTS.items()},
        "final_test_requested": bool(args.run_final_test),
    }
    metrics: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []

    for fold_name, train_months, valid_month in ROLLING_FOLDS:
        print(f"\n===== {fold_name}: {train_months[0]}..{train_months[-1]} -> {valid_month} =====")
        train = spec.frame[spec.frame[spec.date_column].isin(train_months)]
        valid = spec.frame[spec.frame[spec.date_column] == valid_month]
        row, table = fit_and_score(train, valid, spec, fold_name, args)
        metrics.append(row)
        predictions.append(table)
        save_results(args.output_dir, metrics, predictions, metadata)
        print(
            f"error={row['misclassification_rate']:.6f}, auc={row['roc_auc']}, "
            f"fit={row['fit_seconds']:.1f}s"
        )

    oof = pd.concat(predictions, ignore_index=True)
    y_oof = oof["y_true"].to_numpy(dtype=np.int64)
    pred_oof = oof["prediction"].to_numpy(dtype=np.int64)
    score_oof = oof["score"].to_numpy(dtype=float)
    fold_errors = np.array([row["misclassification_rate"] for row in metrics], dtype=float)
    summary = metrics_row("OOF_F7_F9", y_oof, pred_oof, score_oof, 0.0, 0.0)
    summary.update(
        {
            "fold_error_mean": float(fold_errors.mean()),
            "fold_error_std": float(fold_errors.std(ddof=1)),
            "fold_error_min": float(fold_errors.min()),
            "fold_error_max": float(fold_errors.max()),
            "fit_seconds": float(sum(row["fit_seconds"] for row in metrics)),
            "predict_seconds": float(sum(row["predict_seconds"] for row in metrics)),
        }
    )
    metrics.append(summary)
    save_results(args.output_dir, metrics, predictions, metadata)

    if not args.run_final_test:
        print("\nFinal test was NOT touched. Add --run-final-test only after choices are fixed.")
        print(f"OOF results: {args.output_dir.resolve()}")
        return

    print("\n===== FINAL TEST: 202306..202402 -> 202403..202405 =====")
    development = spec.frame[spec.frame[spec.date_column].isin(DEV_MONTHS)]
    test = spec.frame[spec.frame[spec.date_column].isin(TEST_MONTHS)]
    pooled, test_table = fit_and_score(development, test, spec, "FINAL_TEST", args)
    pooled["split"] = "FINAL_TEST_POOLED"
    metrics.append(pooled)
    predictions.append(test_table)
    monthly_errors: dict[int, float] = {}
    for month in TEST_MONTHS:
        month_table = test_table[test_table["month"] == month]
        row = metrics_row(
            f"FINAL_{month}",
            month_table["y_true"].to_numpy(dtype=np.int64),
            month_table["prediction"].to_numpy(dtype=np.int64),
            month_table["score"].to_numpy(dtype=float),
            0.0,
            0.0,
        )
        metrics.append(row)
        monthly_errors[month] = row["misclassification_rate"]
    weighted_error = float(sum(TEST_WEIGHTS[m] * monthly_errors[m] for m in TEST_MONTHS))
    metrics.append(
        {
            "model": "xRFM",
            "split": "FINAL_TEST_WEIGHTED_0.1_0.3_0.6",
            "n": int(len(test_table)),
            "misclassification_rate": weighted_error,
            "accuracy": 1.0 - weighted_error,
            "roc_auc": None,
            "log_loss": None,
            "fit_seconds": pooled["fit_seconds"],
            "predict_seconds": pooled["predict_seconds"],
        }
    )
    save_results(args.output_dir, metrics, predictions, metadata)
    print(
        f"pooled_error={pooled['misclassification_rate']:.6f}, "
        f"weighted_error={weighted_error:.6f}"
    )
    print(f"Results: {args.output_dir.resolve()}")


def main() -> None:
    args = parse_args()
    if args.n_jobs < 1:
        raise ValueError("--n-jobs must be at least 1")
    context = threadpool_limits(limits=args.n_jobs) if threadpool_limits else None
    if context is None:
        run_experiment(args)
    else:
        with context:
            run_experiment(args)


if __name__ == "__main__":
    main()

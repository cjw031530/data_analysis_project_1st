"""HistGradientBoosting baseline with leakage-safe rolling-origin OOF evaluation.

The default run evaluates only F7/F8/F9. The final 2024-03..2024-05 test
period is used only when --run-final-test is supplied explicitly.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from threadpoolctl import threadpool_limits
except ImportError:  # pragma: no cover
    threadpool_limits = None


MODEL_NAME = "HistGradientBoosting"
SEED = 2026
DATA_FILENAME = "kcb_202306_202405_undersampled_1to4.csv"
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


def parse_args() -> argparse.Namespace:
    model_dir = Path(__file__).resolve().parent
    repository_root = model_dir.parent
    parser = argparse.ArgumentParser(
        description="Run HistGradientBoosting on leakage-safe F7/F8/F9 rolling OOF folds."
    )
    parser.add_argument("--data", type=Path, default=repository_root / DATA_FILENAME)
    parser.add_argument("--date-col", default="LNMON")
    parser.add_argument("--target-col", default="TARGET")
    parser.add_argument("--output-dir", type=Path, default=model_dir / "outputs")
    parser.add_argument("--n-jobs", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--run-final-test",
        action="store_true",
        help="Refit on 2023-06..2024-02 and evaluate 2024-03..2024-05.",
    )
    return parser.parse_args()


def read_csv_robust(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found: {path}\n"
            f"Put {DATA_FILENAME} in the repository root or pass --data PATH."
        )
    errors: list[str] = []
    for encoding in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False)
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")
    raise UnicodeError("Could not decode CSV. " + " | ".join(errors))


def resolve_column(
    df: pd.DataFrame,
    requested: str,
    candidates: tuple[str, ...],
    kind: str,
) -> str:
    if requested in df.columns:
        return requested
    lower_to_original = {str(column).lower(): str(column) for column in df.columns}
    if requested.lower() in lower_to_original:
        return lower_to_original[requested.lower()]
    matches = [
        lower_to_original[name.lower()]
        for name in candidates
        if name.lower() in lower_to_original
    ]
    if len(matches) == 1:
        warnings.warn(f"Using detected {kind} column: {matches[0]!r}")
        return matches[0]
    flag = "date-col" if kind == "date" else "target-col"
    raise ValueError(
        f"Could not uniquely identify the {kind} column. "
        f"Requested={requested!r}. Pass --{flag} explicitly."
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
        bad = series[result.isna()].head(10).tolist()
        raise ValueError(f"Date values cannot be converted to YYYYMM: {bad}")
    result = result.astype(int)
    if not (result % 100).between(1, 12).all():
        raise ValueError("The date column contains invalid YYYYMM values.")
    return result


def normalize_binary_target(series: pd.Series) -> tuple[pd.Series, dict[str, int]]:
    if series.isna().any():
        raise ValueError(f"Target contains {int(series.isna().sum())} missing values.")
    unique = list(pd.unique(series))
    if len(unique) != 2:
        raise ValueError(f"Binary classification requires two target values; got {unique!r}")
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all() and set(numeric.astype(int).unique()) == {0, 1}:
        return numeric.astype(np.int64), {"0": 0, "1": 1}
    ordered = sorted(unique, key=lambda value: str(value))
    mapping = {ordered[0]: 0, ordered[1]: 1}
    printable = {str(key): value for key, value in mapping.items()}
    warnings.warn(f"Target mapped to binary labels: {printable}")
    return series.map(mapping).astype(np.int64), printable


def load_dataset(path: Path, date_col: str, target_col: str) -> tuple[DatasetSpec, dict[str, int]]:
    df = read_csv_robust(path)
    unnamed = [column for column in df.columns if str(column).startswith("Unnamed:")]
    if unnamed:
        df = df.drop(columns=unnamed)
    date_col = resolve_column(
        df, date_col, ("LNMON", "period", "month", "date", "base_ym", "ym"), "date"
    )
    target_col = resolve_column(
        df, target_col, ("TARGET", "target", "label", "y", "bad", "bad_flag"), "target"
    )
    df = df.copy()
    df[date_col] = normalize_month(df[date_col])
    df[target_col], label_mapping = normalize_binary_target(df[target_col])
    wanted_months = set(DEV_MONTHS + TEST_MONTHS)
    df = df[df[date_col].isin(wanted_months)].copy()
    counts = df[date_col].value_counts().sort_index()
    missing_months = sorted(wanted_months.difference(counts.index))
    if missing_months:
        raise ValueError(f"Required months are missing: {missing_months}")
    features = [column for column in df.columns if column not in {date_col, target_col}]
    if not features:
        raise ValueError("No feature columns remain after removing date and target.")
    print(f"Loaded {len(df):,} rows, {len(features):,} features")
    print(counts.to_string())
    return DatasetSpec(df, features, date_col, target_col), label_mapping


def make_one_hot_encoder() -> OneHotEncoder:
    kwargs: dict[str, Any] = {"handle_unknown": "ignore", "dtype": np.float32}
    try:
        return OneHotEncoder(sparse_output=False, **kwargs)
    except TypeError:  # scikit-learn < 1.2
        return OneHotEncoder(sparse=False, **kwargs)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical = list(
        X.select_dtypes(include=["object", "category", "bool", "string"]).columns
    )
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
    if not transformers:
        raise ValueError("No usable features were found.")
    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.0)


def build_model(seed: int, _n_jobs: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=31,
        l2_regularization=0.1,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=seed,
    )


def metric_row(
    split: str,
    y_true: np.ndarray,
    pred: np.ndarray,
    score: np.ndarray,
    fit_seconds: float,
    predict_seconds: float,
) -> dict[str, Any]:
    error = float(np.mean(pred != y_true))
    return {
        "model": MODEL_NAME,
        "split": split,
        "n": int(len(y_true)),
        "misclassification_rate": error,
        "accuracy": float(accuracy_score(y_true, pred)),
        "roc_auc": float(roc_auc_score(y_true, score)),
        "log_loss": float(log_loss(y_true, np.clip(score, 1e-7, 1 - 1e-7), labels=[0, 1])),
        "score_type": "probability",
        "fit_seconds": float(fit_seconds),
        "predict_seconds": float(predict_seconds),
    }


def fit_and_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    seed: int,
    n_jobs: int,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    model = build_model(seed, n_jobs)
    started = time.perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    score = np.asarray(model.predict_proba(X_eval))[:, 1].astype(float)
    pred = (score >= 0.5).astype(np.int64)
    predict_seconds = time.perf_counter() - started
    return pred, score, fit_seconds, predict_seconds


def save_results(
    output_dir: Path,
    metrics: list[dict[str, Any]],
    predictions: list[pd.DataFrame],
    metadata: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics).to_csv(output_dir / "metrics.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(
        output_dir / "predictions.csv", index=False
    )
    (output_dir / "results.json").write_text(
        json.dumps({"metadata": metadata, "metrics": metrics}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run_experiment(args: argparse.Namespace) -> None:
    np.random.seed(args.seed)
    spec, label_mapping = load_dataset(args.data.resolve(), args.date_col, args.target_col)
    frame = spec.frame
    metrics: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    metadata = {
        "data": str(args.data.resolve()),
        "date_column": spec.date_column,
        "target_column": spec.target_column,
        "label_mapping": label_mapping,
        "feature_count": len(spec.feature_columns),
        "seed": args.seed,
        "n_jobs": args.n_jobs,
        "threshold_rule": "predict_proba >= 0.5",
        "final_test_requested": bool(args.run_final_test),
        "test_weights": {str(key): value for key, value in TEST_WEIGHTS.items()},
    }

    for fold_name, train_months, valid_month in ROLLING_FOLDS:
        print(f"\n===== {fold_name}: {train_months[0]}..{train_months[-1]} -> {valid_month} =====")
        train = frame[frame[spec.date_column].isin(train_months)]
        valid = frame[frame[spec.date_column] == valid_month]
        X_train_df = train[spec.feature_columns]
        X_valid_df = valid[spec.feature_columns]
        y_train = train[spec.target_column].to_numpy(dtype=np.int64)
        y_valid = valid[spec.target_column].to_numpy(dtype=np.int64)

        preprocessor = build_preprocessor(X_train_df)
        X_train = np.asarray(preprocessor.fit_transform(X_train_df), dtype=np.float32)
        X_valid = np.asarray(preprocessor.transform(X_valid_df), dtype=np.float32)
        pred, score, fit_seconds, predict_seconds = fit_and_predict(
            X_train, y_train, X_valid, args.seed, args.n_jobs
        )
        row = metric_row(
            fold_name, y_valid, pred, score, fit_seconds, predict_seconds
        )
        metrics.append(row)
        predictions.append(
            pd.DataFrame(
                {
                    "model": MODEL_NAME,
                    "split": fold_name,
                    "row_index": valid.index.to_numpy(),
                    "month": valid_month,
                    "y_true": y_valid,
                    "score": score,
                    "prediction": pred,
                }
            )
        )
        save_results(args.output_dir, metrics, predictions, metadata)
        print(
            f"error={row['misclassification_rate']:.6f}, "
            f"auc={row['roc_auc']:.6f}, fit={fit_seconds:.1f}s"
        )

    oof = pd.concat(predictions, ignore_index=True)
    fold_rows = [row for row in metrics if row["split"] in {"F7", "F8", "F9"}]
    errors = np.array([row["misclassification_rate"] for row in fold_rows], dtype=float)
    summary = metric_row(
        "OOF_F7_F9",
        oof["y_true"].to_numpy(dtype=np.int64),
        oof["prediction"].to_numpy(dtype=np.int64),
        oof["score"].to_numpy(dtype=float),
        sum(row["fit_seconds"] for row in fold_rows),
        sum(row["predict_seconds"] for row in fold_rows),
    )
    summary.update(
        {
            "fold_error_mean": float(errors.mean()),
            "fold_error_std": float(errors.std(ddof=1)),
            "fold_error_min": float(errors.min()),
            "fold_error_max": float(errors.max()),
        }
    )
    metrics.append(summary)
    save_results(args.output_dir, metrics, predictions, metadata)

    if not args.run_final_test:
        print("\nFinal test was NOT touched.")
        print(f"OOF results: {args.output_dir.resolve()}")
        return

    print("\n===== FINAL TEST: 202306..202402 -> 202403..202405 =====")
    development = frame[frame[spec.date_column].isin(DEV_MONTHS)]
    test = frame[frame[spec.date_column].isin(TEST_MONTHS)]
    X_dev_df = development[spec.feature_columns]
    X_test_df = test[spec.feature_columns]
    y_dev = development[spec.target_column].to_numpy(dtype=np.int64)
    y_test = test[spec.target_column].to_numpy(dtype=np.int64)
    preprocessor = build_preprocessor(X_dev_df)
    X_dev = np.asarray(preprocessor.fit_transform(X_dev_df), dtype=np.float32)
    X_test = np.asarray(preprocessor.transform(X_test_df), dtype=np.float32)
    pred, score, fit_seconds, predict_seconds = fit_and_predict(
        X_dev, y_dev, X_test, args.seed, args.n_jobs
    )
    final_predictions = pd.DataFrame(
        {
            "model": MODEL_NAME,
            "split": "FINAL_TEST",
            "row_index": test.index.to_numpy(),
            "month": test[spec.date_column].to_numpy(),
            "y_true": y_test,
            "score": score,
            "prediction": pred,
        }
    )
    predictions.append(final_predictions)
    metrics.append(
        metric_row(
            "FINAL_TEST_POOLED",
            y_test,
            pred,
            score,
            fit_seconds,
            predict_seconds,
        )
    )
    monthly_errors: dict[int, float] = {}
    for month in TEST_MONTHS:
        mask = final_predictions["month"].to_numpy() == month
        row = metric_row(
            f"FINAL_{month}", y_test[mask], pred[mask], score[mask], 0.0, 0.0
        )
        metrics.append(row)
        monthly_errors[month] = row["misclassification_rate"]
    weighted_error = float(
        sum(TEST_WEIGHTS[month] * monthly_errors[month] for month in TEST_MONTHS)
    )
    metrics.append(
        {
            "model": MODEL_NAME,
            "split": "FINAL_TEST_WEIGHTED_0.1_0.3_0.6",
            "n": int(len(y_test)),
            "misclassification_rate": weighted_error,
            "accuracy": 1.0 - weighted_error,
            "roc_auc": None,
            "log_loss": None,
            "score_type": "probability",
            "fit_seconds": fit_seconds,
            "predict_seconds": predict_seconds,
        }
    )
    save_results(args.output_dir, metrics, predictions, metadata)
    print(f"All results saved to: {args.output_dir.resolve()}")


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

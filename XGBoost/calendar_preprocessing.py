"""Three preprocessing treatments with an optional one-based calendar index."""

from __future__ import annotations

import pandas as pd

from XGBoost.preprocessing_experiment import ExperimentPreprocessor, all_numeric, numeric_tokens
from XGBoost.train import month_index, normalize_text


BASE_METHODS = {
    "raw_minimal": "원형 최소 전처리",
    "numeric_categories": "숫자 범주를 수치형으로",
    "cardinality5": "범주형 상한 5",
}
CALENDAR_MODES = {"no_month": "시간 제외", "month_index": "시간 포함"}


def variant_key(method: str, mode: str) -> str:
    if method not in BASE_METHODS or mode not in CALENDAR_MODES:
        raise ValueError(f"Unknown treatment: {method}, {mode}")
    return f"{method}__{mode}"


def variant_label(method: str, mode: str) -> str:
    return f"{BASE_METHODS[method]} · {CALENDAR_MODES[mode]}"


def calendar_values(frame: pd.DataFrame) -> pd.Series:
    """202306 -> 1, ..., 202411 -> 18, independent of observed training months."""
    values = frame.LNMON.map(month_index).astype("float32")
    if values.isna().any() or not values.between(1, 18).all():
        raise ValueError("Calendar input must be within 202306 through 202411 (indices 1-18)")
    return values


class CalendarPreprocessor(ExperimentPreprocessor):
    """Retain the existing type map, lowering only its numeric category ceiling.

    The original map already applies integer-level/spacing rules. cardinality5
    converts its numeric-coded categorical columns with >5 observed numeric
    levels to numbers. Text categories, booleans, and existing numeric columns
    retain their roles. Counts and vocabularies use training rows only.
    """

    def __init__(self, base_method: str, calendar_mode: str, missing_cutoff: float = 0.5,
                 schema: dict | None = None):
        variant_key(base_method, calendar_mode)
        if schema is None:
            raise ValueError("A feature schema is required")
        self.base_method = base_method
        self.calendar_mode = calendar_mode
        self._calendar_ready = False
        self.converted_by_cardinality = []
        self.training_cardinality = {}
        parent_method = "current" if base_method == "cardinality5" else base_method
        super().__init__(parent_method, schema, missing_cutoff)

    def fit(self, train: pd.DataFrame):
        self._calendar_ready = False
        self.converted_by_cardinality = []
        self.training_cardinality = {}
        super().fit(train)
        if self.base_method == "cardinality5":
            for name in self.schema["categorical"]:
                if all_numeric(train[name]):
                    values = pd.to_numeric(normalize_text(train[name]), errors="raise").dropna()
                    count = int(values.nunique())
                    self.training_cardinality[name] = count
                    if count > 5:
                        self.converted_by_cardinality.append(name)
            self.numeric_category_columns = list(self.converted_by_cardinality)
        self._calendar_ready = True
        sample = self.transform(train.iloc[:min(8, len(train))])
        self.feature_names = sample.columns.tolist()
        self.feature_count = len(self.feature_names)
        self.numeric_feature_count = len(sample.select_dtypes(include="number").columns)
        self.categorical_feature_count = len(sample.select_dtypes(include="category").columns)
        if {"TARGET", "LNMON"} & set(self.feature_names):
            raise AssertionError("Target or raw YYYYMM entered the predictors")
        if ("month_index" in self.feature_names) != (self.calendar_mode == "month_index"):
            raise AssertionError("Calendar toggle does not match actual predictors")
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = super().transform(frame)
        for name in self.converted_by_cardinality:
            result[name] = numeric_tokens(frame[name])
        if self._calendar_ready and self.calendar_mode == "month_index":
            result["month_index"] = calendar_values(frame)
        return result

    def description(self) -> dict:
        result = super().description()
        result.update({
            "method": variant_key(self.base_method, self.calendar_mode),
            "label": variant_label(self.base_method, self.calendar_mode),
            "base_method": self.base_method, "calendar_mode": self.calendar_mode,
            "include_month": self.calendar_mode == "month_index",
            "calendar_encoding": "202306=1,...,202411=18; numeric float32; not categorical",
            "categorical_ceiling": 5 if self.base_method == "cardinality5" else None,
            "training_cardinality": self.training_cardinality,
            "converted_by_cardinality": self.converted_by_cardinality,
            "feature_names": self.feature_names,
            "numeric_features": self.numeric_feature_count,
            "native_categorical_features": self.categorical_feature_count,
        })
        return result

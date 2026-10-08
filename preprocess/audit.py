"""Streaming diagnostics for the KCB tabular dataset.

This module only profiles data. It never learns a preprocessing rule from a
validation period or changes the source CSV.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

import numpy as np
import pandas as pd


DATA_FILE = "kcb_202306_202405_undersampled_1to4.csv"
MISSING_TOKENS = {"", "NA", "N/A", "NAN", "NULL", "NONE", "<NA>"}
FIRST_MONTH = "202306"
FUTURE_START = "202406"
FUTURE_END = "202411"
BLOCK_WEIGHTS = (0.1, 0.3, 0.6)


def data_path() -> Path:
    """Find the original CSV from the repository root or preprocess/."""
    for directory in (Path.cwd(), Path.cwd().parent):
        candidate = directory / DATA_FILE
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Place {DATA_FILE} in the repository root.")


def month_number(value: str) -> int:
    """Convert YYYYMM to an integer month number, rejecting invalid months."""
    text = str(value).strip()
    if len(text) != 6 or not text.isdecimal():
        raise ValueError(f"Invalid YYYYMM value: {value!r}")
    year, month = int(text[:4]), int(text[4:])
    if not 1 <= month <= 12:
        raise ValueError(f"Invalid month in YYYYMM: {value!r}")
    return 12 * year + month


def month_index(value: str, origin: str = FIRST_MONTH) -> int:
    """One-based calendar month index relative to origin."""
    return month_number(value) - month_number(origin) + 1


def month_after(value: str, offset: int) -> str:
    zero_based = month_number(value) - 1 + offset
    year, month_zero = divmod(zero_based, 12)
    return f"{year:04d}{month_zero + 1:02d}"


def weighted_misclassification(
    y_true, y_pred, months, start_month: str = FUTURE_START
) -> tuple[float, pd.DataFrame]:
    """Average row-level weighted 0/1 errors over all six evaluation months.

    All six months must have at least one labeled row. The weights apply to
    individual rows; the denominator is the number of rows in all six months.
    """
    truth = np.asarray(y_true)
    prediction = np.asarray(y_pred)
    period = np.asarray(months, dtype=str)
    if not (len(truth) == len(prediction) == len(period)):
        raise ValueError("y_true, y_pred and months must have equal lengths")
    if not (np.isin(truth, [0, 1]).all() and np.isin(prediction, [0, 1]).all()):
        raise ValueError("This classification score requires 0/1 labels")
    expected = [month_after(start_month, i) for i in range(6)]
    if not np.isin(period, expected).all():
        raise ValueError("The score input contains months outside the six-month window")
    rows = []
    for block, weight in enumerate(BLOCK_WEIGHTS):
        pair = expected[2 * block : 2 * block + 2]
        if any(np.count_nonzero(period == m) == 0 for m in pair):
            raise ValueError(f"Both months in {pair} need labeled rows")
        included = np.isin(period, pair)
        error = float(np.mean(truth[included] != prediction[included]))
        rows.append(
            {"months": "–".join(pair), "n_rows": int(included.sum()),
             "misclassification_rate": error, "weight": weight,
             "weighted_error_sum": float(np.count_nonzero(truth[included] != prediction[included]) * weight),
             "contribution_to_total_mean": float(np.count_nonzero(truth[included] != prediction[included]) * weight / len(truth))}
        )
    details = pd.DataFrame(rows)
    return float(details["contribution_to_total_mean"].sum()), details


def backtest_windows() -> pd.DataFrame:
    """Two expanding-history analogues of the future six-month evaluation."""
    return pd.DataFrame(
        [
            {"train_start": FIRST_MONTH, "train_end": "202310", "score_start": "202311", "score_end": "202404"},
            {"train_start": FIRST_MONTH, "train_end": "202311", "score_start": "202312", "score_end": "202405"},
        ]
    )


def _candidate_type(name: str, values: set[str], n_unique: int, capped: bool,
                    numeric_rate: float, integer_rate: float) -> tuple[str, str]:
    if name == "TARGET":
        return "target", "exclude from predictors"
    if name == "LNMON":
        return "calendar_month", "parse YYYYMM; compare month index with no-time baseline"
    if n_unique == 0:
        return "all_missing", "check collection period and meaning"
    if n_unique == 1 and not capped:
        return "constant", "may change in future months"
    numeric_values = set()
    if not capped and numeric_rate == 1:
        try:
            numeric_values = {float(v) for v in values}
        except ValueError:
            pass
    if numeric_values and numeric_values <= {0.0, 1.0}:
        return "binary", "use as a binary feature; expose as categorical where supported"
    if name.endswith("_FLAG") or "_SV_FLAG_" in name:
        return "flag_review", "check flag values and source-column missingness"
    if numeric_rate < 0.99:
        return "categorical_or_mixed", "inspect raw tokens; never coerce failed values silently"
    if integer_rate >= 0.99:
        if not capped and n_unique <= 20:
            return "integer_category_or_count", "verify code versus ordered count using data dictionary"
        if not capped and n_unique <= 100:
            return "integer_code_or_ordered", "verify category, ordinal and numeric interpretations"
        return "integer_numeric_or_identifier", "check ID/date/sentinel patterns and meaning"
    return "continuous_candidate", "check units, outliers and sentinel-like values"


def _integer_levels(values: set[str]) -> set[int] | None:
    """Normalize tokens such as '1' and '1.0' before checking code levels."""
    try:
        numbers = {Decimal(value) for value in values}
    except InvalidOperation:
        return None
    if not all(number.is_finite() and number == number.to_integral_value() for number in numbers):
        return None
    return {int(number) for number in numbers}


def _base_type(
    name: str, values: set[str], capped: bool, nonmissing: int,
    numeric_rate: float, integer_rate: float,
) -> tuple:
    """Apply the explicit baseline rules, keeping ambiguous cases for review."""
    if name == "TARGET":
        return "target", "prediction label, never an input feature", None, None, None, None, None, None, None, None, None
    if name == "LNMON":
        return "month_index", "convert YYYYMM to one-based consecutive months", None, None, None, None, None, None, None, None, None
    if nonmissing == 0:
        return "unknown", "all observed values are missing", None, None, None, None, None, None, None, None, None
    if not capped:
        normalized = {value.strip().upper() for value in values}
        if normalized == {"Y", "N"}:
            return "boolean", "Y/N maps to 1/0", None, None, None, None, None, None, None, None, None
        if normalized == {"Y", "N", "P"}:
            return "categorical", "keep Y, N and P as three distinct levels", None, None, None, None, None, None, None, None, None
    levels = _integer_levels(values) if not capped and numeric_rate == 1 and integer_rate == 1 else None
    one_based = zero_based = None
    coverage = None
    step = None
    gap_exceptions = None
    equal_step_small = None
    regular_or_three_levels_small = None
    gap_location = None
    gap_between = None
    if levels:
        ordered = sorted(levels)
        low, high = ordered[0], ordered[-1]
        coverage = len(levels) / (high - low + 1)
        one_based = low == 1 and len(levels) == high
        zero_based = low == 0 and len(levels) == high + 1
        if len(ordered) >= 2:
            differences = [right - left for left, right in zip(ordered, ordered[1:])]
            gap_counts = Counter(differences)
            # For two different gaps (three levels), use the smaller step as
            # the baseline so the larger edge gap is the documented exception.
            step = min(gap_counts, key=lambda gap: (-gap_counts[gap], gap))
            exception_indices = [i for i, gap in enumerate(differences) if gap != step]
            gap_exceptions = len(exception_indices)
            if not exception_indices:
                gap_location = "none"
            elif len(exception_indices) == 1:
                i = exception_indices[0]
                gap_location = (
                    "lower_edge" if i == 0 else
                    "upper_edge" if i == len(differences) - 1 else "internal"
                )
                gap_between = f"{ordered[i]}→{ordered[i + 1]}"
            else:
                gap_location = "multiple"
        equal_step_small = gap_exceptions == 0 and len(ordered) <= 20
        regular_or_three_levels_small = (
            len(ordered) <= 20 and (gap_exceptions == 0 or len(ordered) == 3)
        )
        if levels <= {0, 1}:
            return "boolean", "observed numeric levels are 0/1", one_based, zero_based, coverage, step, gap_exceptions, equal_step_small, regular_or_three_levels_small, gap_location, gap_between
        if regular_or_three_levels_small:
            reason = (
                "three observed integer levels; categorical baseline regardless of spacing"
                if len(ordered) == 3 and gap_exceptions else
                "2–20 integer levels form an arithmetic progression"
            )
            return "categorical", reason, one_based, zero_based, coverage, step, gap_exceptions, equal_step_small, regular_or_three_levels_small, gap_location, gap_between
    if numeric_rate < 1:
        return "categorical", "preserve nonnumeric tokens; inspect mixed values", one_based, zero_based, coverage, step, gap_exceptions, equal_step_small, regular_or_three_levels_small, gap_location, gap_between
    reason = (
        "unequal adjacent integer gaps; use numeric baseline and compare category encoding"
        if gap_exceptions is not None and gap_exceptions > 0 else
        "numeric baseline; compare category encoding where noted"
    )
    return "numeric", reason, one_based, zero_based, coverage, step, gap_exceptions, equal_step_small, regular_or_three_levels_small, gap_location, gap_between


@dataclass
class AuditResult:
    features: pd.DataFrame
    flags: pd.DataFrame
    rows: pd.DataFrame
    sample: pd.DataFrame
    total_rows: int
    distinct_values: dict[str, set[str] | None]


def profile_dataset(
    path: Path, chunksize: int = 10_000, max_rows: int | None = None,
    sample_per_chunk: int = 100, seed: int = 42, cardinality_cap: int = 1000,
    supervised_train_end: str = "202311",
) -> AuditResult:
    """Profile all columns without loading the whole string table into memory.

    Cardinality is exact up to cardinality_cap. Associations with TARGET use
    only the period ending at supervised_train_end; later months are reserved
    for the historical backtests.
    """
    columns = pd.read_csv(path, nrows=0, encoding="utf-8-sig").columns.tolist()
    if len(columns) != len(set(columns)):
        raise ValueError("Duplicate CSV headers require manual resolution")
    if not {"LNMON", "TARGET"} <= set(columns):
        raise ValueError("LNMON and TARGET columns are required")
    source_cols = [c for c in columns if c not in {"LNMON", "TARGET"} and "_SV_FLAG_" not in c]
    flag_cols = [c for c in columns if "_SV_FLAG_" in c]
    totals = defaultdict(lambda: np.zeros(len(columns), dtype=np.int64))
    min_value = np.full(len(columns), np.inf)
    max_value = np.full(len(columns), -np.inf)
    distinct = [set() for _ in columns]
    capped = np.zeros(len(columns), dtype=bool)
    flag_counts = {name: Counter() for name in flag_cols}
    rows_parts, sample_parts = [], []
    total_rows = 0
    reader = pd.read_csv(
        path, dtype="string", keep_default_na=False, na_filter=False,
        encoding="utf-8-sig", chunksize=chunksize, nrows=max_rows,
    )
    for chunk_no, chunk in enumerate(reader):
        raw = chunk.apply(lambda s: s.str.strip())
        missing = raw.apply(lambda s: s.str.upper().isin(MISSING_TOKENS))
        if not raw["TARGET"].isin(["0", "1"]).all():
            raise ValueError("TARGET includes values outside 0/1")
        months = raw["LNMON"]
        try:
            month_numbers = months.map(month_number)
        except ValueError as exc:
            raise ValueError(f"Invalid LNMON in CSV chunk {chunk_no}") from exc
        total_rows += len(chunk)
        train = month_numbers <= month_number(supervised_train_end)
        positive = raw["TARGET"].eq("1")
        train_positive = train & positive
        missing_source = missing[source_cols]
        missing_count = missing_source.sum(axis=1).astype("int32")
        rows_parts.append(pd.DataFrame({
            "LNMON": months.to_numpy(), "TARGET": positive.astype("int8").to_numpy(),
            "missing_count": missing_count.to_numpy(),
        }))
        sample_parts.append(raw.sample(n=min(sample_per_chunk, len(raw)), random_state=seed + chunk_no))
        totals["train_missing"] += missing.loc[train].sum(axis=0).to_numpy(dtype=np.int64)
        totals["train_missing_positive"] += missing.loc[train_positive].sum(axis=0).to_numpy(dtype=np.int64)
        totals["train_rows"] += int(train.sum())
        totals["train_positives"] += int(train_positive.sum())
        for i, name in enumerate(columns):
            series = raw[name]
            is_missing = missing[name]
            valid = series[~is_missing]
            totals["missing"][i] += int(is_missing.sum())
            totals["blank"][i] += int(series.eq("").sum())
            totals["zero"][i] += int(valid.isin(["0", "0.0"]).sum())
            totals["leading_zero"][i] += int(valid.str.fullmatch(r"-?0\d+").sum())
            totals["sentinel_8_9"][i] += int(valid.str.fullmatch(r"-?(?:8{2,}|9{2,})(?:\.0+)?").sum())
            if not capped[i]:
                distinct[i].update(valid.unique().tolist())
                if len(distinct[i]) > cardinality_cap:
                    capped[i] = True
                    distinct[i].clear()
            number = pd.to_numeric(valid, errors="coerce")
            finite = number[np.isfinite(number)]
            totals["numeric"][i] += len(finite)
            totals["integer"][i] += int((finite % 1 == 0).sum())
            if not finite.empty:
                min_value[i] = min(min_value[i], float(finite.min()))
                max_value[i] = max(max_value[i], float(finite.max()))
        for flag in flag_cols:
            base = flag.split("_SV_FLAG_", 1)[0]
            if base not in columns:
                continue
            active = raw[flag].eq("1")
            absent = missing[base]
            counts = flag_counts[flag]
            counts["both"] += int((active & absent).sum())
            counts["flag_only"] += int((active & ~absent).sum())
            counts["missing_only"] += int((~active & absent).sum())
            counts["neither"] += int((~active & ~absent).sum())
    if total_rows == 0:
        raise ValueError("The CSV has no data rows")
    records = []
    for i, name in enumerate(columns):
        nonmissing = total_rows - totals["missing"][i]
        numeric_rate = totals["numeric"][i] / nonmissing if nonmissing else 0.0
        integer_rate = totals["integer"][i] / totals["numeric"][i] if totals["numeric"][i] else 0.0
        unique = cardinality_cap + 1 if capped[i] else len(distinct[i])
        candidate, review = _candidate_type(name, distinct[i], unique, bool(capped[i]), numeric_rate, integer_rate)
        base_type, type_reason, one_based, zero_based, coverage, step, gap_exceptions, equal_step_small, regular_or_three_levels_small, gap_location, gap_between = _base_type(
            name, distinct[i], bool(capped[i]), nonmissing, numeric_rate, integer_rate
        )
        integer_levels = (
            _integer_levels(distinct[i])
            if not capped[i] and numeric_rate == 1 and integer_rate == 1 else None
        )
        train_missing = totals["train_missing"][i]
        train_observed = totals["train_rows"][i] - train_missing
        missing_rate_y = totals["train_missing_positive"][i] / train_missing if train_missing else np.nan
        observed_positive = totals["train_positives"][i] - totals["train_missing_positive"][i]
        observed_rate_y = observed_positive / train_observed if train_observed else np.nan
        records.append({
            "column": name, "candidate_type": candidate, "review": review,
            "base_type": base_type, "type_reason": type_reason,
            "one_based_contiguous": one_based, "zero_based_contiguous": zero_based,
            "integer_level_coverage": coverage, "integer_step": step,
            "integer_level_count": len(integer_levels) if integer_levels is not None else np.nan,
            "equal_step_at_most_20": equal_step_small,
            "integer_gap_exceptions": gap_exceptions,
            "integer_gap_exception_location": gap_location,
            "integer_gap_exception_between": gap_between,
            "regular_or_three_levels_at_most_20": regular_or_three_levels_small,
            "missing_n": int(totals["missing"][i]), "missing_pct": totals["missing"][i] / total_rows,
            "blank_n": int(totals["blank"][i]), "nonmissing_n": int(nonmissing),
            "distinct_at_least": int(unique), "distinct_capped": bool(capped[i]),
            "numeric_parse_rate": numeric_rate, "integer_rate_of_numeric": integer_rate,
            "numeric_min": min_value[i] if np.isfinite(min_value[i]) else np.nan,
            "numeric_max": max_value[i] if np.isfinite(max_value[i]) else np.nan,
            "zero_n": int(totals["zero"][i]), "leading_zero_n": int(totals["leading_zero"][i]),
            "sentinel_like_8_9_n": int(totals["sentinel_8_9"][i]),
            "train_missing_n": int(train_missing), "train_missing_target_rate": missing_rate_y,
            "train_observed_target_rate": observed_rate_y,
            "train_target_rate_gap": missing_rate_y - observed_rate_y,
        })
    flag_records = []
    for name, count in flag_counts.items():
        base = name.split("_SV_FLAG_", 1)[0]
        total_active = count["both"] + count["flag_only"]
        total_absent = count["both"] + count["missing_only"]
        flag_records.append({
            "flag": name, "source": base,
            **{key: count[key] for key in ("both", "flag_only", "missing_only", "neither")},
            "precision_for_missing": count["both"] / total_active if total_active else np.nan,
            "recall_for_missing": count["both"] / total_absent if total_absent else np.nan,
        })
    return AuditResult(
        features=pd.DataFrame(records), flags=pd.DataFrame(flag_records),
        rows=pd.concat(rows_parts, ignore_index=True),
        sample=pd.concat(sample_parts, ignore_index=True), total_rows=total_rows,
        distinct_values={name: None if capped[i] else distinct[i] for i, name in enumerate(columns)},
    )


def _feature_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "해당 변수 없음.\n"
    lines = [
        "| 변수 | 기본 유형 | 원본 고유값 수(하한) | 정수 수준 수 | 최소 | 최대 | 대표 간격 | 다른 간격 수 | 예외 위치 | 예외 구간 | 결측률 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|---|---:|",
    ]
    for row in frame.itertuples(index=False):
        minimum = "—" if pd.isna(row.numeric_min) else f"{row.numeric_min:g}"
        maximum = "—" if pd.isna(row.numeric_max) else f"{row.numeric_max:g}"
        step = "—" if pd.isna(row.integer_step) else str(int(row.integer_step))
        exceptions = "—" if pd.isna(row.integer_gap_exceptions) else str(int(row.integer_gap_exceptions))
        location = "—" if pd.isna(row.integer_gap_exception_location) else row.integer_gap_exception_location
        between = "—" if pd.isna(row.integer_gap_exception_between) else row.integer_gap_exception_between
        unique = f">{row.distinct_at_least - 1}" if row.distinct_capped else str(row.distinct_at_least)
        integer_count = "—" if pd.isna(row.integer_level_count) else str(int(row.integer_level_count))
        lines.append(
            f"| `{row.column}` | {row.base_type} | {unique} | {integer_count} | {minimum} | {maximum} | {step} | {exceptions} | {location} | {between} | {row.missing_pct:.1%} |"
        )
    return "\n".join(lines) + "\n"


def type_experiments_markdown(result: AuditResult) -> str:
    """Record alternatives for every uncertain integer group and text category."""
    frame = result.features
    integer_categories = frame.loc[
        frame.regular_or_three_levels_at_most_20.eq(True) & frame.base_type.eq("categorical")
    ]
    edge_gap = frame.loc[
        frame.base_type.eq("numeric") & frame.integer_level_count.between(4, 20) &
        frame.integer_gap_exceptions.eq(1) &
        frame.integer_gap_exception_location.isin(["lower_edge", "upper_edge"])
    ]
    internal_gap = frame.loc[
        frame.base_type.eq("numeric") & frame.integer_level_count.between(4, 20) &
        frame.integer_gap_exception_location.eq("internal")
    ]
    small_irregular = frame.loc[
        frame.base_type.eq("numeric") &
        frame.candidate_type.isin(["integer_category_or_count", "integer_code_or_ordered"]) &
        frame.integer_level_count.between(4, 20) &
        frame.integer_gap_exceptions.ge(2)
    ]
    medium_integer = frame.loc[
        frame.base_type.eq("numeric") &
        frame.candidate_type.isin(["integer_category_or_count", "integer_code_or_ordered"]) &
        frame.distinct_at_least.between(21, 100)
    ]
    high_integer = frame.loc[
        frame.base_type.eq("numeric") & frame.candidate_type.eq("integer_numeric_or_identifier")
    ]
    text_or_mixed = frame.loc[
        frame.numeric_parse_rate.eq(0) & frame.base_type.isin(["boolean", "categorical"])
    ]
    sparse = frame.loc[
        ~frame.column.isin(["LNMON", "TARGET"]) & frame.missing_pct.ge(0.5)
    ].sort_values("missing_pct", ascending=False)
    parts = [
        "# 변수 유형과 비교 실험\n",
        f"`data_audit.ipynb`가 원본 CSV {result.total_rows:,}행에서 생성한 기록입니다. "
        "`preprocess/feature_types.csv`의 `variable,type`은 **기본 입력 유형**이며 확정된 업무 의미가 아닙니다. "
        "`target`은 입력에서 제외하고, `month_index`는 `YYYYMM`을 202306=1, 202411=18로 바꿉니다. "
        "과거 역검증에서는 각 학습 기간만으로 유형과 인코더를 결정해야 합니다.\n",
        "## 평가 기준\n",
        "각 행에 `0.1`(첫 2개월), `0.3`(다음 2개월), `0.6`(마지막 2개월)을 적용합니다. "
        "점수는 `sum(weight_i * (prediction_i != target_i)) / 전체 행 수`입니다. "
        "분류 임계값도 평가 기간의 정답을 보지 않고 정합니다. "
        "평가 데이터가 실제로 같은 언더샘플 비율인지 확보 후 확인합니다.\n",
        "## 문자열 범주와 불리언\n",
        "`Y/N`만 있는 열은 기본 `boolean` (`Y=1`, `N=0`)입니다. "
        "`Y/N/P`는 기본 `categorical`로 세 값을 모두 보존합니다. "
        "`P`의 뜻은 확인되지 않았으므로 결측으로 확정하지 않습니다. "
        "`SS1200000`은 **3범주 유지**와 **P를 결측으로 바꾸되 P indicator 유지**를 비교합니다. "
        "`TS_VOLATILITY_APS001_FLIPS`는 명목 범주와 순서형 `Steady < Changed_1_Time < Changed_2_Plus`를 비교하되, 이 순서가 변수 정의에 맞는지 확인합니다. "
        "`TS_MOMENTUM_APS001_STATE`는 네 상태를 범주로 유지한 입력과 상태 변화를 분해한 입력을 비교합니다.\n",
        _feature_table(text_or_mixed),
        "## 고유값 2~20개이고 간격이 일정하거나 고유값이 정확히 3개인 정수\n",
        "시작값은 0, 1 또는 다른 정수이고, 간격도 1 이상 어느 정수든 가능합니다. "
        "정렬한 관측 수준의 인접 차이가 **모두 동일**하면 기본 `categorical`입니다. "
        "**고유값이 정확히 3개**이면 두 간격이 달라도 `categorical`로 둡니다. "
        "고유값이 4~20개이고 다른 간격이 하나라도 있으면, 예외가 양 끝에 있더라도 기본 `numeric`입니다. "
        "간격 예외만으로 unknown이나 결측을 추정하지 않습니다. "
        "숫자 0/1은 우선 `boolean`으로 둡니다. 이런 정수는 횟수·점수·순서형도 될 수 있으므로 "
        "각 열에서 **범주형 입력**과 **숫자형/순서형 입력**을 동일한 시간 역검증으로 비교합니다. "
        "범주 인코딩은 학습 기간에서만 맞추고, 미래에 새 수준이 나오면 미지 범주로 처리합니다.\n",
        _feature_table(integer_categories),
        "## 고유값 4~20개이고 끝 간격 예외가 1개인 정수\n",
        "기본 `numeric`입니다. 끝 간격의 차이만으로 특수값이라고 간주하지 않습니다. "
        "숫자형·범주형 입력을 비교합니다.\n",
        _feature_table(edge_gap),
        "## 고유값 4~20개이고 내부 간격 예외가 1개인 정수\n",
        "기본 `numeric`입니다. 숫자형·범주형 입력을 비교합니다. "
        "`예외 구간`은 인접한 두 관측값을 나타냅니다.\n",
        _feature_table(internal_gap),
        "## 고유값 4~20개이고 간격 예외가 2개 이상인 정수\n",
        "기본 `numeric`입니다. 일부 수준이 빠진 범주 코드이거나 특수값이 섞였을 수 있으므로 "
        "숫자형·범주형과 특수값 처리를 비교합니다.\n",
        _feature_table(small_irregular),
        "## 고유값 21~100개의 정수\n",
        "기본 `numeric`입니다. 값의 간격이 일정하더라도 20개 제한을 넘습니다. 범주 코드나 등급일 수 있으므로 "
        "**숫자형·순서형·범주형**을 비교합니다. 높은 결측률과 값 간격도 함께 검토합니다.\n",
        _feature_table(medium_integer),
        "## 고유값 100개 초과 정수\n",
        "기본 `numeric`입니다. 현재 진단 이름의 `identifier`는 식별자 판정이 아닙니다. "
        "식별자 여부는 반복 패턴과 변수 정의로 판단합니다. 범주 코드 근거가 생기면 native categorical/embedding을 비교하고, "
        "높은 고유값 수를 무조건 one-hot으로 확장하지 않습니다. `>1000`은 정확한 고유값 수를 추적하지 않았다는 뜻입니다.\n",
        _feature_table(high_integer),
        "## 결측과 특수값\n",
        "결측률만으로 삭제하지 않습니다. 원본 값만 사용 / 원본 값 + 개별 missing indicator / "
        "행별 또는 변수군별 결측 개수를 같은 역검증에서 비교합니다. "
        "`_SV_FLAG_`가 원본 결측과 정확히 중복되면 중복 특성을 제거하는 실험을 하고, "
        "복수 플래그가 결측 사유를 구분하면 각각 유지하는 실험을 합니다. "
        "특수값의 업무 의미가 없으므로 0이나 8/9 반복 숫자를 자동으로 결측 처리하지 않습니다.\n",
        _feature_table(sparse),
        "## LNMON과 모델별 입력\n",
        "`LNMON`은 연속 월 인덱스와 변수 제외를 우선 비교합니다. 원본 `YYYYMM` 정수 차이는 월 차이가 아닙니다. "
        "미래 연월 one-hot은 학습에 없는 범주가 되므로 기본 표현으로 삼지 않습니다. "
        "트리 모델은 학습 범위 밖의 월 추세를 인덱스 하나만으로 외삽하지 못할 수 있습니다. "
        "RealMLP·TabM·ModernNCA·xRFM은 수치 대체·스케일링과 범주 인코딩을 학습 기간에서만 맞춥니다. "
        "TabICL·TabPFN의 내장 전처리를 사용할 때도 정수 코드와 Y/N/P의 유형을 명시적으로 전달합니다.\n",
    ]
    return "\n".join(parts)


def write_type_artifacts(result: AuditResult, repository_root: Path) -> tuple[Path, Path]:
    """Write the type map and experiment notes inside preprocess/."""
    output_dir = repository_root / "preprocess"
    schema_path = output_dir / "feature_types.csv"
    notes_path = output_dir / "type_experiments.md"
    schema = result.features[["column", "base_type"]].rename(
        columns={"column": "variable", "base_type": "type"}
    )
    schema.to_csv(schema_path, index=False, encoding="utf-8")
    notes_path.write_text(type_experiments_markdown(result), encoding="utf-8")
    return schema_path, notes_path

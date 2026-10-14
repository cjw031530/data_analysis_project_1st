"""Calendar boundaries, cardinality cutoffs, and compatibility with old results."""

import pickle
import unittest
import json

import numpy as np
import pandas as pd

from XGBoost.calendar_preprocessing import BASE_METHODS, CalendarPreprocessor, calendar_values
from XGBoost.preprocessing_experiment import ExperimentPreprocessor
from XGBoost.train import best_threshold, repo_path


class LegacyPreprocessorUnpickler(pickle.Unpickler):
    """Read earlier experiment objects saved by python -m as __main__ classes."""

    def find_class(self, module, name):
        if module == "__main__" and name == "ExperimentPreprocessor":
            return ExperimentPreprocessor
        return super().find_class(module, name)


class CalendarTests(unittest.TestCase):
    def setUp(self):
        self.schema = {"numeric": ["value"], "categorical": ["five", "six", "text", "formatted"],
                       "boolean": ["flag"], "month_index": ["LNMON"], "target": ["TARGET"]}
        self.train = pd.DataFrame({"value": np.arange(6, dtype="float32"),
            "five": pd.Series(["1", "2", "3", "4", "5", "1"], dtype="string"),
            "six": pd.Series(["1", "2", "3", "4", "5", "6"], dtype="string"),
            "text": pd.Series(list("abcdef"), dtype="string"),
            "formatted": pd.Series(["1", "1.0", "2", "3", "4", "5"], dtype="string"),
            "flag": pd.Series(["Y", "N"] * 3, dtype="string"),
            "LNMON": pd.Series(["202306"] * 6, dtype="string"), "TARGET": [0, 1] * 3})

    def prep(self, method, mode="no_month"):
        return CalendarPreprocessor(method, mode, schema=self.schema).fit(self.train)

    def test_all_eighteen_calendar_months(self):
        months = pd.date_range("2023-06-01", "2024-11-01", freq="MS").strftime("%Y%m")
        np.testing.assert_array_equal(calendar_values(pd.DataFrame({"LNMON": months})), np.arange(1, 19))
        with self.assertRaises(ValueError):
            calendar_values(pd.DataFrame({"LNMON": ["202412"]}))

    def test_six_conditions_exclude_target_and_raw_month(self):
        for method in BASE_METHODS:
            for mode in ("no_month", "month_index"):
                prep = self.prep(method, mode)
                values = prep.transform(self.train)
                self.assertFalse({"TARGET", "LNMON"} & set(values.columns))
                self.assertEqual("month_index" in values, mode == "month_index")
                self.assertEqual(values.shape[1], 6 + int(mode == "month_index"))
                if mode == "month_index":
                    self.assertEqual(str(values.month_index.dtype), "float32")

    def test_cardinality_five_is_inclusive_and_numeric_levels_normalized(self):
        prep = self.prep("cardinality5")
        values = prep.transform(self.train)
        self.assertEqual(prep.converted_by_cardinality, ["six"])
        self.assertEqual(prep.training_cardinality["formatted"], 5)
        self.assertEqual(str(values.five.dtype), "category")
        self.assertEqual(str(values.formatted.dtype), "category")
        self.assertEqual(str(values.six.dtype), "float32")
        self.assertEqual(str(values.text.dtype), "category")
        np.testing.assert_array_equal(values.flag, [1, 0] * 3)

    def test_cutoff_uses_training_only_and_future_months_work(self):
        prep = self.prep("cardinality5", "month_index")
        future = self.train.iloc[[0]].copy()
        future["five"] = "6"
        future["six"] = "100"
        future["LNMON"] = "202411"
        values = prep.transform(future)
        self.assertTrue(pd.isna(values.five.iloc[0]))
        self.assertEqual(values.six.iloc[0], 100)
        self.assertEqual(values.month_index.iloc[0], 18)
        self.assertEqual(prep.converted_by_cardinality, ["six"])

    def test_no_calendar_variants_match_original_exactly(self):
        for method in ("raw_minimal", "numeric_categories"):
            old = ExperimentPreprocessor(method, self.schema).fit(self.train)
            new = self.prep(method)
            pd.testing.assert_frame_equal(old.transform(self.train), new.transform(self.train))

    def test_calendar_added_once_and_never_classified_by_cardinality(self):
        for method in BASE_METHODS:
            off, on = self.prep(method), self.prep(method, "month_index")
            pd.testing.assert_frame_equal(off.transform(self.train), on.transform(self.train).drop(columns="month_index"))
            self.assertEqual(on.feature_names.count("month_index"), 1)
            self.assertNotIn("month_index", on.training_cardinality)

    def test_pickle_is_reusable_for_future_prediction(self):
        prep = pickle.loads(pickle.dumps(self.prep("cardinality5", "month_index")))
        future = self.train.copy()
        future["LNMON"] = "202406"
        np.testing.assert_array_equal(prep.transform(future).month_index, np.full(6, 13))


def verify_saved_results(output_dir="XGBoost/outputs/calendar_experiment"):
    """Independently recompute errors from saved OOF/holdout arrays and source labels."""
    import xgboost as xgb
    from XGBoost.preprocessing_experiment import file_hash

    output = repo_path(str(output_dir))
    report = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    signature = report["environment"]
    config = signature["config"]
    assert file_hash(repo_path(config["paths"]["data"])) == signature["data_sha256"]
    assert file_hash(repo_path(config["paths"]["feature_types"])) == signature["types_sha256"]
    for name, digest in signature["code_sha256"].items():
        assert file_hash(repo_path(f"XGBoost/{name}")) == digest
    source = pd.read_csv(repo_path(config["paths"]["data"]), usecols=["LNMON", "TARGET"],
                         dtype={"LNMON": "string", "TARGET": "int8"})
    weights = {str(k): float(v) for k, v in config["test"]["weights"].items()}
    expected_test = source.index[source.LNMON.isin(weights)].to_numpy()
    expected_oof = source.index[source.LNMON.isin([f["valid_month"] for f in config["folds"]])].to_numpy()
    baseline = [r for r in report["results"] if r["phase"] == "screening"]
    assert len(baseline) == 6
    assert all(r["params"] == config["model"]["baseline"] for r in baseline)
    for result in report["results"]:
        path = repo_path(result["checkpoint_path"])
        with np.load(path / "predictions.npz") as saved:
            indices, probabilities, labels = saved["indices"], saved["probability"], saved["labels"]
            np.testing.assert_array_equal(indices, expected_test)
            np.testing.assert_array_equal(labels, source.loc[indices, "TARGET"].to_numpy())
            np.testing.assert_array_equal(saved["months"], source.loc[indices, "LNMON"].to_numpy())
            assert np.isfinite(probabilities).all() and ((probabilities >= 0) & (probabilities <= 1)).all()
            predictions = probabilities >= result["cv"]["threshold"]
            errors = predictions != labels
            row_weights = source.loc[indices, "LNMON"].map(weights).to_numpy(dtype=float)
            assert abs(float(np.dot(row_weights, errors) / len(indices)) - result["test"]["weighted_score"]) < 1e-12
            assert abs(float(errors.mean()) - result["test"]["misclassification_rate"]) < 1e-12
            for month in weights:
                mask = source.loc[indices, "LNMON"].eq(month).to_numpy()
                assert abs(float(errors[mask].mean()) - result["test_by_month"][month]["misclassification_rate"]) < 1e-12
            assert sum(result["test"][k] for k in ("tn", "fp", "fn", "tp")) == len(expected_test)
        model = xgb.Booster()
        model.load_model(path / "model.ubj")
        assert model.num_boosted_rounds() == result["cv"]["rounds"]
        assert model.num_features() == result["preprocessing"]["output_features"]
        assert ("month_index" in model.feature_names) == result["include_month"]
        assert not ({"TARGET", "LNMON"} & set(model.feature_names))
        with (path / "preprocessor.pkl").open("rb") as source_file:
            prep = LegacyPreprocessorUnpickler(source_file).load()
        assert prep.feature_names == model.feature_names
        cv_dir = path
        if result["phase"] == "tuned":
            trial = report["tuning"][result["method"]]["best_trial"]
            cv_dir = path.parent / "screening" if trial == 0 else path.parent / "tuning" / f"trial_{trial:03d}"
        oof_indices, oof_probabilities = [], []
        for fold in result["cv"]["folds"]:
            with np.load(cv_dir / f"{fold['name']}.npz") as saved:
                idx = saved["indices"].copy()
                probability = saved["probability"].copy()
            assert source.loc[idx, "LNMON"].eq(fold["month"]).all()
            assert len(idx) == fold["valid_rows"]
            oof_indices.append(idx)
            oof_probabilities.append(probability)
            if result["base_method"] == "cardinality5":
                prep_info = fold["preprocessing"]
                for name, count in prep_info["training_cardinality"].items():
                    assert (name in prep_info["converted_by_cardinality"]) == (count > 5)
        idx, probability = np.concatenate(oof_indices), np.concatenate(oof_probabilities)
        assert len(np.unique(idx)) == len(expected_oof)
        np.testing.assert_array_equal(np.sort(idx), expected_oof)
        threshold, _ = best_threshold(source.loc[idx, "TARGET"].to_numpy(), probability)
        assert threshold == result["cv"]["threshold"]
        error = float(np.mean((probability >= threshold) != source.loc[idx, "TARGET"].to_numpy()))
        assert abs(error - result["cv"]["pooled"]["misclassification_rate"]) < 1e-12
        assert max(1, int(np.median([f["best_rounds"] for f in result["cv"]["folds"]]))) == result["cv"]["rounds"]
    assert len(report["tuning_finalists"]) == 2
    assert all(choice["trials"] == 12 for choice in report["tuning"].values())
    frozen = json.loads((output / "checkpoints" / "frozen_choices.json").read_text(encoding="utf-8"))
    assert frozen["tuned"] == report["tuning"]
    print(f"Verified {len(report['results'])} result records, {len(expected_oof):,} OOF rows and "
          f"{len(expected_test):,} holdout rows per condition.")
    return {"records": len(report["results"]), "oof_rows": len(expected_oof), "test_rows": len(expected_test)}


if __name__ == "__main__":
    unittest.main()

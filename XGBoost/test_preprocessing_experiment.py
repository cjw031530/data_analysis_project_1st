"""Check preprocessing boundaries and the requested weighted-score definition."""

import unittest

import numpy as np
import pandas as pd

from XGBoost.preprocessing_experiment import ExperimentPreprocessor, METHODS, weighted_error


class PreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.schema = {"numeric": ["zero", "half", "mostly"],
                       "categorical": ["code", "text"], "boolean": ["flag"],
                       "month_index": ["LNMON"], "target": ["TARGET"]}
        self.train = pd.DataFrame({
            "zero": [0., 1., 0., 2.], "half": [np.nan, 2., np.nan, 4.],
            "mostly": [np.nan, np.nan, np.nan, 4.],
            "code": pd.Series(["1", "3", "1", "3"], dtype="string"),
            "text": pd.Series(["a", "b", "a", "b"], dtype="string"),
            "flag": pd.Series(["Y", "N", "Y", "N"], dtype="string"),
            "LNMON": pd.Series(["202306"] * 4, dtype="string"), "TARGET": [0, 1, 0, 1],
        })

    def test_seven_methods_exclude_target_and_month(self):
        for method in METHODS:
            prep = ExperimentPreprocessor(method, self.schema).fit(self.train)
            values = prep.transform(self.train)
            self.assertEqual(len(values), len(self.train))
            self.assertFalse({"TARGET", "LNMON", "month_index"} & set(prep.feature_names))

    def test_missing_cutoff_is_strict_and_training_only(self):
        prep = ExperimentPreprocessor("drop_missing", self.schema).fit(self.train)
        self.assertEqual(prep.dropped, ["mostly"])
        future = self.train.copy()
        future["half"] = np.nan
        self.assertIn("half", prep.transform(future).columns)
        flags = ExperimentPreprocessor("drop_missing_keep_flags", self.schema).fit(self.train).transform(self.train)
        self.assertNotIn("mostly", flags)
        np.testing.assert_array_equal(flags["missing__mostly"], [1, 1, 1, 0])

    def test_onehot_preserves_zero_nan_and_train_vocabulary(self):
        prep = ExperimentPreprocessor("one_hot", self.schema).fit(self.train)
        future = self.train.iloc[[0]].copy()
        future["text"] = "unseen"
        values = prep.transform(future)
        self.assertIsInstance(values, np.ndarray)
        self.assertEqual(values[0, prep.feature_names.index("zero")], 0)
        self.assertTrue(np.isnan(values[0, prep.feature_names.index("half")]))
        indices = [i for i, name in enumerate(prep.feature_names) if name.startswith("text_")]
        self.assertEqual(float(values[0, indices].sum()), 0)
        self.assertNotIn("text_unseen", prep.feature_names)

    def test_numeric_categories_keep_original_values(self):
        prep = ExperimentPreprocessor("numeric_categories", self.schema).fit(self.train)
        values = prep.transform(self.train)
        np.testing.assert_array_equal(values.code, [1, 3, 1, 3])
        self.assertEqual(str(values.text.dtype), "category")
        raw = ExperimentPreprocessor("raw_minimal", self.schema).fit(self.train)
        future = self.train.iloc[[0]].copy()
        future["text"] = "unseen"
        self.assertTrue(np.isnan(raw.transform(future).text.iloc[0]))

    def test_weighted_score_divides_by_rows_not_weight_sum(self):
        frame = pd.DataFrame({"LNMON": ["202403", "202404", "202405", "202405"],
                              "TARGET": [1, 0, 1, 0]})
        probability = np.array([0.2, 0.8, 0.8, 0.2])
        self.assertAlmostEqual(weighted_error(frame, probability, 0.5,
                                             {"202403": 0.1, "202404": 0.3, "202405": 0.6}), 0.1)


if __name__ == "__main__":
    unittest.main()

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import promote_model


class EvaluateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

        self.threshold_path = Path(self.temp.name) / "threshold.json"
        self.write_threshold(0.5)

        self.X = pd.DataFrame({
            "temperature": [1.0, 2.0, 3.0, 4.0],
            "speed": [10, 20, 30, 40],
        })
        self.y = pd.Series([0, 0, 1, 1])

        self.model = Mock()
        self.model.feature_names_in_ = np.array(
            ["temperature", "speed"]
        )
        self.model.classes_ = np.array([0, 1])

        positive = np.array([0.1, 0.4, 0.6, 0.9])
        self.model.predict_proba.return_value = np.column_stack(
            [1 - positive, positive]
        )

    def write_threshold(self, value):
        self.threshold_path.write_text(
            json.dumps({"threshold": value}),
            encoding="utf-8",
        )

    def evaluate(self, X=None):
        if X is None:
            X = self.X

        with patch.object(
            promote_model.joblib, "load", return_value=self.model
        ):
            return promote_model.evaluate(
                "unused-model.pkl",
                self.threshold_path,
                X,
                self.y,
            )

    def test_correct_predictions_have_perfect_scores(self):
        result = self.evaluate()

        self.assertEqual(result["precision"], 1.0)
        self.assertEqual(result["recall"], 1.0)
        self.assertEqual(result["f1"], 1.0)

    def test_probability_equal_to_threshold_is_positive(self):
        self.write_threshold(0.6)

        result = self.evaluate()

        self.assertEqual(result["recall"], 1.0)

    def test_threshold_is_not_rounded(self):
        self.write_threshold(0.60000001)

        result = self.evaluate()

        self.assertEqual(result["threshold"], 0.60000001)
        self.assertEqual(result["recall"], 0.5)

    def test_feature_order_is_restored(self):
        reordered = self.X[["speed", "temperature"]]

        result = self.evaluate(reordered)

        used_X = self.model.predict_proba.call_args.args[0]
        pd.testing.assert_frame_equal(used_X, self.X)
        self.assertEqual(result["f1"], 1.0)

    def test_missing_or_extra_features_are_rejected(self):
        invalid_inputs = [
            self.X[["temperature"]],
            self.X.assign(extra=0),
        ]

        for X in invalid_inputs:
            with self.subTest(columns=list(X.columns)):
                with self.assertRaisesRegex(
                    ValueError, "features do not match"
                ):
                    self.evaluate(X)

    def test_invalid_thresholds_are_rejected(self):
        invalid_values = [
            -0.1,
            1.1,
            float("nan"),
            float("inf"),
            "invalid",
        ]

        for value in invalid_values:
            with self.subTest(value=value):
                self.write_threshold(value)

                with self.assertRaises(ValueError):
                    self.evaluate()

    def test_positive_class_column_is_found_by_label(self):
        self.model.classes_ = np.array([1, 0])
        probabilities = self.model.predict_proba.return_value
        self.model.predict_proba.return_value = probabilities[:, ::-1]

        result = self.evaluate()

        self.assertEqual(result["f1"], 1.0)

    def test_no_positive_predictions_returns_zero_scores(self):
        self.write_threshold(1.0)

        result = self.evaluate()

        self.assertEqual(result["precision"], 0.0)
        self.assertEqual(result["recall"], 0.0)
        self.assertEqual(result["f1"], 0.0)


class PromotionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.previous_directory = Path.cwd()
        os.chdir(self.temp.name)
        self.addCleanup(self.cleanup)

        Path("models/champion").mkdir(parents=True)
        Path("metrics").mkdir()
        Path("data/processed").mkdir(parents=True)

        # Placeholder files. Model evaluation is mocked in these tests.
        Path("models/model.pkl").write_bytes(b"candidate")
        Path("models/champion/model.pkl").write_bytes(b"champion")

        Path("metrics/threshold.json").write_text(
            '{"threshold": 0.5}', encoding="utf-8"
        )
        Path("models/champion/threshold.json").write_text(
            '{"threshold": 0.7}', encoding="utf-8"
        )

        self.rules = {
            "min_recall": 0.8,
            "min_f1_improvement": 0.0,
        }
        self.write_rules()

        self.data = pd.DataFrame({
            "temperature": [1.0, 2.0, 3.0, 4.0],
            "speed": [10, 20, 30, 40],
            "Machine failure": [0, 0, 1, 1],
        })
        self.write_data()

    def cleanup(self):
        os.chdir(self.previous_directory)
        self.temp.cleanup()

    def write_rules(self):
        params = {
            "min_recall": self.rules["min_recall"],
            "promotion": {
                "min_f1_improvement": self.rules["min_f1_improvement"],
            },
        }
        Path("params.yaml").write_text(
            yaml.safe_dump(params),
            encoding="utf-8",
        )

    def write_data(self):
        self.data.to_csv("data/processed/test.csv", index=False)

    @staticmethod
    def scores(precision=0.9, recall=0.9, f1=0.9, threshold=0.5):
        return {
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    def remove_champion(self):
        Path("models/champion/model.pkl").unlink()
        Path("models/champion/threshold.json").unlink()

    def run_main(self, candidate=None, champion=None, *arguments):
        if candidate is None:
            candidate = self.scores()

        if champion is None:
            champion = self.scores(
                precision=0.8,
                recall=0.8,
                f1=0.8,
                threshold=0.7,
            )

        with patch.object(
            promote_model,
            "evaluate",
            side_effect=[candidate, champion],
        ) as evaluate_mock:
            with patch.object(
                sys, "argv", ["promote_model.py", *arguments]
            ):
                with contextlib.redirect_stdout(io.StringIO()):
                    promote_model.main()

        report = json.loads(
            Path("metrics/promotion.json").read_text(encoding="utf-8")
        )
        return report, evaluate_mock

    def test_better_candidate_is_approved(self):
        report, _ = self.run_main()

        self.assertTrue(report["promotion_approved"])
        self.assertAlmostEqual(report["f1_improvement"], 0.1)

    def test_identical_candidate_is_rejected(self):
        scores = self.scores()

        report, _ = self.run_main(scores, scores)

        self.assertFalse(report["promotion_approved"])
        self.assertEqual(report["f1_improvement"], 0.0)

    def test_worse_candidate_is_rejected(self):
        candidate = self.scores(
            precision=0.8, recall=0.8, f1=0.8
        )
        champion = self.scores()

        report, _ = self.run_main(candidate, champion)

        self.assertFalse(report["promotion_approved"])
        self.assertLess(report["f1_improvement"], 0)

    def test_higher_f1_with_low_recall_is_rejected(self):
        candidate = self.scores(
            precision=1.0, recall=0.75, f1=6 / 7
        )

        report, _ = self.run_main(candidate)

        self.assertGreater(report["f1_improvement"], 0)
        self.assertFalse(report["promotion_approved"])

    def test_minimum_recall_boundary_is_accepted(self):
        candidate = self.scores(
            precision=1.0, recall=0.8, f1=8 / 9
        )

        report, _ = self.run_main(candidate)

        self.assertTrue(report["promotion_approved"])

    def test_f1_gain_must_strictly_exceed_configured_minimum(self):
        self.rules["min_f1_improvement"] = 0.25
        self.write_rules()

        candidate = self.scores(
            precision=0.6, recall=1.0, f1=0.75
        )
        champion = self.scores(
            precision=1 / 3, recall=1.0, f1=0.5
        )

        report, _ = self.run_main(candidate, champion)

        self.assertEqual(report["f1_improvement"], 0.25)
        self.assertFalse(report["promotion_approved"])

    def test_models_use_same_evaluation_data_and_own_thresholds(self):
        _, evaluate_mock = self.run_main()

        candidate_call, champion_call = evaluate_mock.call_args_list

        self.assertEqual(
            candidate_call.args[1], "metrics/threshold.json"
        )
        self.assertEqual(
            champion_call.args[1],
            Path("models/champion/threshold.json"),
        )

        pd.testing.assert_frame_equal(
            candidate_call.args[2], champion_call.args[2]
        )
        pd.testing.assert_series_equal(
            candidate_call.args[3], champion_call.args[3]
        )

    def test_bootstrap_approves_first_eligible_candidate(self):
        self.remove_champion()

        report, evaluate_mock = self.run_main(
            self.scores(), None, "--bootstrap"
        )

        self.assertTrue(report["promotion_approved"])
        self.assertIsNone(report["champion"])
        self.assertIsNone(report["f1_improvement"])
        self.assertEqual(evaluate_mock.call_count, 1)

    def test_bootstrap_rejects_low_recall(self):
        self.remove_champion()
        candidate = self.scores(
            precision=1.0, recall=0.75, f1=6 / 7
        )

        report, _ = self.run_main(
            candidate, None, "--bootstrap"
        )

        self.assertFalse(report["promotion_approved"])

    def test_missing_champion_requires_bootstrap(self):
        self.remove_champion()

        with self.assertRaisesRegex(
            FileNotFoundError, "No champion found"
        ):
            self.run_main()

    def test_existing_champion_rejects_bootstrap(self):
        with self.assertRaisesRegex(
            ValueError, "Champion already exists"
        ):
            self.run_main(None, None, "--bootstrap")

    def test_incomplete_champion_is_rejected(self):
        paths = [
            Path("models/champion/model.pkl"),
            Path("models/champion/threshold.json"),
        ]

        for path in paths:
            with self.subTest(missing=path.name):
                content = path.read_bytes()
                path.unlink()

                try:
                    with self.assertRaisesRegex(
                        FileNotFoundError,
                        "Champion model or threshold is missing",
                    ):
                        self.run_main(None, None, "--bootstrap")
                finally:
                    path.write_bytes(content)

    def test_invalid_promotion_rules_are_rejected(self):
        original = self.rules.copy()

        for name in original:
            for value in (-0.1, 1.1, float("nan"), float("inf")):
                with self.subTest(rule=name, value=value):
                    self.rules = {**original, name: value}
                    self.write_rules()

                    with self.assertRaises(ValueError):
                        self.run_main()

    def test_invalid_evaluation_labels_are_rejected(self):
        invalid_labels = [
            [0, 0, 0, 0],
            [0, 0, 2, 2],
            [0, 0, 1, np.nan],
        ]

        for labels in invalid_labels:
            with self.subTest(labels=labels):
                self.data["Machine failure"] = labels
                self.write_data()

                with self.assertRaisesRegex(
                    ValueError, "both binary classes"
                ):
                    self.run_main()

    def test_promotion_check_does_not_modify_champion_files(self):
        directory = Path("models/champion")
        before = {
            path.name: path.read_bytes()
            for path in directory.iterdir()
        }

        self.run_main()

        after = {
            path.name: path.read_bytes()
            for path in directory.iterdir()
        }
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
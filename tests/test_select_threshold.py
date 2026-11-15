import json
import sys
import unittest

import joblib
from sklearn.ensemble import RandomForestClassifier

from pipeline_helpers import SRC, Workspace, make_raw

sys.path.insert(0, str(SRC))
from features import build_features  # noqa: E402


class SelectThresholdTests(unittest.TestCase):
    def run_selection(self, min_recall):
        ws = Workspace(min_recall=min_recall)
        self.addCleanup(ws.cleanup)
        X, y = build_features(make_raw(n=1500, seed=0))
        ws.write_csv("data/processed/train.csv", X.assign(**{"Machine failure": y}))
        (ws.path / "models").mkdir()
        joblib.dump(RandomForestClassifier(n_estimators=50, class_weight="balanced", random_state=0),
                    ws.path / "models/model.pkl")
        result = ws.run("select_threshold.py")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((ws.path / "plots/threshold.png").exists())
        return json.loads((ws.path / "metrics/threshold.json").read_text())

    def test_threshold_is_a_probability(self):
        result = self.run_selection(0.80)
        self.assertGreater(result["threshold"], 0)
        self.assertLessEqual(result["threshold"], 1)

    def test_chosen_threshold_meets_min_recall(self):
        for min_recall in (0.80, 0.95):
            with self.subTest(min_recall=min_recall):
                result = self.run_selection(min_recall)
                self.assertGreaterEqual(result["cv_recall"], min_recall)


if __name__ == "__main__":
    unittest.main()
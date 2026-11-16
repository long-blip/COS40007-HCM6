import json
import sys
import unittest

import joblib
from sklearn.ensemble import RandomForestClassifier

from pipeline_helpers import SRC, Workspace, make_raw, shift

sys.path.insert(0, str(SRC))
from features import build_features  # noqa: E402


class DriftTests(unittest.TestCase):
    def setUp(self):
        self.ws = Workspace()
        self.addCleanup(self.ws.cleanup)
        self.ws.write_csv("data/ai4i2020.csv", make_raw(n=3000, seed=0))

    def drift(self, batch):
        self.ws.write_csv("data/new/batch.csv", batch)
        result = self.ws.run("detect_drift.py", "data/new/batch.csv")
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads((self.ws.path / "metrics/drift_report.json").read_text())

    def test_same_distribution_has_no_drift(self):
        report = self.drift(make_raw(n=1000, seed=1))
        self.assertFalse(report["drift_detected"], report["drifted_features"])

    def test_hotter_machines_are_detected(self):
        report = self.drift(shift(make_raw(n=1000, seed=1)))
        self.assertTrue(report["drift_detected"])
        self.assertIn("Air temperature [K]", report["drifted_features"])
        self.assertIn("Process temperature [K]", report["drifted_features"])
        self.assertNotIn("Torque [Nm]", report["drifted_features"])

    def test_type_mix_change_is_detected_by_chi_square(self):
        report = self.drift(make_raw(n=1000, seed=1).assign(Type="H"))
        self.assertTrue(report["features"]["Type"]["drift"])


class MonitorTests(unittest.TestCase):
    def setUp(self):
        self.ws = Workspace()
        self.addCleanup(self.ws.cleanup)
        reference = make_raw(n=3000, seed=0)
        self.ws.write_csv("data/ai4i2020.csv", reference)
        X, y = build_features(reference)
        model = RandomForestClassifier(n_estimators=50, class_weight="balanced", random_state=0).fit(X, y)
        (self.ws.path / "models").mkdir()
        joblib.dump(model, self.ws.path / "models/model.pkl")
        (self.ws.path / "metrics").mkdir()
        (self.ws.path / "metrics/threshold.json").write_text(json.dumps({"threshold": 0.5}))

    def monitor(self, batch, name="batch", *extra):
        path = f"data/new/{name}.csv"
        self.ws.write_csv(path, batch)
        self.assertEqual(self.ws.run("detect_drift.py", path).returncode, 0)
        result = self.ws.run("monitor.py", path, *extra)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads((self.ws.path / "metrics/monitoring_report.json").read_text())

    def test_normal_batch_is_ok(self):
        report = self.monitor(make_raw(n=1000, seed=1))
        self.assertEqual(report["status"], "OK", report["alerts"])
        self.assertFalse(report["retrain_recommended"])

    def test_drifted_batch_raises_alert_and_recommends_retraining(self):
        report = self.monitor(shift(make_raw(n=1000, seed=1)))
        self.assertEqual(report["status"], "ALERT")
        self.assertTrue(report["retrain_recommended"])
        self.assertTrue(any("drift" in alert.lower() for alert in report["alerts"]))

    def test_each_run_appends_one_log_row_and_a_plot(self):
        self.monitor(make_raw(n=500, seed=1), "batch_a")
        self.monitor(make_raw(n=500, seed=2), "batch_b")
        log = (self.ws.path / "logs/monitoring_log.csv").read_text().strip().splitlines()
        self.assertEqual(len(log), 1 + 2)  # header + two runs
        self.assertTrue((self.ws.path / "plots/drift_batch_b.png").exists())

    def test_model_path_can_point_to_the_champion(self):
        champion = self.ws.path / "models/champion"
        champion.mkdir()
        (champion / "model.pkl").write_bytes((self.ws.path / "models/model.pkl").read_bytes())
        (champion / "threshold.json").write_text(json.dumps({"threshold": 0.5}))
        report = self.monitor(make_raw(n=500, seed=1), "batch", "--model", "models/champion/model.pkl",
                              "--threshold", "models/champion/threshold.json")
        self.assertEqual(report["model"], "models/champion/model.pkl")

    def test_stops_if_drift_report_is_for_another_batch(self):
        self.ws.write_csv("data/new/a.csv", make_raw(n=300, seed=1))
        self.ws.write_csv("data/new/b.csv", make_raw(n=300, seed=2))
        self.ws.run("detect_drift.py", "data/new/a.csv")
        result = self.ws.run("monitor.py", "data/new/b.csv")
        self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
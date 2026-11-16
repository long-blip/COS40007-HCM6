import math
import sys
import unittest

from pipeline_helpers import SRC, make_raw

sys.path.insert(0, str(SRC))
from features import build_features  # noqa: E402

LEAKAGE = ["UDI", "Product ID", "Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"]


class BuildFeaturesTests(unittest.TestCase):
    def setUp(self):
        self.raw = make_raw(n=50)
        self.X, self.y = build_features(self.raw)

    def test_leakage_columns_are_removed(self):
        for column in LEAKAGE:
            self.assertNotIn(column, self.X.columns)

    def test_columns_are_snake_case_without_brackets(self):
        for column in self.X.columns:
            self.assertNotRegex(column, r"[\[\]<\s]")
        for column in ["air_temp_k", "process_temp_k", "rot_speed_rpm", "torque_nm", "tool_wear_min"]:
            self.assertIn(column, self.X.columns)

    def test_type_is_one_hot_encoded(self):
        for column in ["Type_H", "Type_L", "Type_M"]:
            self.assertIn(column, self.X.columns)
        self.assertTrue((self.X[["Type_H", "Type_L", "Type_M"]].sum(axis=1) == 1).all())

    def test_physics_features(self):
        row = self.raw.iloc[0]
        features = self.X.iloc[0]
        self.assertAlmostEqual(features["temp_diff"],
                               row["Process temperature [K]"] - row["Air temperature [K]"])
        self.assertAlmostEqual(features["power"],
                               row["Torque [Nm]"] * row["Rotational speed [rpm]"] * 2 * math.pi / 60)
        self.assertAlmostEqual(features["strain"], row["Tool wear [min]"] * row["Torque [Nm]"])

    def test_target_is_machine_failure(self):
        self.assertTrue(self.y.equals(self.raw["Machine failure"]))

    def test_batch_with_one_type_still_works(self):
        raw = self.raw.assign(Type="H")
        X, _ = build_features(raw)
        self.assertIn("Type_H", X.columns)
        self.assertNotIn("Type_L", X.columns)  # preprocess.py and monitor.py reindex to the full column list


if __name__ == "__main__":
    unittest.main()
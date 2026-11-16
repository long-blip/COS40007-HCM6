import unittest

import pandas as pd

from pipeline_helpers import Workspace, make_raw, shift


class PreprocessTests(unittest.TestCase):
    def setUp(self):
        self.ws = Workspace()
        self.addCleanup(self.ws.cleanup)
        self.raw = make_raw(n=1000)
        self.ws.write_csv("data/validated/ai4i2020.csv", self.raw)
        (self.ws.path / "data/validated/new").mkdir(parents=True)

    def preprocess(self):
        result = self.ws.run("preprocess.py")
        self.assertEqual(result.returncode, 0, result.stderr)
        train = pd.read_csv(self.ws.path / "data/processed/train.csv")
        test = pd.read_csv(self.ws.path / "data/processed/test.csv")
        return train, test

    def test_stratified_80_20_split(self):
        train, test = self.preprocess()
        self.assertEqual((len(train), len(test)), (800, 200))
        self.assertAlmostEqual(train["Machine failure"].mean(), test["Machine failure"].mean(), delta=0.01)

    def test_copied_rows_in_a_batch_are_ignored(self):
        before = self.preprocess()
        self.ws.write_csv("data/validated/new/batch_001.csv", self.raw.sample(200, random_state=1))
        after = self.preprocess()
        self.assertTrue(before[0].equals(after[0]))
        self.assertTrue(before[1].equals(after[1]))

    def test_new_rows_are_added_and_original_split_is_kept(self):
        before_train, before_test = self.preprocess()
        batch = shift(make_raw(n=200, seed=1))
        self.ws.write_csv("data/validated/new/batch_002.csv", batch)
        train, test = self.preprocess()
        self.assertEqual((len(train), len(test)), (800 + 160, 200 + 40))
        # The champion is evaluated on the old test rows, so they must not move into training
        self.assertTrue(test.iloc[:200].reset_index(drop=True).equals(before_test))
        self.assertTrue(train.iloc[:800].reset_index(drop=True).equals(before_train))

    def test_outputs_use_lf_line_endings(self):
        self.preprocess()
        content = (self.ws.path / "data/processed/train.csv").read_bytes()
        self.assertNotIn(b"\r\n", content)


if __name__ == "__main__":
    unittest.main()
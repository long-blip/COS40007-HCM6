import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("ingest_validate", ROOT / "src/ingest_validate.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
SCHEMA_PATH = ROOT / "config/ingestion_schema.yaml"


def raw_frame():
    return pd.DataFrame({
        "UDI": [1, 2, 3], "Product ID": ["M100", "L200", "H300"],
        "Type": ["M", "L", "H"], "Air temperature [K]": [302.0, 297.0, 301.0],
        "Process temperature [K]": [310.9, 308.3, 311.6],
        "Rotational speed [rpm]": [1456, 1399, 1357],
        "Torque [Nm]": [47.2, 46.4, 45.6], "Tool wear [min]": [54, 132, 137],
        "Machine failure": [0, 0, 1], "TWF": [0, 0, 1],
        "HDF": [0, 0, 0], "PWF": [0, 0, 0], "OSF": [0, 0, 0], "RNF": [0, 0, 0],
    })


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.schema, _ = MODULE.load_schema(SCHEMA_PATH)
        self.df = raw_frame()

    def issues(self, df=None, required=False, schema=None):
        data = self.df if df is None else df
        _, errors = MODULE.validate_csv(data.to_csv(index=False).encode(), schema or self.schema, required)
        return errors

    def assert_code(self, code, errors):
        self.assertIn(code, {issue["code"] for issue in errors})

    def test_valid_raw_input(self):
        self.assertEqual(self.issues(required=True), [])

    def test_shifted_batch_remains_valid(self):
        self.df["Air temperature [K]"] += 2
        self.df["Process temperature [K]"] += 2
        self.df["Tool wear [min]"] = (self.df["Tool wear [min]"] * 1.3).round()
        self.df.loc[0, "Tool wear [min]"] = 350
        self.assertEqual(self.issues(required=True), [])

    def test_unlabelled_batch_can_be_used_for_drift(self):
        self.df = self.df.drop(columns=["Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"])
        self.assertEqual(self.issues(), [])
        self.assert_code("missing_column", self.issues(required=True))

    def test_single_class_batch_is_allowed(self):
        self.df["Machine failure"] = 0
        self.assertEqual(self.issues(required=True), [])

    def test_missing_sensor_column(self):
        self.assert_code("missing_column", self.issues(self.df.drop(columns="Torque [Nm]")))

    def test_unexpected_column(self):
        self.df["unknown_sensor"] = 1
        self.assert_code("unexpected_column", self.issues())

    def test_column_order_is_not_required(self):
        self.assertEqual(self.issues(self.df[self.df.columns[::-1]]), [])

    def test_missing_values(self):
        for value in ["", "   "]:
            df = self.df.copy().astype(object)
            df.loc[0, "Torque [Nm]"] = value
            self.assert_code("missing_value", self.issues(df))

    def test_bad_numeric_and_nonfinite_values(self):
        for value in ["bad", "NaN", "inf", "-inf", "1e999"]:
            df = self.df.copy().astype(object)
            df.loc[0, "Torque [Nm]"] = value
            self.assert_code("invalid_number", self.issues(df))

    def test_temperature_must_be_positive_kelvin(self):
        for value in [0, -1]:
            self.df.loc[0, "Air temperature [K]"] = value
            self.assert_code("out_of_range", self.issues())

    def test_nonnegative_measurements(self):
        for name in ["Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]:
            df = self.df.copy()
            df.loc[0, name] = -1
            self.assert_code("out_of_range", self.issues(df))

    def test_fractional_integer_is_rejected(self):
        df = self.df.astype(object)
        df.loc[0, "Tool wear [min]"] = 1.5
        self.assert_code("invalid_integer", self.issues(df))

    def test_type_and_binary_labels(self):
        self.df.loc[0, "Type"] = "X"
        self.df.loc[0, "Machine failure"] = 2
        self.df.loc[0, "HDF"] = -1
        issues = self.issues()
        bad = {e.get("column") for e in issues if e["code"] == "invalid_category"}
        self.assertEqual(bad, {"Type", "Machine failure", "HDF"})

    def test_duplicate_udi(self):
        self.df.loc[1, "UDI"] = 1
        self.assert_code("duplicate_value", self.issues())

    def test_repeated_product_id_is_allowed(self):
        self.df.loc[1, "Product ID"] = self.df.loc[0, "Product ID"]
        self.assertEqual(self.issues(), [])

    def test_product_id_format(self):
        self.df.loc[0, "Product ID"] = "invalid"
        self.assert_code("invalid_format", self.issues())

    def test_configured_upper_bound_is_enforced(self):
        schema = copy.deepcopy(self.schema)
        schema["columns"]["Tool wear [min]"]["maximum"] = 200
        self.df.loc[0, "Tool wear [min]"] = 201
        self.assert_code("out_of_range", self.issues(schema=schema))
        self.df.loc[0, "Tool wear [min]"] = 200
        self.assertEqual(self.issues(schema=schema), [])

    def test_header_only_or_empty_input(self):
        for raw in [b"", self.df.head(0).to_csv(index=False).encode()]:
            _, issues = MODULE.validate_csv(raw, self.schema)
            self.assertTrue(issues)

    def test_duplicate_header_and_malformed_rows(self):
        for raw, expected in [
            (b"UDI,UDI\n1,2\n", "duplicate_header"),
            (b"UDI,Type\n1,M,extra\n", "row_width"),
            (b'UDI,Type\n1,"unterminated\n', "invalid_csv"),
        ]:
            _, issues = MODULE.validate_csv(raw, self.schema)
            self.assert_code(expected, issues)

    def test_utf8_bom(self):
        raw = b"\xef\xbb\xbf" + self.df.to_csv(index=False).encode()
        _, issues = MODULE.validate_csv(raw, self.schema)
        self.assertEqual(issues, [])


class IngestionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.source = self.folder / "input.csv"
        self.output = self.folder / "validated.csv"
        self.report = self.folder / "report.json"
        raw_frame().to_csv(self.source, index=False)

    def run_ingest(self, schema=SCHEMA_PATH):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = MODULE.ingest(self.source, schema, self.output, self.report, True)
        return code

    def test_valid_snapshot_is_byte_identical_and_reported(self):
        self.assertEqual(self.run_ingest(), 0)
        self.assertEqual(self.output.read_bytes(), self.source.read_bytes())
        report = json.loads(self.report.read_text())
        self.assertTrue(report["passed"])
        self.assertTrue(report["output_written"])
        self.assertEqual(report["rows"], 3)
        self.assertEqual(len(report["input_sha256"]), 64)

    def test_invalid_input_has_report_and_no_output(self):
        self.source.write_text("UDI,Type\n1,X\n")
        self.assertEqual(self.run_ingest(), 1)
        self.assertFalse(self.output.exists())
        self.assertFalse(json.loads(self.report.read_text())["passed"])

    def test_failed_input_does_not_overwrite_previous_snapshot(self):
        self.assertEqual(self.run_ingest(), 0)
        previous = self.output.read_bytes()
        self.source.write_text("UDI,Type\n1,X\n")
        self.assertEqual(self.run_ingest(), 1)
        self.assertEqual(self.output.read_bytes(), previous)
        self.assertFalse(json.loads(self.report.read_text())["output_written"])

    def test_missing_input(self):
        self.source.unlink()
        self.assertEqual(self.run_ingest(), 1)
        self.assertFalse(self.output.exists())
        self.assertEqual(json.loads(self.report.read_text())["errors"][0]["code"], "unreadable_input")

    def test_bad_schema_returns_configuration_error(self):
        schema = self.folder / "bad.yaml"
        schema.write_text("columns: []\n")
        self.assertEqual(self.run_ingest(schema), 2)
        self.assertFalse(self.output.exists())

    def test_empty_range_is_a_configuration_error(self):
        schema = yaml.safe_load(SCHEMA_PATH.read_text())
        schema["columns"]["Torque [Nm]"]["minimum"] = 5
        schema["columns"]["Torque [Nm]"]["exclusive_maximum"] = 5
        path = self.folder / "empty-range.yaml"
        path.write_text(yaml.safe_dump(schema))
        self.assertEqual(self.run_ingest(path), 2)

    def test_raw_input_cannot_be_overwritten(self):
        before = self.source.read_bytes()
        with contextlib.redirect_stderr(io.StringIO()):
            code = MODULE.ingest(self.source, SCHEMA_PATH, self.source, self.report)
        self.assertEqual(code, 2)
        self.assertEqual(self.source.read_bytes(), before)

    def test_invalid_cli_input_returns_nonzero(self):
        self.source.write_text("UDI,Type\n1,X\n")
        args = [sys.executable, str(ROOT / "src/ingest_validate.py"),
                "--input", str(self.source), "--schema", str(SCHEMA_PATH),
                "--output", str(self.output), "--report", str(self.report)]
        result = subprocess.run(args, text=True, capture_output=True)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("Validation FAILED", result.stdout)


if __name__ == "__main__":
    unittest.main()

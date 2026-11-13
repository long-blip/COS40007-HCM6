"""Validate raw AI4I CSV data and copy valid input without changing its values."""

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile

import numpy as np
import pandas as pd
import yaml


def atomic_write(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_schema(path):
    raw = Path(path).read_bytes()
    schema = yaml.safe_load(raw)
    if not isinstance(schema, dict) or not isinstance(schema.get("columns"), dict):
        raise ValueError("Schema must contain a columns mapping.")
    columns = schema["columns"]
    if not columns or schema.get("label_column") not in columns:
        raise ValueError("Schema must declare its label_column in columns.")
    if not isinstance(schema.get("allow_extra_columns", False), bool):
        raise ValueError("allow_extra_columns must be true or false.")
    fields = {"type", "required", "unique", "allowed", "pattern",
              "minimum", "maximum", "exclusive_minimum", "exclusive_maximum"}
    for name, rule in columns.items():
        if not isinstance(name, str) or not isinstance(rule, dict):
            raise ValueError("Each column needs a string name and a rule mapping.")
        if set(rule) - fields or rule.get("type") not in {"string", "integer", "number"}:
            raise ValueError(f"Unsupported validation rule for {name}.")
        for key in ["required", "unique"]:
            if key in rule and not isinstance(rule[key], bool):
                raise ValueError(f"{name}.{key} must be true or false.")
        if "allowed" in rule and (not isinstance(rule["allowed"], list) or not rule["allowed"]):
            raise ValueError(f"{name}.allowed must be a non-empty list.")
        if "pattern" in rule:
            if rule["type"] != "string":
                raise ValueError(f"{name}.pattern requires a string column.")
            re.compile(rule["pattern"])
        lower, upper = [], []
        for key in ["minimum", "maximum", "exclusive_minimum", "exclusive_maximum"]:
            if key not in rule:
                continue
            value = rule[key]
            if (rule["type"] == "string" or isinstance(value, bool)
                    or not isinstance(value, (int, float)) or not math.isfinite(value)):
                raise ValueError(f"{name}.{key} must be a finite numeric bound.")
            (lower if key in {"minimum", "exclusive_minimum"} else upper).append(value)
        if lower and upper and max(lower) > min(upper):
            raise ValueError(f"Lower bound exceeds upper bound for {name}.")
        if lower and upper and max(lower) == min(upper):
            boundary = max(lower)
            if (rule.get("exclusive_minimum") == boundary
                    or rule.get("exclusive_maximum") == boundary):
                raise ValueError(f"Configured range is empty for {name}.")
    return schema, hashlib.sha256(raw).hexdigest()


def validate_csv(raw, schema, require_labels=False):
    errors = []

    def add(code, message, column=None, mask=None):
        item = {"code": code, "message": message}
        if column is not None:
            item["column"] = column
        if mask is not None:
            positions = np.flatnonzero(np.asarray(mask, dtype=bool))
            item["count"] = int(len(positions))
            item["sample_records"] = (positions[:10] + 1).tolist()
        errors.append(item)

    try:
        text = raw.decode("utf-8-sig")
        rows = csv.reader(io.StringIO(text, newline=""), strict=True)
        header = next(rows, None)
        if not header or any(not name.strip() for name in header):
            add("empty_header", "The CSV needs non-empty column names.")
            return None, errors
        if len(header) != len(set(header)):
            add("duplicate_header", "The CSV contains duplicate column names.")
            return None, errors
        for number, row in enumerate(rows, start=2):
            if row and len(row) != len(header):
                add("row_width", f"CSV record near line {number} has the wrong field count.")
                return None, errors
        df = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
    except (UnicodeError, csv.Error, pd.errors.ParserError, pd.errors.EmptyDataError):
        add("invalid_csv", "Input must be a readable UTF-8 CSV with a valid header.")
        return None, errors

    if df.empty:
        add("empty_data", "The CSV has no data records.")
    rules = schema["columns"]
    required = {name for name, rule in rules.items() if rule.get("required", False)}
    if require_labels:
        required.add(schema["label_column"])
    for name in sorted(required - set(df.columns)):
        add("missing_column", "A required column is missing.", name)
    if not schema.get("allow_extra_columns", False):
        for name in sorted(set(df.columns) - set(rules)):
            add("unexpected_column", "Column is not declared in the raw schema.", name)

    for name, rule in rules.items():
        if name not in df:
            continue
        values = df[name]
        missing = values.str.strip().eq("")
        if missing.any():
            add("missing_value", "Column contains empty values.", name, missing)
        if rule["type"] in {"number", "integer"}:
            checked = pd.to_numeric(values, errors="coerce")
            finite = pd.Series(np.isfinite(checked.to_numpy(dtype=float)), index=df.index)
            bad_numeric = ~missing & ~finite
            if bad_numeric.any():
                add("invalid_number", "Values must be numeric and finite.", name, bad_numeric)
            valid = ~missing & finite
            if rule["type"] == "integer":
                fractional = valid & checked.mod(1).ne(0)
                if fractional.any():
                    add("invalid_integer", "Values must be whole numbers.", name, fractional)
                valid &= ~fractional
            for key, comparison in [
                ("minimum", lambda x, b: x < b),
                ("maximum", lambda x, b: x > b),
                ("exclusive_minimum", lambda x, b: x <= b),
                ("exclusive_maximum", lambda x, b: x >= b),
            ]:
                if key in rule:
                    bad = valid & comparison(checked, rule[key])
                    if bad.any():
                        add("out_of_range", f"Values violate {key}={rule[key]}.", name, bad)
        else:
            checked, valid = values, ~missing
            if "pattern" in rule:
                bad = valid & ~values.str.fullmatch(rule["pattern"], na=False)
                if bad.any():
                    add("invalid_format", "Values do not match the configured format.", name, bad)
        if "allowed" in rule:
            bad = valid & ~checked.isin(rule["allowed"])
            if bad.any():
                add("invalid_category", "Values are outside the allowed set.", name, bad)
        if rule.get("unique", False):
            bad = valid & checked.where(valid).duplicated(keep=False)
            if bad.any():
                add("duplicate_value", "Values must be unique within this input file.", name, bad)
    return df, errors


def ingest(input_path, schema_path, output_path, report_path, require_labels=False):
    paths = [Path(path).resolve() for path in [input_path, schema_path, output_path, report_path]]
    if len(paths) != len(set(paths)):
        print("Input, schema, output and report paths must be different.", file=sys.stderr)
        return 2
    report = {
        "input": str(input_path), "output": str(output_path),
        "require_labels": bool(require_labels), "passed": False,
        "output_written": False, "rows": 0, "columns": [], "errors": [],
    }
    code = 1
    try:
        schema, schema_hash = load_schema(schema_path)
        report.update(schema_version=schema.get("version"), schema_sha256=schema_hash)
    except (OSError, ValueError, TypeError, re.error, yaml.YAMLError) as error:
        report["errors"].append({"code": "invalid_schema", "message": str(error)})
        code = 2
    else:
        try:
            raw = Path(input_path).read_bytes()
        except OSError:
            report["errors"].append({"code": "unreadable_input", "message": "Input file could not be read."})
        else:
            report["input_sha256"] = hashlib.sha256(raw).hexdigest()
            df, errors = validate_csv(raw, schema, require_labels)
            report["errors"] = errors
            if df is not None:
                report.update(rows=int(len(df)), columns=df.columns.tolist())
            if not errors:
                try:
                    atomic_write(output_path, raw)
                except OSError:
                    report["errors"].append({"code": "output_write_failed", "message": "Validated output could not be saved."})
                    code = 2
                else:
                    report.update(passed=True, output_written=True)
                    code = 0
    try:
        atomic_write(report_path, (json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8"))
    except (OSError, TypeError, ValueError):
        print("Could not write the validation report.", file=sys.stderr)
        return 2
    print(f"Validation {'PASSED' if report['passed'] else 'FAILED'}: {input_path}")
    print(f"Rows: {report['rows']}; errors: {len(report['errors'])}; report: {report_path}")
    for issue in report["errors"]:
        print(f"- {issue['code']}: {issue.get('column', '')} {issue['message']}")
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Raw CSV input.")
    parser.add_argument("--schema", default="config/ingestion_schema.yaml")
    parser.add_argument("--output", required=True, help="Validated CSV snapshot.")
    parser.add_argument("--report", required=True, help="Validation report JSON.")
    parser.add_argument("--require-labels", action="store_true", help="Require Machine failure for supervised use.")
    args = parser.parse_args()
    return ingest(args.input, args.schema, args.output, args.report, args.require_labels)


if __name__ == "__main__":
    raise SystemExit(main())

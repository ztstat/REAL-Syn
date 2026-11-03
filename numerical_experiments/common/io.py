"""Small output helpers; statistical summaries live with each experiment."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import numpy as np


def jsonable(value):
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    Path(path).write_text(
        json.dumps(jsonable(value), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path, rows):
    if not rows:
        raise ValueError("No rows to save.")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def collect(config, source, repetitions, core):
    """Run paired repetitions without changing their random streams."""
    test_x, test_y = core.fixed_test_data(source)
    records = {}
    for rep in range(repetitions):
        result = core.run_repetition(config, source, rep, test_x, test_y)
        for key, value in result.items():
            if key.endswith("_rows") or key == "rows":
                records.setdefault(key, []).extend(value)
            elif key.endswith("_row"):
                records.setdefault(key, []).append(value)
        print(f"Completed repetition {rep + 1}/{repetitions}", flush=True)
    return records

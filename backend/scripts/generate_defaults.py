"""Regenerate ``backend/app/defaults.json`` from ``data/train.csv``.

The serving API only asks the user for a handful of fields, but the Milestone 1
pipeline was fitted on all 79 raw columns. This script freezes the information
needed to reconstruct a full row:

* the column order used at training time,
* the numeric/categorical split (identical to pandas ``select_dtypes``),
* a default per column (median for numeric, mode for categorical),
* the observed categories and numeric ranges, used for validation and for the
  frontend form.

Only the Python standard library is used, so it runs anywhere. Run it from the
repository root::

    python backend/scripts/generate_defaults.py
"""

from __future__ import annotations

import collections
import csv
import json
import statistics
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TRAIN_CSV = REPO_ROOT / "data" / "train.csv"
OUTPUT = REPO_ROOT / "backend" / "app" / "defaults.json"

#: Columns that are not model inputs.
DROP = {"Id", "SalePrice"}

#: How the source CSV encodes a missing value.
MISSING = {"", "NA"}


def _looks_numeric(value: str) -> bool | None:
    """True/False for a present value, None when the value is missing."""
    if value in MISSING:
        return None
    try:
        float(value)
    except ValueError:
        return False
    return True


def main() -> None:
    with TRAIN_CSV.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    columns = [name for name in rows[0] if name not in DROP]

    numeric_columns: list[str] = []
    categorical_columns: list[str] = []
    for column in columns:
        present = [flag for flag in (_looks_numeric(r[column]) for r in rows) if flag is not None]
        # An all-missing column is float64 in pandas, hence numeric.
        if not present or all(present):
            numeric_columns.append(column)
        else:
            categorical_columns.append(column)

    defaults: dict[str, Any] = {}
    numeric_ranges: dict[str, dict[str, float]] = {}
    for column in numeric_columns:
        values = [float(r[column]) for r in rows if r[column] not in MISSING]
        median = statistics.median(values) if values else 0.0
        defaults[column] = int(median) if median == int(median) else round(median, 4)
        numeric_ranges[column] = (
            {"min": min(values), "max": max(values)} if values else {"min": 0, "max": 0}
        )

    categories: dict[str, list[str]] = {}
    for column in categorical_columns:
        values = [r[column] for r in rows if r[column] not in MISSING]
        defaults[column] = collections.Counter(values).most_common(1)[0][0]
        categories[column] = sorted(set(values))

    payload = {
        "columns": columns,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "defaults": defaults,
        "categories": categories,
        "numeric_ranges": numeric_ranges,
    }

    with OUTPUT.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")

    print(
        f"Wrote {OUTPUT.relative_to(REPO_ROOT)}: "
        f"{len(columns)} columns "
        f"({len(numeric_columns)} numeric, {len(categorical_columns)} categorical)"
    )


if __name__ == "__main__":
    main()

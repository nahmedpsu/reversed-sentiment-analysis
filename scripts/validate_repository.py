#!/usr/bin/env python3
"""Validate the GitHub-facing repository without requiring restricted raw data."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "reversed-sentiment-paper-results.ipynb"
OUTPUTS = ROOT / "results" / "paper_outputs"

EXPECTED_TABLE_COLUMNS = {
    "table1_dataset.csv": {"CompanyID", "Company", "TradingDays", "Articles"},
    "table4_main_results.csv": {"n", "r", "nw_t", "nw_p", "Alignment", "Estimator"},
    "table5_walkforward.csv": {"n", "r", "Alignment", "TestYear", "TrainYears"},
    "table6_placebo.csv": {"Alignment", "Observed r", "Placebo mean", "Permutations"},
    "table12_trading.csv": {"Strategy", "Annualised return", "Sharpe", "Max drawdown"},
}

EXPECTED_FIGURES = {
    "fig3_return.png",
    "fig5_contribution.png",
    "fig_horizon.png",
    "fig_lexicon_distribution.png",
    "fig_per_firm_r.png",
    "fig_placebo.png",
    "fig_scatter.png",
    "fig_timing_alignment.png",
    "fig_walkforward.png",
    "fig_walkforward_r.png",
}


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def validate_notebook() -> None:
    try:
        data = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"cannot read notebook: {exc}")
    if data.get("nbformat") != 4:
        fail("notebook is not nbformat 4")
    cells = data.get("cells")
    if not isinstance(cells, list) or not cells:
        fail("notebook has no cells")
    if not any(cell.get("cell_type") == "markdown" for cell in cells):
        fail("notebook has no markdown cells")
    if not any(cell.get("cell_type") == "code" for cell in cells):
        fail("notebook has no code cells")


def validate_tables() -> None:
    for name, required in EXPECTED_TABLE_COLUMNS.items():
        path = OUTPUTS / name
        if not path.is_file():
            fail(f"missing required table: {path.relative_to(ROOT)}")
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = set(reader.fieldnames or [])
            if not required <= columns:
                fail(f"{name} missing columns: {sorted(required - columns)}")
            if next(reader, None) is None:
                fail(f"{name} has no data rows")


def validate_figures() -> None:
    for name in EXPECTED_FIGURES:
        path = OUTPUTS / name
        if not path.is_file():
            fail(f"missing required figure: {path.relative_to(ROOT)}")
        if path.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
            fail(f"invalid PNG signature: {path.relative_to(ROOT)}")


def validate_key_numbers() -> None:
    path = OUTPUTS / "key_numbers.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"cannot read key_numbers.json: {exc}")
    for key in ("n_articles_rows", "n_price_rows", "walkforward", "horizon", "trading"):
        if key not in data:
            fail(f"key_numbers.json missing {key!r}")


def main() -> None:
    validate_notebook()
    validate_tables()
    validate_figures()
    validate_key_numbers()
    print("Repository validation passed.")


if __name__ == "__main__":
    main()


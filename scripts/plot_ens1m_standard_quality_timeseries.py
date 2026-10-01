# -*- coding: utf-8 -*-
"""Plot ENS1M ensemble time series of Class1S+Class1 probability.

Example:
    python scripts/plot_ens1m_standard_quality_timeseries.py \
        --start-date 20260628 \
        --end-date 20260806 \
        --output-root outputs \
        --output outputs/ens1m_standard_quality_timeseries_20260628_20260806.png
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def parse_yyyymmdd(value: str) -> date:
    value = value.strip()
    for fmt in ("%Y%m%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            pass
    raise ValueError(f"invalid date: {value}; use YYYYMMDD or YYYY-MM-DD")


def iter_dates(start: date, end: date):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def load_daily_summary(output_root: Path, initial_date: date) -> dict[str, float] | None:
    path = output_root / f"ens1m_{initial_date:%Y%m%d}" / "ensemble_summary.csv"
    if not path.exists():
        print(f"[SKIP] missing {path}")
        return None

    df = pd.read_csv(path)
    if df.empty:
        print(f"[SKIP] empty {path}")
        return None

    if "standard_quality_percent" not in df.columns:
        if {"class_1S_probability", "class_1_probability"}.issubset(df.columns):
            df["standard_quality_percent"] = (
                df["class_1S_probability"] + df["class_1_probability"]
            ) * 100.0
        else:
            raise ValueError(f"{path} has no standard quality columns")

    values = df["standard_quality_percent"].dropna()
    if values.empty:
        print(f"[SKIP] no valid standard_quality_percent in {path}")
        return None

    return {
        "date": pd.Timestamp(initial_date),
        "count": int(values.count()),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)),
        "min": float(values.min()),
        "p05": float(values.quantile(0.05)),
        "p25": float(values.quantile(0.25)),
        "p50": float(values.quantile(0.50)),
        "p75": float(values.quantile(0.75)),
        "p95": float(values.quantile(0.95)),
        "max": float(values.max()),
    }


def plot_timeseries(summary: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))

    x = summary["date"]

    ax.fill_between(
        x,
        summary["p05"],
        summary["p95"],
        color="#9ecae1",
        alpha=0.35,
        linewidth=0,
        label="5-95%",
    )
    ax.fill_between(
        x,
        summary["p25"],
        summary["p75"],
        color="#3182bd",
        alpha=0.28,
        linewidth=0,
        label="25-75%",
    )
    ax.plot(x, summary["p50"], color="#08519c", linewidth=2.0, label="median")
    ax.plot(x, summary["mean"], color="#de2d26", linewidth=1.8, label="mean")
    ax.plot(x, summary["min"], color="#6baed6", linewidth=0.8, alpha=0.6, label="min/max")
    ax.plot(x, summary["max"], color="#6baed6", linewidth=0.8, alpha=0.6)

    ax.set_ylim(0, 100)
    ax.set_ylabel("Class1S + Class1 probability (%)")
    ax.set_xlabel("ENS1M initial date")
    ax.set_title("ENS1M ensemble spread of standard quality probability")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best", frameon=False)

    fig.autofmt_xdate()
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="ENS1M初期日ごとのClass1S+Class1確率のアンサンブル分布を時系列図化します。"
    )
    parser.add_argument("--start-date", default="20260628", help="開始日。YYYYMMDD または YYYY-MM-DD。")
    parser.add_argument("--end-date", default="20260806", help="終了日。YYYYMMDD または YYYY-MM-DD。")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument(
        "--output",
        default="outputs/ens1m_standard_quality_timeseries_20260628_20260806.png",
    )
    parser.add_argument(
        "--summary-output",
        default="outputs/ens1m_standard_quality_timeseries_20260628_20260806.csv",
    )
    args = parser.parse_args()

    start = parse_yyyymmdd(args.start_date)
    end = parse_yyyymmdd(args.end_date)
    if end < start:
        raise ValueError("--end-date must be on or after --start-date")

    output_root = Path(args.output_root)
    rows = []

    for initial_date in iter_dates(start, end):
        row = load_daily_summary(output_root, initial_date)
        if row is not None:
            rows.append(row)

    if not rows:
        raise RuntimeError("No ensemble_summary.csv files were loaded.")

    summary = pd.DataFrame(rows).sort_values("date")
    summary_output = Path(args.summary_output)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_output, index=False)

    plot_timeseries(summary, Path(args.output))

    print(f"Loaded {len(summary)} dates.")
    print(f"Wrote {summary_output}")
    print(f"Wrote {args.output}")
    print(summary[["date", "count", "mean", "p05", "p50", "p95"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
# -*- coding: utf-8 -*-
"""Run rice_quality_ens1m_members.py for a continuous range of ENS1M initial dates.

Example:
    python scripts/run_rice_quality_ens1m_period.py \
        --start-date 20260501 \
        --end-date 20260806 \
        --output-root outputs
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import date, datetime, timedelta
from pathlib import Path


DEFAULT_ENS1M_DIR = Path("/mnt/e/data/ENS1M")
DEFAULT_CHILD_SCRIPT = Path(__file__).with_name("rice_quality_ens1m_members.py")
REQUIRED_ELEMENTS = ("TMP", "GSR", "APCP")


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


def ens1m_daily_file(ens1m_dir: Path, initial_date: date, element: str) -> Path:
    return (
        ens1m_dir
        / str(initial_date.year)
        / element
        / f"ENS1M_daily_{initial_date:%Y%m%d}_{element}.nc"
    )


def missing_ens1m_files(ens1m_dir: Path, initial_date: date) -> list[Path]:
    return [
        ens1m_daily_file(ens1m_dir, initial_date, element)
        for element in REQUIRED_ELEMENTS
        if not ens1m_daily_file(ens1m_dir, initial_date, element).exists()
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="指定期間のENS1M初期日ごとにタンパク推定を連続実行します。")
    parser.add_argument("--start-date", required=True, help="開始日。YYYYMMDD または YYYY-MM-DD。")
    parser.add_argument("--end-date", required=True, help="終了日。YYYYMMDD または YYYY-MM-DD。")
    parser.add_argument("--ens1m-dir", default=str(DEFAULT_ENS1M_DIR))
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--child-script", default=str(DEFAULT_CHILD_SCRIPT))
    parser.add_argument("--python", default=sys.executable, help="子スクリプト実行に使うPython。既定は現在のPython。")
    parser.add_argument("--points-csv", help="子スクリプトへ渡す地点CSV。")
    parser.add_argument("--amd-url", help="子スクリプトへ渡すAMD URL。")
    parser.add_argument("--force", action="store_true", help="summaryが既にあっても再実行します。")
    parser.add_argument("--stop-on-error", action="store_true", help="1日でも失敗したらそこで停止します。")
    args = parser.parse_args()

    start = parse_yyyymmdd(args.start_date)
    end = parse_yyyymmdd(args.end_date)
    if end < start:
        raise ValueError("--end-date must be on or after --start-date")

    ens1m_dir = Path(args.ens1m_dir)
    output_root = Path(args.output_root)
    child_script = Path(args.child_script)

    output_root.mkdir(parents=True, exist_ok=True)

    failures: list[tuple[date, int | str]] = []
    skipped: list[date] = []
    completed: list[date] = []

    for initial_date in iter_dates(start, end):
        ymd = f"{initial_date:%Y%m%d}"
        iso = initial_date.isoformat()
        output_dir = output_root / f"ens1m_{ymd}"
        summary_path = output_dir / "ensemble_summary.csv"

        missing = missing_ens1m_files(ens1m_dir, initial_date)
        if missing:
            print(f"[SKIP] {ymd}: missing ENS1M files")
            for path in missing:
                print(f"       {path}")
            skipped.append(initial_date)
            continue

        if summary_path.exists() and not args.force:
            print(f"[SKIP] {ymd}: {summary_path} already exists")
            skipped.append(initial_date)
            continue

        command = [
            args.python,
            str(child_script),
            "--ens1m-start-date",
            iso,
            "--ens1m-dir",
            str(ens1m_dir),
            "--output-dir",
            str(output_dir),
        ]

        if args.points_csv:
            command.extend(["--points-csv", args.points_csv])
        if args.amd_url:
            command.extend(["--amd-url", args.amd_url])

        print(f"[RUN] {ymd}")
        print("      " + " ".join(command))

        result = subprocess.run(command)
        if result.returncode == 0:
            completed.append(initial_date)
            print(f"[OK]  {ymd}")
        else:
            failures.append((initial_date, result.returncode))
            print(f"[NG]  {ymd}: return code {result.returncode}")
            if args.stop_on_error:
                break

    print("\nBatch summary")
    print(f"completed: {len(completed)}")
    print(f"skipped:   {len(skipped)}")
    print(f"failed:    {len(failures)}")

    if failures:
        print("\nFailed dates")
        for failed_date, code in failures:
            print(f"{failed_date:%Y%m%d}: {code}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
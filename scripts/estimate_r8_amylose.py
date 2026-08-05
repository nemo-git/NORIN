#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Estimate amylose content for the R8 ゆめぴりか survey points."""

from __future__ import annotations

import argparse
from pathlib import Path

from rice_quality import (
    estimate_amylose_for_points,
    fake_weather_provider,
    fetch_amd_point_weather,
    load_r8_survey_points_xlsx,
)


DEFAULT_WORKBOOK = Path(__file__).resolve().parents[1] / "rice_data" / "R8年産ゆめぴりか定点観測集計表.xlsx"


def main() -> int:
    parser = argparse.ArgumentParser(description="R8調査表からアミロース含有率を推定します。")
    parser.add_argument("--workbook", default=str(DEFAULT_WORKBOOK))
    parser.add_argument("--target-year", type=int, default=2026)
    parser.add_argument("--use-amd", action="store_true", help="AMD_Tools4.pyで実データを取得します。")
    parser.add_argument("--amd-url", default="https://amd.rd.naro.go.jp/opendap/AMD/")
    parser.add_argument("--keep-original-year", action="store_true", help="移植期の年をR8年に補正しません。")
    parser.add_argument("--output", default="scripts/r8_amylose_estimates.csv")
    args = parser.parse_args()

    workbook = _resolve_workbook(Path(args.workbook))
    points = load_r8_survey_points_xlsx(
        workbook,
        target_year=args.target_year,
        normalize_transplant_year=not args.keep_original_year,
    )
    normalized = points[points["transplant_year_normalized"]]
    if len(normalized):
        print("Warning: transplant year normalized to target year for these points:")
        print(normalized[["point_id", "survey_no", "ja", "district", "transplant_date_original", "transplant_date"]].to_string(index=False))

    if args.use_amd:
        provider = lambda p, s, e, elements: fetch_amd_point_weather(p, s, e, elements, amd_url=args.amd_url)
    else:
        provider = fake_weather_provider

    estimates = estimate_amylose_for_points(points, provider)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    estimates.to_csv(output, index=False)

    columns = [
        "point_id",
        "survey_no",
        "ja",
        "district",
        "lat",
        "lon",
        "transplant_date",
        "heading_date",
        "A108_corrected_TMP_mea",
        "amylose_percent",
        "amylose_over_19_probability_percent",
    ]
    print(f"Calculated {len(estimates)} R8 survey points from {workbook}")
    print(estimates[columns].to_string(index=False))
    print(f"\nWrote {output}")
    return 0


def _resolve_workbook(path: Path) -> Path:
    if path.exists():
        return path
    parent = path.parent
    if parent.exists():
        matches = sorted(parent.glob("R8年産ゆめ*.xlsx"))
        if matches:
            return matches[0]
    raise FileNotFoundError(path)


if __name__ == "__main__":
    raise SystemExit(main())

# -*- coding: utf-8 -*-
"""Estimate rice protein for each ENS1M ensemble member.

Example:
    python scripts/rice_quality_ens1m_members.py \
        --ens1m-start-date 2026-08-01 \
        --output-dir outputs/ens1m_20260801
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from rice_quality import (
    AMD_ELEMENTS,
    DEFAULT_POINTS_CSV,
    PointWeather,
    estimate_protein_distribution,
    estimate_quality_for_points,
    fetch_amd_point_weather,
    load_points_csv,
    parse_date,
)


DEFAULT_ENS1M_DIR = Path("/mnt/e/data/ENS1M")

ENS1M_MEMBER_VARIABLES = {
    "TMP_mea": ("TMP", "TMP_daymean"),
    "GSR": ("GSR", "GSR"),
    "APCP": ("APCP", "APCP_daysum"),
}

RATIO_BIAS_ELEMENTS = {"GSR", "APCP"}


class HybridEns1mMemberProvider:
    def __init__(
        self,
        points: pd.DataFrame,
        switch_date: date,
        ens1m_dir: Path,
        member: int,
        amd_url: str,
        amd_cache: dict,
        ratio_eps: float = 1.0e-6,
    ):
        self.points = points
        self.switch_date = switch_date
        self.ens1m_dir = ens1m_dir
        self.member = member
        self.amd_url = amd_url
        self.ratio_eps = ratio_eps
        self._amd_cache = amd_cache

    def __call__(
        self,
        points: pd.DataFrame,
        start_date: date,
        end_date: date,
        elements: Sequence[str],
    ) -> list[PointWeather]:
        start = parse_date(start_date)
        end = parse_date(end_date)
        switch = self.switch_date

        # 全期間ぶんAMDを取得する。ENS1M前は実績、ENS1M後は平年値を使う。
        amd = self._fetch_amd(points, start, end, elements)
        amd_by_point = {w.point_id: w for w in amd}

        if end < switch:
            return amd

        ens = self._fetch_ens_member(points, switch, end, elements)
        ens_by_point = {w.point_id: w for w in ens}

        result = []
        switch_ts = pd.Timestamp(switch)
        end_ts = pd.Timestamp(end)

        for _, row in points.iterrows():
            point_id = str(row["point_id"])
            frame = pd.DataFrame(index=pd.DatetimeIndex([]))
            amd_frame = amd_by_point[point_id].data
            ens_frame = ens_by_point[point_id].data

            for element in elements:
                # 1. ENS1M開始日前はAMD
                if start < switch:
                    actual = amd_frame.loc[
                        (amd_frame.index >= pd.Timestamp(start))
                        & (amd_frame.index < switch_ts),
                        element,
                    ]
                    frame = frame.reindex(frame.index.union(actual.index))
                    frame.loc[actual.index, element] = actual

                ens_series = (
                    ens_frame[element].dropna()
                    if element in ens_frame.columns
                    else pd.Series(dtype=float)
                )

                # 2. ENS1Mでカバーできる期間は、切替日でバイアス補正して使用
                if not ens_series.empty:
                    if switch_ts not in ens_series.index:
                        raise ValueError(f"ENS1M {element} is missing switch date {switch}")

                    amd_anchor = float(amd_frame.loc[switch_ts, element])
                    ens_anchor = float(ens_series.loc[switch_ts])

                    if element in RATIO_BIAS_ELEMENTS:
                        factor = 1.0 if abs(ens_anchor) <= self.ratio_eps else amd_anchor / ens_anchor
                        corrected = ens_series * factor
                    else:
                        corrected = ens_series + (amd_anchor - ens_anchor)

                    frame = frame.reindex(frame.index.union(corrected.index))
                    frame.loc[corrected.index, element] = corrected

                    fallback_start = pd.Timestamp(ens_series.index.max()) + pd.Timedelta(days=1)
                else:
                    fallback_start = switch_ts

                # 3. ENS1M最終日より後はAMD、予報期間外なら平年値を期待
                if fallback_start <= end_ts:
                    fallback = amd_frame.loc[
                        (amd_frame.index >= fallback_start)
                        & (amd_frame.index <= end_ts),
                        element,
                    ]
                    frame = frame.reindex(frame.index.union(fallback.index))
                    frame.loc[fallback.index, element] = fallback

            result.append(PointWeather(point_id, float(row["lat"]), float(row["lon"]), frame))

        return result


    def _fetch_amd(self, points, start, end, elements):
        key = (start, end, tuple(elements))
        if key not in self._amd_cache:
            self._amd_cache[key] = fetch_amd_point_weather(
                points,
                start,
                end,
                elements,
                amd_url=self.amd_url,
            )
        return self._amd_cache[key]

    def _fetch_ens_member(self, points, start, end, elements):
        frames = {
            str(row["point_id"]): pd.DataFrame(index=pd.DatetimeIndex([]))
            for _, row in points.iterrows()
        }

        for element in elements:
            ens_element, variable = ENS1M_MEMBER_VARIABLES[element]
            path = (
                self.ens1m_dir
                / str(self.switch_date.year)
                / ens_element
                / f"ENS1M_daily_{self.switch_date:%Y%m%d}_{ens_element}.nc"
            )
            if not path.exists():
                raise FileNotFoundError(path)

            with xr.open_dataset(path) as ds:
                data = ds[variable].sel(
                    ensemble=self.member,
                    time=slice(pd.Timestamp(start), pd.Timestamp(end)),
                )
                data = _convert_units(element, data)

                lats = np.asarray(ds["latitude"].values, dtype=float)
                lons = np.asarray(ds["longitude"].values, dtype=float)
                times = pd.to_datetime(data["time"].values).normalize()
                values = np.asarray(data.values, dtype=float)

                for _, row in points.iterrows():
                    point_id = str(row["point_id"])
                    lat_i = int(np.nanargmin(np.abs(lats - float(row["lat"]))))
                    lon_i = int(np.nanargmin(np.abs(lons - float(row["lon"]))))
                    series = pd.Series(values[:, lat_i, lon_i], index=times)

                    frames[point_id] = frames[point_id].reindex(
                        frames[point_id].index.union(series.index)
                    )
                    frames[point_id].loc[series.index, element] = series

        return [
            PointWeather(str(row["point_id"]), float(row["lat"]), float(row["lon"]), frames[str(row["point_id"])])
            for _, row in points.iterrows()
        ]


def _convert_units(element: str, data):
    units = str(data.attrs.get("units", "")).lower()
    if element == "TMP_mea" and (units in {"k", "kelvin"} or float(data.mean(skipna=True)) > 100.0):
        return data - 273.15
    return data


def list_members(ens1m_dir: Path, switch_date: date) -> list[int]:
    path = ens1m_dir / str(switch_date.year) / "TMP" / f"ENS1M_daily_{switch_date:%Y%m%d}_TMP.nc"
    with xr.open_dataset(path) as ds:
        return [int(v) for v in ds["ensemble"].values]


def plot_distribution(summary: pd.DataFrame, output_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(summary["protein_mean_all_points"], bins=12, edgecolor="black", alpha=0.75)
    ax.axvline(summary["protein_mean_all_points"].mean(), color="red", linewidth=2, label="mean")
    ax.set_xlabel("Predicted protein mean (%)")
    ax.set_ylabel("Number of ensemble members")
    ax.set_title("ENS1M member distribution of predicted protein")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_png, dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="ENS1M各メンバでタンパク質含量を推定します。")
    parser.add_argument("--points-csv", default=str(DEFAULT_POINTS_CSV))
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--ens1m-start-date", required=True)
    parser.add_argument("--ens1m-dir", default=str(DEFAULT_ENS1M_DIR))
    parser.add_argument("--amd-url", default="https://amd.rd.naro.go.jp/opendap/AMD/")
    parser.add_argument("--output-dir", default="outputs/rice_quality_ens1m_members")
    args = parser.parse_args()

    points = load_points_csv(args.points_csv)
    switch_date = parse_date(args.ens1m_start_date)
    ens1m_dir = Path(args.ens1m_dir)
    output_dir = Path(args.output_dir)
    member_dir = output_dir / "members"
    member_dir.mkdir(parents=True, exist_ok=True)

    members = list_members(ens1m_dir, switch_date)
    summaries = []
    amd_cache = {}
    
    
    for member in members:
        
        point_path = member_dir / f"rice_quality_member_{member:+03d}.csv"
        
        if point_path.exists():
            print(f"Loading existing member {member}; {point_path} already exists.")
            point_results = pd.read_csv(point_path)
            aggregate = estimate_protein_distribution(point_results)
            summaries.append(
                {
                    "member": member,
                    **aggregate,
                    "protein_mean_min_point": float(point_results["protein_mean"].min()),
                    "protein_mean_max_point": float(point_results["protein_mean"].max()),
                }
            )
            continue
        
        print(f"Running member {member}...")
        provider = HybridEns1mMemberProvider(
            points=points,
            switch_date=switch_date,
            ens1m_dir=ens1m_dir,
            member=member,
            amd_url=args.amd_url,
            amd_cache=amd_cache,
        )
        point_results, aggregate = estimate_quality_for_points(
            points,
            provider,
            year=args.year,
            include_amylose=False,
        )

        point_path = member_dir / f"rice_quality_member_{member:+03d}.csv"
        point_results.to_csv(point_path, index=False)

        summaries.append(
            {
                "member": member,
                **aggregate,
                "protein_mean_min_point": float(point_results["protein_mean"].min()),
                "protein_mean_max_point": float(point_results["protein_mean"].max()),
            }
        )

    summary = pd.DataFrame(summaries).sort_values("member")
    summary_path = output_dir / "ensemble_summary.csv"
    summary.to_csv(summary_path, index=False)

    stats = summary["protein_mean_all_points"].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
    stats_path = output_dir / "ensemble_distribution_stats.csv"
    stats.to_csv(stats_path, header=["protein_mean_all_points"])

    plot_distribution(summary, output_dir / "protein_mean_distribution.png")

    print("\nEnsemble summary")
    print(stats.to_string())
    print(f"\nWrote {summary_path}")
    print(f"Wrote {stats_path}")
    print(f"Wrote {output_dir / 'protein_mean_distribution.png'}")
    return 0
    def __init__(
        self,
        points,
        switch_date,
        ens1m_dir,
        member,
        amd_url,
        amd_cache,
        ratio_eps=1.0e-6,
    ):
        self.points = points
        self.switch_date = switch_date
        self.ens1m_dir = ens1m_dir
        self.member = member
        self.amd_url = amd_url
        self.ratio_eps = ratio_eps
        self._amd_cache = amd_cache



if __name__ == "__main__":
    raise SystemExit(main())
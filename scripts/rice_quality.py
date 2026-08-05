# -*- coding: utf-8 -*-
"""Rice quality estimation from Mesh Agro-Meteorological Data.

The calculation functions are separated from the AMD data access path so they
can be tested without Oracle authentication.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence
from zipfile import ZipFile

import numpy as np
import pandas as pd


AMD_ELEMENTS = ("TMP_mea", "GSR", "APCP")
DEFAULT_POINTS_CSV = Path(__file__).with_name("amydas2_409points.csv")
R8_SURVEY_SHEET_NAME = "R8調査表"


@dataclass(frozen=True)
class PointWeather:
    point_id: str
    lat: float
    lon: float
    data: pd.DataFrame


def parse_date(value: str | date | pd.Timestamp) -> date:
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.strptime(str(value), "%Y-%m-%d").date()


def dvr(mean_temperature: float) -> float:
    """Daily developmental rate from daily mean temperature."""
    return (1.0 / (1.0 + math.exp(-0.4410648 * (mean_temperature - 14.33413)))) / 54.19644


def estimate_heading_date(
    mean_temperatures: pd.Series,
    transplant_date: str | date,
    threshold: float = 1.0,
) -> date:
    """Return the first date where accumulated DVR exceeds threshold."""
    transplant = parse_date(transplant_date)
    values = _slice_by_date(mean_temperatures, transplant, None)
    total = 0.0
    for ts, temp in values.items():
        if pd.isna(temp):
            continue
        total += dvr(float(temp))
        if total > threshold:
            return _date_from_index(ts)
    raise ValueError(
        f"DVI did not exceed {threshold} from {transplant}; extend the weather period."
    )


def estimate_amylose(
    mean_temperatures: pd.Series,
    heading_date: str | date,
) -> dict[str, float]:
    """Estimate amylose content and probability of amylose > 19%."""
    heading = parse_date(heading_date)
    a106 = _mean_between(mean_temperatures, heading - timedelta(days=5), heading + timedelta(days=4))
    a107 = _mean_between(mean_temperatures, heading + timedelta(days=5), heading + timedelta(days=24))
    a108 = a107 + (a106 - 21.0) * 10.0 / 30.0 if a106 > 21.0 else a107

    amylose = round(37.188 - 0.9504 * a108, 1)
    probability = round(
        -0.680462687 + 100.9949657 / (1.0 + (a108 / 19.12940973) ** 30.22541297),
        1,
    )
    probability = max(min(probability, 100.0), 0.0)
    return {
        "A106_heading_minus5_to_plus4_TMP_mea": a106,
        "A107_heading_plus5_to_plus24_TMP_mea": a107,
        "A108_corrected_TMP_mea": a108,
        "amylose_percent": amylose,
        "amylose_over_19_probability_percent": probability,
    }


def estimate_protein_at_point(
    weather: pd.DataFrame,
    heading_date: str | date,
    mode: int = 1,
    correction: float = 0.0,
) -> dict[str, float | date | int]:
    """Estimate protein mean and standard deviation at one point."""
    heading = parse_date(heading_date)
    panicle_formation = heading - timedelta(days=25)
    previous_start = panicle_formation
    previous_end = panicle_formation + timedelta(days=13)

    gsr_panicle_20 = _mean_between(weather["GSR"], panicle_formation, panicle_formation + timedelta(days=20))
    apcp_panicle_20 = _mean_between(weather["APCP"], panicle_formation, panicle_formation + timedelta(days=20))
    tmp_mea_previous = _mean_between(weather["TMP_mea"], previous_start, previous_end)
    gsr_heading_5_20 = _mean_between(weather["GSR"], heading + timedelta(days=5), heading + timedelta(days=20))

    mean = (
        11.4700256740705
        - 0.0543439419443871 * gsr_panicle_20
        - 0.221540550371125 * gsr_heading_5_20
        + correction
    )
    if mode == 1:
        stddev = 0.54
    elif mode == 2:
        stddev = (
            0.773412167082859
            - 0.0131900737102685 * tmp_mea_previous
            + 0.00844337108509267 * apcp_panicle_20
        )
    else:
        raise ValueError("mode must be 1 or 2")
    if stddev <= 0:
        raise ValueError(f"protein stddev must be positive, got {stddev}")

    return {
        "heading_date": heading,
        "panicle_formation_date": panicle_formation,
        "mode": mode,
        "correction": correction,
        "GSR_panicle_to_plus20_mean": gsr_panicle_20,
        "APCP_panicle_to_plus20_mean": apcp_panicle_20,
        "TMP_mea_previous_period_mean": tmp_mea_previous,
        "GSR_heading_plus5_to_plus20_mean": gsr_heading_5_20,
        "protein_mean": mean,
        "protein_stddev": stddev,
    }


def estimate_protein_distribution(point_results: pd.DataFrame) -> dict[str, float]:
    """Aggregate point estimates and calculate quality class probabilities."""
    protein_mean = float(point_results["protein_mean"].mean())
    protein_stddev = float(point_results["protein_stddev"].mean())
    first_s = _normal_cdf(6.8, protein_mean, protein_stddev)
    first = _normal_cdf(7.4, protein_mean, protein_stddev) - _normal_cdf(6.8, protein_mean, protein_stddev)
    second = _normal_cdf(7.9, protein_mean, protein_stddev) - _normal_cdf(7.4, protein_mean, protein_stddev)
    third = 1.0 - _normal_cdf(7.9, protein_mean, protein_stddev)
    standard_quality = round((first_s + first) * 100.0, 1)
    return {
        "protein_mean_all_points": protein_mean,
        "protein_stddev_all_points": protein_stddev,
        "class_1S_probability": first_s,
        "class_1_probability": first,
        "class_2_probability": second,
        "class_3_probability": third,
        "standard_quality_percent": standard_quality,
    }


def load_points_csv(path: str | Path) -> pd.DataFrame:
    """Load point definitions.

    Required columns: lat, lon. Optional columns: point_id, transplant_date,
    heading_date, mode, correction. Headerless 3-column CSV is interpreted as
    lat, lon, region.
    """
    path = Path(path)
    points = pd.read_csv(path)
    if {"lat", "lon"}.isdisjoint(points.columns) and len(points.columns) >= 2:
        points = pd.read_csv(path, header=None)
        names = ["lat", "lon", "region"]
        extra = [f"extra_{i}" for i in range(max(0, len(points.columns) - len(names)))]
        points.columns = names[: len(points.columns)] + extra
    aliases = {
        "latitude": "lat",
        "緯度": "lat",
        "longitude": "lon",
        "経度": "lon",
        "id": "point_id",
        "地点ID": "point_id",
        "移植日": "transplant_date",
        "出穂日": "heading_date",
        "実測出穂日": "heading_date",
        "MODE": "mode",
        "補正値": "correction",
        "g_タンパク質含有率補正値": "correction",
    }
    points = points.rename(columns={c: aliases.get(c, c) for c in points.columns})
    missing = {"lat", "lon"} - set(points.columns)
    if missing:
        raise ValueError(f"points CSV is missing required column(s): {', '.join(sorted(missing))}")
    if "point_id" not in points:
        points["point_id"] = [f"P{i + 1:04d}" for i in range(len(points))]
    points["lat"] = points["lat"].astype(float)
    points["lon"] = points["lon"].astype(float)
    return points


def load_r8_survey_points_xlsx(
    path: str | Path,
    target_year: int = 2026,
    normalize_transplant_year: bool = True,
) -> pd.DataFrame:
    """Extract 40 point definitions from the R8調査表 sheet.

    The workbook is parsed directly as XLSX XML to avoid requiring openpyxl.
    Columns used from R8調査表 are: No, JA, district, lat, lon, transplant date.
    """
    path = Path(path)
    sheet_path = _xlsx_sheet_path(path, R8_SURVEY_SHEET_NAME)
    rows = _xlsx_sheet_rows(path, sheet_path)
    extracted: list[dict[str, object]] = []
    for excel_row, values in rows:
        lat = _float_or_none(values.get(7))
        lon = _float_or_none(values.get(8))
        transplant_serial = _float_or_none(values.get(10))
        if lat is None or lon is None or transplant_serial is None:
            continue
        transplant_original = _excel_serial_date(transplant_serial)
        transplant = transplant_original
        normalized = False
        if normalize_transplant_year and transplant.year != target_year:
            transplant = date(target_year, transplant.month, transplant.day)
            normalized = True
        index = len(extracted) + 1
        extracted.append(
            {
                "point_id": f"R8_{index:02d}",
                "survey_no": index,
                "survey_no_raw": values.get(1, ""),
                "ja": values.get(3, ""),
                "district": values.get(4, ""),
                "lat": lat,
                "lon": lon,
                "transplant_date": transplant.isoformat(),
                "transplant_date_original": transplant_original.isoformat(),
                "transplant_year_normalized": normalized,
                "excel_row": excel_row,
            }
        )
    if len(extracted) != 40:
        raise ValueError(f"expected 40 R8 survey points, got {len(extracted)}")
    return pd.DataFrame(extracted)


def estimate_amylose_for_points(
    points: pd.DataFrame,
    weather_provider: Callable[[pd.DataFrame, date, date, Sequence[str]], list[PointWeather]],
    max_dvi_days: int = 140,
) -> pd.DataFrame:
    """Estimate heading date and amylose content for each point."""
    points = points.copy()
    if "point_id" not in points:
        points["point_id"] = [f"P{i + 1:04d}" for i in range(len(points))]
    if "transplant_date" not in points:
        raise ValueError("points must have transplant_date")

    first_transplant = min(parse_date(v) for v in points["transplant_date"])
    dvi_end = max(parse_date(v) for v in points["transplant_date"]) + timedelta(days=max_dvi_days)
    weather = weather_provider(points, first_transplant, dvi_end, ("TMP_mea",))
    weather_by_point = {w.point_id: w for w in weather}

    rows: list[dict[str, object]] = []
    for _, point in points.iterrows():
        point_id = str(point["point_id"])
        series = weather_by_point[point_id].data["TMP_mea"]
        heading = estimate_heading_date(series, point["transplant_date"])
        amylose = estimate_amylose(series, heading)
        rows.append(
            {
                **point.to_dict(),
                "heading_date": heading,
                **amylose,
            }
        )
    return pd.DataFrame(rows)


def fetch_amd_point_weather(
    points: pd.DataFrame,
    start_date: str | date,
    end_date: str | date,
    elements: Sequence[str] = AMD_ELEMENTS,
    amd_url: str = "https://amd.rd.naro.go.jp/opendap/AMD/",
    cli: bool = False,
    padding_degree: float = 0.02,
) -> list[PointWeather]:
    """Fetch AMD data once per element and extract nearest grid cells for points."""
    import AMD_Tools4 as AMD

    start = parse_date(start_date)
    end = parse_date(end_date)
    lat_min = float(points["lat"].min()) - padding_degree
    lat_max = float(points["lat"].max()) + padding_degree
    lon_min = float(points["lon"].min()) - padding_degree
    lon_max = float(points["lon"].max()) + padding_degree
    timedomain = [start.isoformat(), end.isoformat()]
    lalodomain = [lat_min, lat_max, lon_min, lon_max]

    fetched: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for element in elements:
        values, times, lats, lons = AMD.GetMetData(element, timedomain, lalodomain, cli=cli)
        fetched[element] = (np.asarray(values, dtype=float), np.asarray(times), np.asarray(lats), np.asarray(lons))

    result: list[PointWeather] = []
    for _, row in points.iterrows():
        frame = pd.DataFrame()
        time_index: pd.DatetimeIndex | None = None
        for element, (values, times, lats, lons) in fetched.items():
            lat_i = int(np.nanargmin(np.abs(lats - float(row["lat"]))))
            lon_i = int(np.nanargmin(np.abs(lons - float(row["lon"]))))
            if time_index is None:
                time_index = pd.to_datetime(times).normalize()
                frame.index = time_index
            frame[element] = values[:, lat_i, lon_i]
        result.append(PointWeather(str(row["point_id"]), float(row["lat"]), float(row["lon"]), frame))
    return result


def estimate_quality_for_points(
    points: pd.DataFrame,
    weather_provider: Callable[[pd.DataFrame, date, date, Sequence[str]], list[PointWeather]],
    year: int,
    default_transplant_date: str = "05-25",
    default_mode: int = 1,
    default_correction: float = 0.0,
    max_dvi_days: int = 140,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Run heading, amylose, and protein calculations for all points."""
    points = points.copy()
    if "transplant_date" not in points:
        points["transplant_date"] = f"{year}-{default_transplant_date}"
    points["transplant_date"] = points["transplant_date"].fillna(f"{year}-{default_transplant_date}")

    first_transplant = min(parse_date(v) for v in points["transplant_date"])
    dvi_end = max(parse_date(v) for v in points["transplant_date"]) + timedelta(days=max_dvi_days)
    dvi_weather = weather_provider(points, first_transplant, dvi_end, ("TMP_mea",))

    heading_dates: dict[str, date] = {}
    dvi_by_point = {w.point_id: w.data["TMP_mea"] for w in dvi_weather}
    for _, row in points.iterrows():
        point_id = str(row["point_id"])
        if "heading_date" in points and pd.notna(row.get("heading_date")):
            heading_dates[point_id] = parse_date(row["heading_date"])
        else:
            heading_dates[point_id] = estimate_heading_date(dvi_by_point[point_id], row["transplant_date"])

    weather_start = min(d - timedelta(days=30) for d in heading_dates.values())
    weather_end = max(d + timedelta(days=24) for d in heading_dates.values())
    weather = weather_provider(points, weather_start, weather_end, AMD_ELEMENTS)
    weather_by_point = {w.point_id: w for w in weather}

    rows: list[dict[str, object]] = []
    for _, row in points.iterrows():
        point_id = str(row["point_id"])
        point_weather = weather_by_point[point_id]
        heading = heading_dates[point_id]
        mode = int(row["mode"]) if "mode" in points and pd.notna(row.get("mode")) else default_mode
        correction = (
            float(row["correction"])
            if "correction" in points and pd.notna(row.get("correction"))
            else default_correction
        )
        amylose = estimate_amylose(point_weather.data["TMP_mea"], heading)
        protein = estimate_protein_at_point(point_weather.data, heading, mode=mode, correction=correction)
        rows.append(
            {
                "point_id": point_id,
                **({"region": row["region"]} if "region" in points and pd.notna(row.get("region")) else {}),
                "lat": point_weather.lat,
                "lon": point_weather.lon,
                "transplant_date": parse_date(row["transplant_date"]),
                **amylose,
                **protein,
            }
        )

    point_results = pd.DataFrame(rows)
    return point_results, estimate_protein_distribution(point_results)


def fake_weather_provider(
    points: pd.DataFrame,
    start_date: date,
    end_date: date,
    elements: Sequence[str],
) -> list[PointWeather]:
    """Deterministic provider used by the smoke test; it does not access AMD."""
    index = pd.date_range(start_date, end_date, freq="D")
    rows = []
    for _, point in points.iterrows():
        day = np.arange(len(index), dtype=float)
        frame = pd.DataFrame(index=index)
        if "TMP_mea" in elements:
            frame["TMP_mea"] = 21.0 + 3.0 * np.sin(day / 18.0) + float(point["lat"] - points["lat"].mean()) * 0.2
        if "GSR" in elements:
            frame["GSR"] = 17.0 + 1.5 * np.cos(day / 13.0)
        if "APCP" in elements:
            frame["APCP"] = 3.0 + 2.0 * (np.sin(day / 7.0) > 0)
        rows.append(PointWeather(str(point["point_id"]), float(point["lat"]), float(point["lon"]), frame))
    return rows


def write_sample_points(path: str | Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["point_id", "lat", "lon", "transplant_date", "mode", "correction"])
        writer.writerow(["hokkaido_1", 43.76884, 143.81094, "2026-05-25", 1, 0.0])
        writer.writerow(["hokkaido_2", 42.50210, 140.80080, "2026-05-25", 2, 0.1])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="米品質推定の動作確認")
    parser.add_argument(
        "--points-csv",
        default=str(DEFAULT_POINTS_CSV),
        help="地点CSV。未指定時は scripts/amydas2_409points.csv を使用します。",
    )
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--use-amd", action="store_true", help="AMD_Tools4.pyで実データを取得します。")
    parser.add_argument("--amd-url", default="https://amd.rd.naro.go.jp/opendap/AMD/")
    parser.add_argument("--output", default="rice_quality_points.csv")
    args = parser.parse_args(argv)

    points = load_points_csv(args.points_csv)

    if args.use_amd:
        provider = lambda p, s, e, elements: fetch_amd_point_weather(p, s, e, elements, amd_url=args.amd_url)
    else:
        provider = fake_weather_provider
    point_results, aggregate = estimate_quality_for_points(points, provider, year=args.year)
    point_results.to_csv(args.output, index=False)
    print(f"Calculated {len(point_results)} points from {args.points_csv}")
    if len(point_results) > 20:
        print(point_results.head(10).to_string(index=False))
        print(f"... {len(point_results) - 10} more rows written to {args.output}")
    else:
        print(point_results.to_string(index=False))
    print("\nAggregate")
    for key, value in aggregate.items():
        print(f"{key}: {value}")
    print(f"\nWrote {args.output}")
    return 0


def _slice_by_date(series: pd.Series, start: date, end: date | None) -> pd.Series:
    idx = pd.to_datetime(series.index).normalize()
    normalized = pd.Series(series.to_numpy(dtype=float), index=idx)
    mask = idx.date >= start
    if end is not None:
        mask &= idx.date <= end
    return normalized.loc[mask]


def _mean_between(series: pd.Series, start: date, end: date) -> float:
    values = _slice_by_date(series, start, end)
    expected_days = (end - start).days + 1
    if len(values) != expected_days:
        raise ValueError(f"weather period is incomplete: {start} to {end}")
    mean = float(values.mean(skipna=True))
    if math.isnan(mean):
        raise ValueError(f"weather period has no valid values: {start} to {end}")
    return mean


def _date_from_index(value: object) -> date:
    return pd.Timestamp(value).date()


def _normal_cdf(x: float, mean: float, stddev: float) -> float:
    return 0.5 * (1.0 + math.erf((x - mean) / (stddev * math.sqrt(2.0))))


def _xlsx_sheet_path(path: Path, sheet_name: str) -> str:
    ns = {
        "a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
        "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
        "rel": "http://schemas.openxmlformats.org/package/2006/relationships",
    }
    with ZipFile(path) as z:
        workbook = ET.fromstring(z.read("xl/workbook.xml"))
        rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
        relmap = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels}
        for sheet in workbook.find("a:sheets", ns):
            if sheet.attrib["name"] == sheet_name:
                rid = sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
                target = relmap[rid]
                return target if target.startswith("xl/") else f"xl/{target}"
    raise ValueError(f"sheet not found: {sheet_name}")


def _xlsx_sheet_rows(path: Path, sheet_path: str) -> list[tuple[int, dict[int, str]]]:
    ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with ZipFile(path) as z:
        shared = _xlsx_shared_strings(z)
        root = ET.fromstring(z.read(sheet_path))
        rows: list[tuple[int, dict[int, str]]] = []
        for row in root.findall(".//a:sheetData/a:row", ns):
            row_number = int(row.attrib["r"])
            values: dict[int, str] = {}
            for cell in row.findall("a:c", ns):
                col = _xlsx_column_number(cell.attrib["r"])
                values[col] = _xlsx_cell_value(cell, shared)
            if values:
                rows.append((row_number, values))
        return rows


def _xlsx_shared_strings(z: ZipFile) -> list[str]:
    ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    if "xl/sharedStrings.xml" not in z.namelist():
        return []
    root = ET.fromstring(z.read("xl/sharedStrings.xml"))
    return ["".join(t.text or "" for t in item.findall(".//a:t", ns)) for item in root.findall("a:si", ns)]


def _xlsx_cell_value(cell: ET.Element, shared: Sequence[str]) -> str:
    ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    cell_type = cell.attrib.get("t")
    value = cell.find("a:v", ns)
    if cell_type == "s":
        return shared[int(value.text)] if value is not None else ""
    if cell_type == "inlineStr":
        return "".join(t.text or "" for t in cell.findall(".//a:t", ns))
    return value.text if value is not None else ""


def _xlsx_column_number(cell_ref: str) -> int:
    match = re.match(r"([A-Z]+)[0-9]+", cell_ref)
    if not match:
        raise ValueError(f"invalid cell reference: {cell_ref}")
    number = 0
    for char in match.group(1):
        number = number * 26 + ord(char) - 64
    return number


def _excel_serial_date(value: float) -> date:
    return (datetime(1899, 12, 30) + timedelta(days=value)).date()


def _float_or_none(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    raise SystemExit(main())

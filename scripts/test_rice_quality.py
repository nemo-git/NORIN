# -*- coding: utf-8 -*-

from datetime import date

import pandas as pd

from rice_quality import (
    dvr,
    estimate_amylose,
    estimate_heading_date,
    estimate_protein_at_point,
    estimate_quality_for_points,
    fake_weather_provider,
)


def test_dvr_matches_known_value():
    assert round(dvr(20.0), 8) == 0.01705043


def test_heading_date_from_constant_temperature():
    temps = pd.Series(20.0, index=pd.date_range("2026-05-25", periods=80, freq="D"))
    assert estimate_heading_date(temps, "2026-05-25") == date(2026, 7, 22)


def test_amylose_uses_corrected_temperature():
    temps = pd.Series(22.0, index=pd.date_range("2026-07-01", periods=80, freq="D"))
    result = estimate_amylose(temps, "2026-07-30")
    assert result["A106_heading_minus5_to_plus4_TMP_mea"] == 22.0
    assert round(result["A108_corrected_TMP_mea"], 6) == 22.333333
    assert result["amylose_percent"] == 16.0


def test_protein_mode_1_and_aggregate_smoke():
    index = pd.date_range("2026-07-01", periods=80, freq="D")
    weather = pd.DataFrame(
        {
            "TMP_mea": 22.0,
            "GSR": 17.0,
            "APCP": 4.0,
        },
        index=index,
    )
    point = estimate_protein_at_point(weather, "2026-07-30", mode=1, correction=0.0)
    assert point["panicle_formation_date"] == date(2026, 7, 5)
    assert point["protein_stddev"] == 0.54
    assert round(point["protein_mean"], 6) == 6.779989


def test_quality_for_points_with_fake_provider():
    points = pd.DataFrame(
        [
            {"point_id": "A", "lat": 43.0, "lon": 142.0, "transplant_date": "2026-05-25"},
            {"point_id": "B", "lat": 43.1, "lon": 142.1, "transplant_date": "2026-05-25", "mode": 2},
        ]
    )
    point_results, aggregate = estimate_quality_for_points(points, fake_weather_provider, year=2026)
    assert len(point_results) == 2
    assert "amylose_percent" in point_results
    assert 0.0 <= aggregate["standard_quality_percent"] <= 100.0


if __name__ == "__main__":
    test_dvr_matches_known_value()
    test_heading_date_from_constant_temperature()
    test_amylose_uses_corrected_temperature()
    test_protein_mode_1_and_aggregate_smoke()
    test_quality_for_points_with_fake_provider()
    print("OK")

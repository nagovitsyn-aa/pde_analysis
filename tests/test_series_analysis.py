import numpy as np
import pandas as pd
import pytest

from pde_analysis.db_analysis.series_analysis import (
    compute_wsat_metrics,
    filter_series_dataframe,
    normalize_gamma_values,
)
from pde_analysis.processing.compute import (
    compute_amplification_by_max,
    compute_amplification_by_median_after_time,
    find_increment_peak,
)


def test_filter_series_dataframe_excludes_run_ids_and_applies_selection():
    df = pd.DataFrame(
        [
            {"run_id": 1, "u": 0.1, "lambda_": 0.5, "ts_name": "A", "t": 0.0, "value": 1.0, "series_name": "s1"},
            {"run_id": 2, "u": 0.1, "lambda_": 0.5, "ts_name": "A", "t": 1.0, "value": 2.0, "series_name": "s1"},
            {"run_id": 3, "u": 0.2, "lambda_": 0.6, "ts_name": "B", "t": 0.0, "value": 3.0, "series_name": "s2"},
        ]
    )

    filtered = filter_series_dataframe(
        df,
        exclude_run_ids=[2],
        selected_u=[0.1],
        selected_lambda=[0.5],
        selected_ts_name=["A"],
    )

    assert filtered["run_id"].tolist() == [1]
    assert filtered["series_name"].tolist() == ["s1"]


def test_compute_amplification_by_max_and_median_helpers():
    t = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
    y = np.array([1.0, 5.0, 10.0, 4.0, 20.0], dtype=float)

    assert compute_amplification_by_max(t, y, t_max=2.0) == pytest.approx(10.0)
    assert compute_amplification_by_median_after_time(t, y, t_start=3.0) == pytest.approx(12.0)


def test_compute_wsat_metrics_respects_time_window_and_ts_filter():
    df = pd.DataFrame(
        [
            {"u": 0.1, "lambda_": 0.5, "ts_name": "A", "t": 0.0, "value": 1.0},
            {"u": 0.1, "lambda_": 0.5, "ts_name": "A", "t": 1.0, "value": 3.0},
            {"u": 0.1, "lambda_": 0.5, "ts_name": "A", "t": 2.0, "value": 2.0},
            {"u": 0.1, "lambda_": 0.5, "ts_name": "B", "t": 1.0, "value": 99.0},
            {"u": 0.2, "lambda_": 0.6, "ts_name": "A", "t": 1.0, "value": 7.0},
        ]
    )

    metrics = compute_wsat_metrics(
        df,
        ts_name="A",
        t_min=0.5,
        t_max=1.5,
    )

    assert metrics == [
        {"u": 0.1, "lambda_": 0.5, "Wsat": pytest.approx(3.0)},
        {"u": 0.2, "lambda_": 0.6, "Wsat": pytest.approx(7.0)},
    ]


def test_find_increment_peak_uses_the_second_peak():
    t = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=float)
    log_y = np.array([0.0, 0.1, 0.2, 0.7, 1.0, 1.2, 1.9, 1.7, 2.0], dtype=float)
    y = np.exp(log_y)

    slope, _, _ = find_increment_peak(
        t,
        y,
        min_len=3,
        max_len=10,
        t_min=0.0,
        t_max=8.0,
        peak_number=2,
        half_width=0.5,
    )

    assert slope == pytest.approx(0.45)


def test_normalize_gamma_values_supports_reference_scaling():
    gamma = np.array([2.0, 4.0], dtype=float)

    normalized = normalize_gamma_values(
        gamma,
        gamma_values_ref=gamma,
        u_value=1.0,
        lambda_value=1.0,
        gamma_1dz_interp=lambda lam: 2.0,
        normalization_type="1Dz",
    )

    assert normalized.tolist() == pytest.approx([1.0, 2.0])

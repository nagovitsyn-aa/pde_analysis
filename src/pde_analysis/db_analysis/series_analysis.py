from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd

from pde_analysis.processing.compute import (
    compute_amplification_by_max,
    compute_amplification_by_median_after_time,
    find_increment_peak,
)


def normalize_gamma_values(
    gamma_values,
    *,
    gamma_values_ref=None,
    u_value=None,
    lambda_value=None,
    gamma_1dz_interp=None,
    gamma_1dx_interp=None,
    normalization_type: str = "1Dz",
):
    """Normalize gamma values using the same conventions as the notebook."""
    gamma_array = np.asarray(gamma_values, dtype=float)
    if gamma_values_ref is not None:
        gamma_ref = np.asarray(gamma_values_ref, dtype=float)
    else:
        gamma_ref = gamma_array

    if normalization_type == "1Dz":
        if gamma_1dz_interp is None:
            return gamma_array
        return gamma_array / np.asarray(gamma_1dz_interp(lambda_value), dtype=float)

    if normalization_type == "1Dx":
        if gamma_1dx_interp is None:
            return gamma_array
        return gamma_array / np.asarray(gamma_1dx_interp(u_value), dtype=float)

    if normalization_type == "1Dz*1Dx":
        if gamma_1dz_interp is None or gamma_1dx_interp is None:
            return gamma_array
        return 2.0 * gamma_array / (
            np.asarray(gamma_1dz_interp(lambda_value), dtype=float)
            * np.asarray(gamma_1dx_interp(u_value), dtype=float)
        )

    return gamma_array


def filter_series_dataframe(
    df: pd.DataFrame,
    *,
    exclude_run_ids: Optional[Iterable[int]] = None,
    selected_u: Optional[Iterable[float]] = None,
    selected_lambda: Optional[Iterable[float]] = None,
    selected_ts_name: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Filter a timeseries dataframe for a single experiment selection."""
    filtered = df.copy()

    if exclude_run_ids:
        exclude_run_ids = set(exclude_run_ids)
        filtered = filtered[~filtered["run_id"].isin(exclude_run_ids)]

    if selected_u is not None:
        filtered = filtered[filtered["u"].isin(list(selected_u))]

    if selected_lambda is not None:
        filtered = filtered[filtered["Lambda"].isin(list(selected_lambda))]

    if selected_ts_name is not None:
        filtered = filtered[filtered["ts_name"].isin(list(selected_ts_name))]

    return filtered.reset_index(drop=True)


def _grouping_column(df: pd.DataFrame) -> str:
    for column_name in ("experiment_name", "series_name"):
        if column_name in df.columns:
            return column_name
    return "experiment_name"


def build_series_groups(
    df: pd.DataFrame,
    *,
    group_by: Optional[str] = None,
) -> list[dict]:
    """Prepare grouped data for plotting and analysis."""
    group_column = group_by or _grouping_column(df)
    groups = []
    for key, group in df.groupby(group_column, dropna=False):
        groups.append({group_column: key, "data": group})
    return groups


def prepare_series_dataframe(
    df: pd.DataFrame,
    *,
    experiment_name: Optional[str] = None,
    series_name: Optional[str] = None,
) -> pd.DataFrame:
    """Attach the experiment label used by the current notebook workflow."""
    prepared = df.copy()

    label = experiment_name or series_name
    if label is not None:
        prepared["experiment_name"] = label

    if "experiment_name" not in prepared.columns:
        prepared["experiment_name"] = "default"

    if "series_name" not in prepared.columns:
        prepared["series_name"] = prepared["experiment_name"]

    if "u" in prepared.columns:
        prepared["u"] = prepared["u"].astype(float)

    if "Lambda" in prepared.columns:
        prepared["Lambda"] = prepared["Lambda"].astype(float)

    return prepared


def prepare_plot_data(
    df: pd.DataFrame,
    *,
    mode: str = "single",
) -> list[dict]:
    """Create plot-ready data for single or compare mode."""
    if df.empty:
        return []

    plot_data = []
    group_column = _grouping_column(df)
    for _, row in df.groupby([group_column, "u", "Lambda"], dropna=False):
        label = row[group_column].iloc[0]
        plot_data.append(
            {
                "experiment_name": label,
                "u": row["u"].iloc[0],
                "Lambda": row["Lambda"].iloc[0],
                "t": row["t"].to_numpy(dtype=float),
                "value": row["value"].to_numpy(dtype=float),
                "mode": mode,
            }
        )
    return plot_data


def compute_wsat_metrics(
    df: pd.DataFrame,
    *,
    ts_name: Optional[str] = None,
    t_min: float = 0.0,
    t_max: Optional[float] = None,
    aggregation: str = "max",
    t_start: Optional[float] = None,
) -> list[dict]:
    """Compute Wsat-like amplification values for grouped (u, Lambda) data."""
    filtered = df.copy()
    if ts_name is not None:
        filtered = filtered[filtered["ts_name"] == ts_name]

    results = []
    for (u_value, lambda_value), group in filtered.groupby(["u", "Lambda"], dropna=False):
        t = group["t"].to_numpy(dtype=float)
        y = group["value"].to_numpy(dtype=float)

        mask = t >= t_min
        if t_max is not None:
            mask &= t <= t_max

        t = t[mask]
        y = y[mask]

        if len(t) == 0:
            continue

        if aggregation == "median_after_time":
            if t_start is None:
                t_start_value = float(np.nanmax(t))
            else:
                t_start_value = t_start
            amplification = compute_amplification_by_median_after_time(
                t,
                y,
                t_start=t_start_value,
            )
        else:
            amplification = compute_amplification_by_max(t, y, t_max=t_max)

        if np.isnan(amplification):
            continue
        results.append(
            {
                "u": float(u_value),
                "Lambda": float(lambda_value),
                "Wsat": float(amplification),
            }
        )

    return results


def compute_increment_metrics(
    df: pd.DataFrame,
    *,
    min_len: int = 3,
    max_len: int = 10,
    t_min: float = 0.0,
    t_max: Optional[float] = None,
    ts_name: Optional[str] = None,
    peak_number: int = 2,
    half_width: float = 0.5,
) -> list[dict]:
    """Compute notebook-style peak-based increment metrics for each (u, Lambda) group."""
    filtered = df.copy()
    if ts_name is not None:
        filtered = filtered[filtered["ts_name"] == ts_name]

    results = []
    for (u_value, lambda_value), group in filtered.groupby(["u", "Lambda"], dropna=False):
        t = group["t"].to_numpy(dtype=float)
        y = group["value"].to_numpy(dtype=float)

        increment = find_increment_peak(
            t,
            y,
            min_len=min_len,
            max_len=max_len,
            t_min=t_min,
            t_max=t_max,
            peak_number=peak_number,
            half_width=half_width,
        )
        if increment is None:
            continue

        slope, _, _ = increment
        results.append(
            {
                "u": float(u_value),
                "Lambda": float(lambda_value),
                "gamma": float(slope),
            }
        )

    return results

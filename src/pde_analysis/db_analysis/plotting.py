import matplotlib.pyplot as plt
import pandas as pd
from typing import Iterable, Optional


def plot_metric_vs_u(
    df: pd.DataFrame,
    metric: str,
    lambdas: Optional[Iterable[float]] = None,
    ax=None,
):
    """
    Строит metric(u) для разных Lambda

    lambdas:
        список Lambda, которые нужно оставить (если None — все)
    """

    if ax is None:
        fig, ax = plt.subplots()

    grouped = df.groupby("Lambda")

    for Lambda, subdf in grouped:
        if lambdas is not None and Lambda not in lambdas:
            continue

        subdf = subdf.sort_values("u")
        subdf = subdf.dropna(subset=[metric])

        if subdf.empty:
            continue

        ax.plot(
            subdf["u"],
            subdf[metric],
            marker="o",
            label=f"Λ={Lambda}",
        )

    ax.set_xlabel("u")
    ax.set_ylabel(metric)
    ax.legend()
    ax.grid(True)

    return ax


def plot_metric_vs_lambda(
    df: pd.DataFrame,
    metric: str,
    us: Optional[Iterable[float]] = None,
    ax=None,
):
    """
    Строит metric(Lambda) для разных u

    us:
        список u, которые нужно оставить
    """

    if ax is None:
        fig, ax = plt.subplots()

    grouped = df.groupby("u")

    for u, subdf in grouped:
        if us is not None and u not in us:
            continue

        subdf = subdf.sort_values("Lambda")
        subdf = subdf.dropna(subset=[metric])

        if subdf.empty:
            continue

        ax.plot(
            subdf["Lambda"],
            subdf[metric],
            marker="o",
            label=f"u={u}",
        )

    ax.set_xlabel("Lambda")
    ax.set_ylabel(metric)
    ax.legend()
    ax.grid(True)

    return ax
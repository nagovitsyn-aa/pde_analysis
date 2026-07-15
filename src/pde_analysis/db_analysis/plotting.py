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
    Строит metric(u) для разных lambda
    lambda_vals:
        список lambda_, которые нужно оставить (если None — все)
    """

    if ax is None:
        fig, ax = plt.subplots()

    grouped = df.groupby("lambda_")

    for lambda_, subdf in grouped:
        if lambda_vals is not None and lambda_ not in lambda_vals:
            continue

        subdf = subdf.sort_values("u")
        subdf = subdf.dropna(subset=[metric])

        if subdf.empty:
            continue

        ax.plot(
            subdf["u"],
            subdf[metric],
            marker="o",
            label=f"\u039b={lambda_}",
        )

    ax.set_xlabel("u")
    ax.set_ylabel(metric)
    ax.legend()
    ax.grid(True)

    return ax


def plot_metric_vs_lambda(
    df: pd.DataFrame,
    metric: str,
    u_vals: Optional[Iterable[float]] = None,
    ax=None,
):
    """
    Строит metric(lambda) для разных u

    u_vals:
        список u, которые нужно оставить
    """

    if ax is None:
        fig, ax = plt.subplots()

    grouped = df.groupby("u")

    for u, subdf in grouped:
        if u_vals is not None and u not in u_vals:
            continue

        subdf = subdf.sort_values("lambda_")
        subdf = subdf.dropna(subset=[metric])

        if subdf.empty:
            continue

        ax.plot(
            subdf["lambda_"],
            subdf[metric],
            marker="o",
            label=f"u={u}",
        )

    ax.set_xlabel(r"$\Lambda$")
    ax.set_ylabel(metric)
    ax.legend()
    ax.grid(True)

    return ax
import marimo

__generated_with = "0.23.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt

    from pde_analysis.db_analysis.load_timeseries import load_timeseries_dataframe
    from pde_analysis.db_analysis.preprocess import normalize_parameters

    return load_timeseries_dataframe, mo, normalize_parameters, plt


@app.cell
def _(mo):
    db_path = mo.ui.text(
        value="data/db/simulations.db",
        label="db path",
        full_width=True,
    )
    db_path
    return (db_path,)


@app.cell
def _(db_path, mo):
    def load_table(db_path: str, query: str):
        import sqlite3
        import pandas as pd

        conn = sqlite3.connect(db_path)
        try:
            return pd.read_sql_query(query, conn)
        finally:
            conn.close()

    runs_df = load_table(
        db_path.value,
        """
        SELECT
            r.run_id,
            e.name AS experiment_name,
            r.file_name,
            r.dimension,
            r.Lambda,
            r.u,
            r.tend,
            r.x0,
            r.rangeX,
            r.rangeY,
            r.dx,
            r.dy,
            r.status,
            r.note,
            r.created_at
        FROM runs r
        JOIN experiments e ON r.experiment_id = e.experiment_id
        ORDER BY e.name, r.run_id
        """,
    )

    experiment_names = (
        sorted(runs_df["experiment_name"].dropna().unique().tolist())
        if not runs_df.empty
        else []
    )

    experiment_ui = mo.ui.dropdown(
        options=experiment_names,
        value=experiment_names[0] if experiment_names else None,
        label="experiment",
    )

    experiment_ui
    return (experiment_ui,)


@app.cell
def _(experiment_ui, load_timeseries_dataframe, normalize_parameters):
    df = load_timeseries_dataframe(
        experiment_name=experiment_ui.value
    )

    df = normalize_parameters(df, ndigits=5)

    df
    return (df,)


@app.cell
def _(df, mo):
    ts_options = sorted(df["ts_name"].unique())

    ts_ui = mo.ui.dropdown(
        options=ts_options,
        value=ts_options[0] if ts_options else None,
        label="timeseries",
    )

    ts_ui
    return (ts_ui,)


@app.cell
def _(mo):
    mode_ui = mo.ui.radio(
        options=["fixed_u", "fixed_Lambda"],
        value="fixed_u",
        label="mode",
    )

    mode_ui
    return (mode_ui,)


@app.cell
def _(df, mo):
    u_values = sorted(df["u"].dropna().unique())
    Lambda_values = sorted(df["Lambda"].dropna().unique())

    fixed_u_ui = mo.ui.dropdown(
        options=u_values,
        value=u_values[0] if u_values else None,
        label="fixed u",
    )

    fixed_Lambda_ui = mo.ui.dropdown(
        options=Lambda_values,
        value=Lambda_values[0] if Lambda_values else None,
        label="fixed Lambda",
    )

    visible_u_ui = mo.ui.multiselect(
        options=u_values,
        value=u_values,
        label="visible u",
    )

    visible_Lambda_ui = mo.ui.multiselect(
        options=Lambda_values,
        value=Lambda_values,
        label="visible Lambda",
    )

    fixed_u_ui, fixed_Lambda_ui, visible_u_ui, visible_Lambda_ui
    return fixed_Lambda_ui, fixed_u_ui, visible_Lambda_ui, visible_u_ui


@app.cell
def _(mo):
    scale_ui = mo.ui.radio(
        options=["linear", "log"],
        value="linear",
        label="y-scale",
    )

    scale_ui
    return (scale_ui,)


@app.cell
def _(df, fixed_Lambda_ui, fixed_u_ui, mode_ui, ts_ui):
    if ts_ui.value is None:
        df_filtered = df.iloc[0:0]
    else:
        df_filtered = df[df["ts_name"] == ts_ui.value]

        if mode_ui.value == "fixed_u":
            df_filtered = df_filtered[
                df_filtered["u"] == fixed_u_ui.value
            ]
        else:
            df_filtered = df_filtered[
                df_filtered["Lambda"] == fixed_Lambda_ui.value
            ]

    df_filtered
    return (df_filtered,)


@app.cell
def _(df_filtered, mode_ui, plt, scale_ui, visible_Lambda_ui, visible_u_ui):
    import numpy as np

    plt.figure()

    linestyles = [
        "-", "--", "-.", ":",
        (0, (1, 1)),
        (0, (3, 1, 1, 1)),
        (0, (5, 2)),
        (0, (2, 2, 5, 2))
    ]

    if mode_ui.value == "fixed_u":
        grouped = df_filtered.groupby("Lambda")

        for i, (Lambda, subdf) in enumerate(grouped):
            if Lambda not in visible_Lambda_ui.value:
                continue

            subdf = subdf.sort_values("t")

            linestyle = linestyles[i % len(linestyles)]

            plt.plot(
                subdf["t"],
                subdf["value"],
                label=f"Λ={Lambda}",
                linestyle=linestyle,
                linewidth=2,
                alpha=0.8,
            )

    else:
        grouped = df_filtered.groupby("u")

        for i, (u, subdf) in enumerate(grouped):
            if u not in visible_u_ui.value:
                continue

            subdf = subdf.sort_values("t")

            linestyle = linestyles[i % len(linestyles)]

            plt.plot(
                subdf["t"],
                subdf["value"],
                label=f"u={u}",
                linestyle=linestyle,
                linewidth=2,
                alpha=0.8,
            )

    if scale_ui.value == "log":
        plt.yscale("log")

    plt.xlabel("t")
    plt.ylabel("value")
    plt.grid()
    plt.legend()

    plt.gca()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

import marimo

__generated_with = "0.22.0"
app = marimo.App()


@app.cell
def _():
    import sqlite3
    import pandas as pd
    import matplotlib.pyplot as plt
    import marimo as mo

    return mo, pd, plt, sqlite3


@app.function
def load_table(db_path: str, query: str):
    import sqlite3
    import pandas as pd

    conn = sqlite3.connect(db_path)
    try:
        return pd.read_sql_query(query, conn)
    finally:
        conn.close()


@app.function
def format_range(series):
    s = series.dropna()
    if len(s) == 0:
        return None
    return (float(s.min()), float(s.max()))


@app.function
def build_experiments_summary(runs_df):
    import pandas as pd

    if runs_df.empty:
        return pd.DataFrame()

    grouped = runs_df.groupby(["experiment_name", "dimension"], dropna=False)

    rows = []
    for (experiment_name, dimension), g in grouped:
        rows.append(
            {
                "experiment_name": experiment_name,
                "dimension": dimension,
                "n_runs": len(g),
                "Lambda_min": g["Lambda"].min(),
                "Lambda_max": g["Lambda"].max(),
                "u_min": g["u"].min(),
                "u_max": g["u"].max(),
                "x0_min": g["x0"].min(),
                "x0_max": g["x0"].max(),
                "tend_min": g["tend"].min(),
                "tend_max": g["tend"].max(),
                "dx_min": g["dx"].min(),
                "dx_max": g["dx"].max(),
                "dy_min": g["dy"].min() if "dy" in g.columns else None,
                "dy_max": g["dy"].max() if "dy" in g.columns else None,
            }
        )

    summary = pd.DataFrame(rows)
    return summary.sort_values(["experiment_name", "dimension"]).reset_index(drop=True)


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
def _(db_path, load_table):
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

    metric_defs_df = load_table(
        db_path.value,
        """
        SELECT metric_def_id, name, version
        FROM metric_definitions
        ORDER BY name, version
        """,
    )

    metrics_df = load_table(
        db_path.value,
        """
        SELECT
            m.metric_id,
            m.run_id,
            md.name AS metric_name,
            md.version,
            m.value
        FROM metrics m
        JOIN metric_definitions md
            ON m.metric_def_id = md.metric_def_id
        """,
    )

    timeseries_df = load_table(
        db_path.value,
        """
        SELECT run_id, name
        FROM timeseries
        """,
    )

    return metric_defs_df, metrics_df, runs_df, timeseries_df


@app.cell
def _(build_experiments_summary, runs_df):
    experiments_summary = build_experiments_summary(runs_df)
    return (experiments_summary,)


@app.cell
def _(metric_defs_df, mo, runs_df):
    experiment_names = (
        sorted(runs_df["experiment_name"].dropna().unique().tolist())
        if not runs_df.empty
        else []
    )

    dimensions = (
        sorted(runs_df["dimension"].dropna().unique().tolist())
        if not runs_df.empty and "dimension" in runs_df.columns
        else []
    )

    metric_names = (
        sorted(metric_defs_df["name"].dropna().unique().tolist())
        if not metric_defs_df.empty
        else []
    )

    experiment_ui = mo.ui.dropdown(
        options=experiment_names,
        value=experiment_names[0] if experiment_names else None,
        label="experiment",
    )

    dimension_ui = mo.ui.dropdown(
        options=["ALL"] + dimensions,
        value="ALL",
        label="dimension",
    )

    metric_ui = mo.ui.dropdown(
        options=metric_names,
        value=metric_names[0] if metric_names else None,
        label="metric",
    )

    controls = mo.hstack([experiment_ui, dimension_ui, metric_ui], justify="start")
    controls
    return dimension_ui, experiment_ui, metric_ui


@app.cell
def _(dimension_ui, experiment_ui, runs_df):
    if runs_df.empty or experiment_ui.value is None:
        selected_runs = runs_df.iloc[0:0].copy()
    else:
        selected_runs = runs_df[runs_df["experiment_name"] == experiment_ui.value].copy()
        if dimension_ui.value != "ALL":
            selected_runs = selected_runs[selected_runs["dimension"] == dimension_ui.value].copy()

    return (selected_runs,)


@app.cell
def _(metrics_df, metric_ui, selected_runs):
    if selected_runs.empty or metric_ui.value is None:
        selected_metrics = metrics_df.iloc[0:0].copy()
    else:
        run_ids = selected_runs["run_id"].tolist()
        selected_metrics = metrics_df[
            (metrics_df["run_id"].isin(run_ids))
            & (metrics_df["metric_name"] == metric_ui.value)
        ].copy()

    return (selected_metrics,)


@app.cell
def _(experiments_summary, metric_defs_df, mo, runs_df):
    n_experiments = runs_df["experiment_name"].nunique() if not runs_df.empty else 0
    n_runs = len(runs_df)
    n_metrics = len(metric_defs_df)

    n_1d = 0
    n_2d = 0
    if not runs_df.empty and "dimension" in runs_df.columns:
        n_1d = int((runs_df["dimension"] == "1D").sum())
        n_2d = int((runs_df["dimension"] == "2D").sum())

    overview = mo.vstack(
        [
            mo.md("## Database overview"),
            mo.md(f"- experiments: **{n_experiments}**"),
            mo.md(f"- runs: **{n_runs}**"),
            mo.md(f"- 1D runs: **{n_1d}**"),
            mo.md(f"- 2D runs: **{n_2d}**"),
            mo.md(f"- metric definitions: **{n_metrics}**"),
        ]
    )

    experiments_view = mo.vstack(
        [
            mo.md("## Experiments summary"),
            experiments_summary,
        ]
    )

    tabs = mo.ui.tabs(
        {
            "Overview": overview,
            "Experiments": experiments_view,
        }
    )
    tabs
    return


@app.cell
def _(mo, selected_runs):
    series_view = mo.vstack(
        [
            mo.md("## Selected series: runs"),
            selected_runs,
        ]
    )
    series_view
    return


@app.cell
def _(plt, selected_metrics, selected_runs):
    if not selected_runs.empty:
        plt.figure()

        if "Lambda" in selected_runs.columns and "u" in selected_runs.columns:
            plt.scatter(selected_runs["Lambda"], selected_runs["u"])
            plt.xlabel("Lambda")
            plt.ylabel("u")
            plt.title("Coverage in parameter plane")

        plt.gca()
    return


@app.cell
def _(metric_ui, plt, selected_metrics, selected_runs):
    if not selected_runs.empty and not selected_metrics.empty:
        merged = selected_runs.merge(
            selected_metrics[["run_id", "value"]],
            on="run_id",
            how="inner",
        )

        if "u" in merged.columns:
            plt.figure()
            plt.scatter(merged["u"], merged["value"])
            plt.xlabel("u")
            plt.ylabel(metric_ui.value)
            plt.title(f"{metric_ui.value} vs u")
            plt.gca()
    return


@app.cell
def _(metric_ui, plt, selected_metrics, selected_runs):
    if not selected_runs.empty and not selected_metrics.empty:
        merged = selected_runs.merge(
            selected_metrics[["run_id", "value"]],
            on="run_id",
            how="inner",
        )

        if "Lambda" in merged.columns:
            plt.figure()
            plt.scatter(merged["Lambda"], merged["value"])
            plt.xlabel("Lambda")
            plt.ylabel(metric_ui.value)
            plt.title(f"{metric_ui.value} vs Lambda")
            plt.gca()
    return


@app.cell
def _(metrics_df, mo, runs_df, timeseries_df):
    missing_metrics = runs_df[
        ~runs_df["run_id"].isin(metrics_df["run_id"].unique())
    ].copy()

    missing_timeseries = runs_df[
        ~runs_df["run_id"].isin(timeseries_df["run_id"].unique())
    ].copy()

    diagnostics = mo.vstack(
        [
            mo.md("## Diagnostics"),
            mo.md("### Runs without metrics"),
            missing_metrics,
            mo.md("### Runs without timeseries"),
            missing_timeseries,
        ]
    )

    diagnostics
    return


if __name__ == "__main__":
    app.run()
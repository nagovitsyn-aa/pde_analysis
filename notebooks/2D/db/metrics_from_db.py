import marimo

__generated_with = "0.23.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt

    from pde_analysis.db_analysis.load import load_metrics_dataframe
    from pde_analysis.db_analysis.preprocess import normalize_parameters
    from pde_analysis.db_analysis.plotting import (
        plot_metric_vs_u,
        plot_metric_vs_lambda,
    )

    return (
        load_metrics_dataframe,
        mo,
        normalize_parameters,
        plot_metric_vs_lambda,
        plot_metric_vs_u,
        plt,
    )


@app.cell
def _(mo):
    experiment_ui = mo.ui.text(
        value="one_decay_parameters_scan_with_IC_dx=0p05",
        label="experiment name",
        full_width=True,
    )

    experiment_ui
    return (experiment_ui,)


@app.cell
def _(experiment_ui, load_metrics_dataframe, normalize_parameters):
    df = load_metrics_dataframe(
        experiment_name=experiment_ui.value
    )

    df = normalize_parameters(df, ndigits=5)

    df
    return (df,)


@app.cell
def _(df, mo):
    metric_options = [
        c for c in df.columns
        if c not in ["run_id", "u", "Lambda"]
    ]

    metric_ui = mo.ui.dropdown(
        options=metric_options,
        value=metric_options[0] if metric_options else None,
        label="metric",
    )

    metric_ui
    return (metric_ui,)


@app.cell
def _(mo):
    mode_ui = mo.ui.radio(
        options=["metric_vs_u", "metric_vs_Lambda"],
        value="metric_vs_u",
        label="mode",
    )

    mode_ui
    return (mode_ui,)


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
def _(df, mo):
    lambdas = sorted(df["Lambda"].dropna().unique())
    us = sorted(df["u"].dropna().unique())

    lambda_ui = mo.ui.multiselect(
        options=lambdas,
        value=lambdas,
        label="Lambda curves",
    )

    u_ui = mo.ui.multiselect(
        options=us,
        value=us,
        label="u curves",
    )

    lambda_ui, u_ui
    return lambda_ui, u_ui


@app.cell
def _(df, lambda_ui, u_ui):
    df_filtered = df[
        (df["Lambda"].isin(lambda_ui.value)) &
        (df["u"].isin(u_ui.value))
    ]

    df_filtered
    return (df_filtered,)


@app.cell
def _(
    df_filtered,
    metric_ui,
    mode_ui,
    plot_metric_vs_lambda,
    plot_metric_vs_u,
    plt,
    scale_ui,
):
    fig, ax = plt.subplots()

    if mode_ui.value == "metric_vs_u":
        plot_metric_vs_u(
            df_filtered,
            metric_ui.value,
            lambdas=None,
            ax=ax,
        )
    else:
        plot_metric_vs_lambda(
            df_filtered,
            metric_ui.value,
            us=None,
            ax=ax,
        )

    if scale_ui.value == "log":
        ax.set_yscale("log")

    fig
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

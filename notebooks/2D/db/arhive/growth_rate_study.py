import marimo

__generated_with = "0.23.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from pde_analysis.db_analysis.load import load_metrics_dataframe
    from pde_analysis.db_analysis.preprocess import normalize_parameters

    return load_metrics_dataframe, mo, normalize_parameters, np, plt


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
def _(lambda_range_ui, df, u_range_ui, visible_lambda_ui, visible_u_ui):
    df_filtered = df[
        (df["u"].between(*u_range_ui.value)) &
        (df["lambda_"].between(*lambda_range_ui.value))
    ]

    df_filtered = df_filtered[
        (df_filtered["u"].isin(visible_u_ui.value)) &
        (df_filtered["lambda_"].isin(visible_lambda_ui.value))
    ]

    df_filtered
    return (df_filtered,)


@app.cell(hide_code=True)
def _(np):
    data = np.array([
        [0.01, 1.9928050716967245],[0.1, 1.8161333200993417],[0.2, 1.6849609053565109],
        [0.3, 1.5799951885209726],[0.4, 1.4911154516411378],[0.5, 1.413789849033274],
        [0.6, 1.345388182960182],[0.7, 1.2841702745086352],[0.8, 1.22889470081888],
        [0.9, 1.178629086948092],[1.0, 1.132649643809105],[1.1, 1.090379883854399],
        [1.2, 1.0513513768786171],[1.3, 1.0151789562253077],[1.4, 0.9815368066318896],
        [1.5, 0.9501532130414946],[1.6, 0.9207952676389994],[1.7, 0.8932490300534381],
        [1.8, 0.8673683523181209],[1.9, 0.8429883944856915],[2.0, 0.8199766323645665],
        [2.5, 0.7218010496439926],[3.0, 0.6448274730878004],[3.5, 0.5826444868025907],
        [4.0, 0.5313354941995087],[4.5, 0.48820702277786343],[5.0, 0.4514127004981953],
        [6.0, 0.3918923008150795],[7.0, 0.34573326331168525],[8.0, 0.30882845685744115],
        [9.0, 0.2786077964395182],[10.0, 0.25559712030931114],[11.0, 0.23435768308055907],
        [12.0, 0.21610031290388074],[13.0, 0.20022589811693325],[14.0, 0.18628582937996452],
        [15.0, 0.1739391439678295],[16.0, 0.16292101500721307],[17.0, 0.1530227712749439],
        [18.0, 0.14413842133731422],[19.0, 0.13601207241602384],[20.0, 0.13023232911625723],
    ])

    lambda_num = data[:, 0]
    gamma_num = data[:, 1]

    lambda_num, gamma_num
    return lambda_num, gamma_num


@app.cell
def _(mo):
    mode_ui = mo.ui.radio(
        options=["vs_Lambda", "vs_u"],
        value="vs_Lambda",
        label="mode",
    )

    normalized_ui = mo.ui.checkbox(
        value=False,
        label="normalize by gamma_1Dz_num",
    )

    mode_ui, normalized_ui
    return mode_ui, normalized_ui


@app.cell
def _(df, mo):
    u_values = sorted(df["u"].dropna().unique())
    lambda_values = sorted(df["lambda_"].dropna().unique())

    visible_u_ui = mo.ui.multiselect(
        options=u_values,
        value=u_values,
        label="visible u",
    )

    visible_lambda_ui = mo.ui.multiselect(
        options=lambda_values,
        value=lambda_values,
        label="visible lambda",
    )

    visible_u_ui, visible_lambda_ui
    return visible_lambda_ui, visible_u_ui


@app.cell
def _(df, mo):
    u_range_ui = mo.ui.range_slider(
        start=float(df["u"].min()),
        stop=float(df["u"].max()),
        value=(float(df["u"].min()), float(df["u"].max())),
        label="u range",
    )

    lambda_range_ui = mo.ui.range_slider(
        start=float(df["lambda_"].min()),
        stop=float(df["lambda_"].max()),
        value=(float(df["lambda_"].min()), float(df["lambda_"].max())),
        label="lambda range",
    )

    u_range_ui, lambda_range_ui
    return lambda_range_ui, u_range_ui


@app.cell
def _(mo):
    scale_ui = mo.ui.radio(
        options=["linear", "log"],
        value="linear",
        label="y-scale",
    )

    legend_position_ui = mo.ui.radio(
        options=["inside", "right"],
        value="inside",
        label="legend position",
    )

    scale_ui, legend_position_ui
    return legend_position_ui, scale_ui


@app.cell
def _(lambda_num, gamma_num, np):
    def gamma_interp(L):
        return np.interp(L, lambda_num, gamma_num)

    return (gamma_interp,)


@app.cell
def _(
    lambda_num,
    df_filtered,
    gamma_interp,
    gamma_num,
    legend_position_ui,
    mode_ui,
    normalized_ui,
    np,
    plt,
    scale_ui,
):
    fig, ax = plt.subplots()
    colors = plt.cm.tab10.colors

    def transform(y, lam):
        if not normalized_ui.value:
            return y
        return y / gamma_interp(lam)

    if mode_ui.value == "vs_Lambda":
        grouped = list(df_filtered.groupby("u"))

        for i, (u, subdf) in enumerate(grouped):
            subdf = subdf.sort_values("lambda_")
            color = colors[i % len(colors)]

            lam = subdf["lambda_"].values

            if "growth_rate_avg" in subdf.columns:
                ax.plot(
                    lam,
                    transform(subdf["growth_rate_avg"].values, lam),
                    linestyle="-",
                    color=color,
                    label=f"u={u}",
                )

            if "growth_rate_stage" in subdf.columns:
                ax.plot(
                    lam,
                    transform(subdf["growth_rate_stage"].values, lam),
                    linestyle="--",
                    color=color,
                )

        lam_min = float(df_filtered["lambda_"].min())
        lam_max = float(df_filtered["lambda_"].max())

        if not normalized_ui.value:
            lambda_grid = np.linspace(lam_min, lam_max, 300)
            gamma_approx = 2.0 / (1.0 + 2.0 * lambda_grid / np.pi)

            ax.plot(
                lambda_grid,
                gamma_approx,
                "k--",
                label=r"$2\gamma_{1Dz}^{approx}(\Lambda)$",
            )

            mask = (lambda_num >= lam_min) & (lambda_num <= lam_max)

            ax.plot(
                lambda_num[mask],
                gamma_num[mask],
                color="gray",
                linestyle="-",
                label=r"$2\gamma_{1Dz}^{num}(\Lambda)$",
            )

        ax.set_xlim(lam_min, lam_max)
        ax.set_xlabel(r"$\Lambda$")

    else:
        grouped = list(df_filtered.groupby("lambda_"))

        for i, (lambda_, subdf) in enumerate(grouped):
            subdf = subdf.sort_values("u")
            color = colors[i % len(colors)]

            lam = np.full_like(subdf["u"].values, lambda_)

            if "growth_rate_avg" in subdf.columns:
                ax.plot(
                    subdf["u"],
                    transform(subdf["growth_rate_avg"].values, lam),
                    linestyle="-",
                    color=color,
                    label=f"\u039b={lambda_}",
                )

            if "growth_rate_stage" in subdf.columns:
                ax.plot(
                    subdf["u"],
                    transform(subdf["growth_rate_stage"].values, lam),
                    linestyle="--",
                    color=color,
                )

        ax.set_xlabel("u")

    if scale_ui.value == "log":
        ax.set_yscale("log")

    if normalized_ui.value:
        ax.set_ylabel(r"$\gamma(u,\Lambda)/\gamma_{1Dz}^{num}(\Lambda)$")
    else:
        ax.set_ylabel("growth rate")

    ax.grid()

    if legend_position_ui.value == "inside":
        ax.legend()
    else:
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
        fig.tight_layout()

    fig
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

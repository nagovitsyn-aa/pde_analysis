import marimo

__generated_with = "0.23.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from pde_analysis.db_analysis.load_timeseries import load_timeseries_dataframe
    from pde_analysis.db_analysis.preprocess import normalize_parameters

    return load_timeseries_dataframe, mo, normalize_parameters, np, plt


@app.cell
def _(mo):
    experiment_ui = mo.ui.text(
        value="one_decay_parameters_scan_with_IC_dx=0p05",
        label="experiment name",
    )
    experiment_ui
    return (experiment_ui,)


@app.cell
def _(experiment_ui, load_timeseries_dataframe, normalize_parameters):
    df_data = load_timeseries_dataframe(experiment_name=experiment_ui.value)
    df_data = normalize_parameters(df_data, ndigits=5)
    return (df_data,)


@app.cell
def _(df_data, visible_L_selector, visible_u_selector):
    df_filtered = df_data[
        (df_data["u"].isin(visible_u_selector.value)) &
        (df_data["Lambda"].isin(visible_L_selector.value))
    ]

    df_filtered
    return (df_filtered,)


@app.cell
def _(df_data, mo):
    ts_options_local = sorted(df_data["ts_name"].unique())

    ts_selector = mo.ui.dropdown(
        options=ts_options_local,
        value=ts_options_local[0] if ts_options_local else None,
    )
    ts_selector
    return (ts_selector,)


@app.cell
def _(mo):
    mode_selector = mo.ui.radio(
        options=["fixed_u", "fixed_Lambda"],
        value="fixed_u",
    )
    mode_selector
    return (mode_selector,)


@app.cell
def _(df_data, mo):
    u_vals_local = sorted(df_data["u"].dropna().unique())
    L_vals_local = sorted(df_data["Lambda"].dropna().unique())

    fixed_u_selector = mo.ui.dropdown(options=u_vals_local, value=u_vals_local[0])
    fixed_L_selector = mo.ui.dropdown(options=L_vals_local, value=L_vals_local[0])

    visible_u_selector = mo.ui.multiselect(options=u_vals_local, value=u_vals_local)
    visible_L_selector = mo.ui.multiselect(options=L_vals_local, value=L_vals_local)
    fixed_L_selector, fixed_u_selector, visible_u_selector, visible_L_selector
    return (
        fixed_L_selector,
        fixed_u_selector,
        visible_L_selector,
        visible_u_selector,
    )


@app.cell
def _(mo):
    scale_selector = mo.ui.radio(options=["linear", "log"], value="log")

    min_len_selector = mo.ui.slider(start=2, stop=50, value=3)
    max_len_selector = mo.ui.slider(start=4, stop=100, value=10)

    tmin_selector = mo.ui.number(value=2.0, label="t_min")
    tmax_selector = mo.ui.number(value=5.0, label="t_max")

    legend_position_selector = mo.ui.radio(
        options=["inside", "right", "outside"],
        value="inside",
        label="legend position",
    )
    legend_position_selector,min_len_selector,max_len_selector,scale_selector,tmin_selector,tmax_selector
    return (
        legend_position_selector,
        max_len_selector,
        min_len_selector,
        scale_selector,
        tmax_selector,
        tmin_selector,
    )


@app.cell
def _(np):
    def find_increment_internal(
        t_arr_internal,
        y_arr_internal,
        min_len_internal,
        max_len_internal,
        tmin_internal,
        tmax_internal,
    ):
        mask_internal = (
            (y_arr_internal > 0) &
            (t_arr_internal >= tmin_internal) &
            (t_arr_internal <= tmax_internal)
       )
        t_arr_internal = t_arr_internal[mask_internal]
        y_arr_internal = y_arr_internal[mask_internal]

        if len(t_arr_internal) < min_len_internal:
            return None

        log_y_internal = np.log(y_arr_internal)
        n_internal = len(t_arr_internal)

        best_tuple = None
        best_r2_val = -1

        for i_internal in range(n_internal - min_len_internal):
            for j_internal in range(i_internal + min_len_internal, n_internal):
                length_internal = j_internal - i_internal

                if length_internal > max_len_internal:
                    break

                x_internal = t_arr_internal[i_internal:j_internal]
                y_internal = log_y_internal[i_internal:j_internal]

                A_internal = np.vstack([x_internal, np.ones_like(x_internal)]).T
                coef_internal, *_ = np.linalg.lstsq(A_internal, y_internal, rcond=None)

                slope_internal = coef_internal[0]
                if slope_internal <= 0:
                    continue

                y_fit_internal = A_internal @ coef_internal
                ss_res_internal = np.sum((y_internal - y_fit_internal) ** 2)
                ss_tot_internal = np.sum((y_internal - y_internal.mean()) ** 2)

                if ss_tot_internal == 0:
                    continue

                r2_internal = 1 - ss_res_internal / ss_tot_internal

                if r2_internal > best_r2_val:
                    best_r2_val = r2_internal
                    best_tuple = (slope_internal, t_arr_internal[i_internal], t_arr_internal[j_internal])

        return best_tuple

    return (find_increment_internal,)


@app.cell
def _(
    df_filtered,
    fixed_L_selector,
    fixed_u_selector,
    mode_selector,
    ts_selector,
):
    grouped_output = []

    if ts_selector.value is not None:
        df_tmp_2 = df_filtered[df_filtered["ts_name"] == ts_selector.value]

        if mode_selector.value == "fixed_u":
            df_tmp_2 = df_tmp_2[df_tmp_2["u"] == fixed_u_selector.value]
            grouped_iter = df_tmp_2.groupby("Lambda")

            for key_local, df_group in grouped_iter:
                grouped_output.append(
                    {
                        "label_local": str(key_local),
                        "t_array": df_group["t"].values,
                        "y_array": df_group["value"].values,
                        "u_val": fixed_u_selector.value,
                    }
                )
        else:
            df_tmp_2 = df_tmp_2[df_tmp_2["Lambda"] == fixed_L_selector.value]
            grouped_iter = df_tmp_2.groupby("u")

            for key_local, df_group in grouped_iter:
                grouped_output.append(
                    {
                        "label_local": str(key_local),
                        "t_array": df_group["t"].values,
                        "y_array": df_group["value"].values,
                        "u_val": key_local,
                    }
                )
    return (grouped_output,)


@app.cell
def _(grouped_output, np):
    slope_output = []

    for entry_local in grouped_output:
        t_local = entry_local["t_array"]
        y_local = entry_local["y_array"]

        mask_local = y_local > 0
        t_local = t_local[mask_local]
        y_local = y_local[mask_local]

        if len(t_local) < 3:
            continue

        slope_vals = np.gradient(np.log(y_local), t_local)

        slope_output.append(
            {
                "label_local": entry_local["label_local"],
                "t_array": t_local,
                "slope_array": slope_vals,
            }
        )
    return (slope_output,)


@app.cell
def _(
    find_increment_internal,
    fixed_u_selector,
    grouped_output,
    max_len_selector,
    min_len_selector,
    tmax_selector,
    tmin_selector,
):
    fit_output = []

    for entry_fit in grouped_output:
        result_fit = find_increment_internal(
            entry_fit["t_array"],
            entry_fit["y_array"],
            min_len_selector.value,
            max_len_selector.value,
            tmin_selector.value,
            tmax_selector.value,
        )

        fit_output.append(
            {
                "label_local": entry_fit["label_local"],
                "t_array": entry_fit["t_array"],
                "y_array": entry_fit["y_array"],
                "fit": result_fit,
                "u_val": fixed_u_selector.value
            }
        )
    return (fit_output,)


@app.cell
def _(
    grouped_output,
    legend_position_selector,
    mode_selector,
    np,
    plt,
    scale_selector,
):
    plt.figure()

    for entry_plot in grouped_output:
        plt.plot(
            entry_plot["t_array"],
            entry_plot["y_array"],
            label=entry_plot["label_local"]
        )

        if mode_selector.value == "fixed_u":
            u_val_local = entry_plot["u_val"]
            level_val = np.exp(2 * np.pi / (u_val_local ** 2))

            plt.axhline(
                level_val,
                linestyle="--",
                color="black",
                linewidth=2,
                alpha=0.6,
            )

    if scale_selector.value == "log":
        plt.yscale("log")

    plt.grid()
    if legend_position_selector.value == "inside":
        plt.legend()

    elif legend_position_selector.value == "right":
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    else:  # outside
        plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1))
    plt.gca()
    return


@app.cell
def _(legend_position_selector, plt, slope_output):
    plt.figure()

    for entry_plot2 in slope_output:
        plt.plot(entry_plot2["t_array"], entry_plot2["slope_array"], label=entry_plot2["label_local"])

    plt.grid()
    if legend_position_selector.value == "inside":
        plt.legend()

    elif legend_position_selector.value == "right":
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    else:  # outside
        plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1))
    plt.gca()
    return


@app.cell
def _(fit_output, legend_position_selector, np, plt, scale_selector):
    plt.figure()

    for entry_plot3 in fit_output:
        t_vals_local = entry_plot3["t_array"]
        y_vals_local = entry_plot3["y_array"]

        line_main, = plt.plot(t_vals_local, y_vals_local, alpha=0.6)

        fit_val_local = entry_plot3["fit"]

        if fit_val_local is not None:
            slope_val_local, t_start_val, t_end_val = fit_val_local

            mask_val_local = (t_vals_local >= t_start_val) & (t_vals_local <= t_end_val)

            color_local = line_main.get_color()

            plt.plot(
                t_vals_local[mask_val_local],
                y_vals_local[mask_val_local],
                linewidth=4,
                color=color_local,
                label=f"{entry_plot3['label_local']} γ={slope_val_local:.3f}",
            )

            u_val_local3 = entry_plot3["u_val"]

            level_pred = np.exp(2 * np.pi * (slope_val_local / 2) / (u_val_local3 ** 2))

            plt.axhline(
                level_pred,
                linestyle="--",
                color=color_local,
                linewidth=2,
            )

    if scale_selector.value == "log":
        plt.yscale("log")

    plt.grid()
    if legend_position_selector.value == "inside":
        plt.legend()

    elif legend_position_selector.value == "right":
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    else:  # outside
        plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1))
    plt.gca()
    return


@app.cell
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

    Lambda_num = data[:, 0]
    gamma_num = data[:, 1]

    def gamma_interp(L):
        return np.interp(L, Lambda_num, gamma_num)

    return Lambda_num, gamma_interp, gamma_num


@app.cell
def _(mo):
    param_mode_selector = mo.ui.radio(
        options=["vs_u", "vs_Lambda"],
        value="vs_u",
        label="increment plot mode",
    )

    normalized_selector = mo.ui.checkbox(
        value=False,
        label="normalize by gamma_1Dz_num",
    )
    normalized_selector, param_mode_selector
    return normalized_selector, param_mode_selector


@app.cell
def _(
    df_filtered,
    find_increment_internal,
    max_len_selector,
    min_len_selector,
    tmax_selector,
    tmin_selector,
    ts_selector,
):
    increment_param_data = []

    if ts_selector.value is not None:
        df_tmp = df_filtered[df_filtered["ts_name"] == ts_selector.value]

        grouped = df_tmp.groupby(["u", "Lambda"])

        for (u_val, L_val), df_group_2 in grouped:
            t_arr = df_group_2["t"].values
            y_arr = df_group_2["value"].values

            res = find_increment_internal(
                t_arr,
                y_arr,
                min_len_selector.value,
                max_len_selector.value,
                tmin_selector.value,
                tmax_selector.value,
            )

            if res is None:
                continue

            slope_val, _, _ = res

            increment_param_data.append(
                {
                    "u": u_val,
                    "Lambda": L_val,
                    "gamma": slope_val,
                }
            )
    return (increment_param_data,)


@app.cell
def _(
    Lambda_num,
    gamma_interp,
    gamma_num,
    increment_param_data,
    legend_position_selector,
    normalized_selector,
    np,
    param_mode_selector,
    plt,
):
    fig, ax = plt.subplots()

    colors = plt.cm.tab10.colors

    def transform(gamma_vals, lam_vals):
        if not normalized_selector.value:
            return gamma_vals
        return gamma_vals / gamma_interp(lam_vals)

    # ========================
    # vs Lambda
    # ========================
    if param_mode_selector.value == "vs_Lambda":
        from collections import defaultdict
        grouped_by_u = defaultdict(list)  # Changed name

        for item in increment_param_data:
            grouped_by_u[item["u"]].append(item)

        for i, (current_u_val, items) in enumerate(grouped_by_u.items()):  # Changed name
            items_sorted = sorted(items, key=lambda x: x["Lambda"])

            lam = np.array([it["Lambda"] for it in items_sorted])
            gamma_vals = np.array([it["gamma"] for it in items_sorted])

            color = colors[i % len(colors)]

            ax.plot(
                lam,
                transform(gamma_vals, lam),
                color=color,
                linestyle="-",
                label=f"u={current_u_val}",
            )

        lam_min = np.min([it["Lambda"] for it in increment_param_data])
        lam_max = np.max([it["Lambda"] for it in increment_param_data])

        # аналитика + num только если НЕ нормировано
        if not normalized_selector.value:
            lam_grid = np.linspace(lam_min, lam_max, 300)

            gamma_approx = 2.0 / (1.0 + 2.0 * lam_grid / np.pi)

            ax.plot(
                lam_grid,
                gamma_approx,
                "k--",
                label=r"$2\gamma_{1Dz}^{approx}$",
            )

            mask = (Lambda_num >= lam_min) & (Lambda_num <= lam_max)

            ax.plot(
                Lambda_num[mask],
                gamma_num[mask],
                color="gray",
                linestyle="-",
                label=r"$2\gamma_{1Dz}^{num}$",
            )

        ax.set_xlabel("Lambda")

    # ========================
    # vs u
    # ========================
    else:
        from collections import defaultdict
        grouped_by_lambda = defaultdict(list)  # Changed name

        for item in increment_param_data:
            grouped_by_lambda[item["Lambda"]].append(item)

        for i, (lam_val, items) in enumerate(grouped_by_lambda.items()):
            items_sorted = sorted(items, key=lambda x: x["u"])

            u_vals = np.array([it["u"] for it in items_sorted])
            gamma_vals = np.array([it["gamma"] for it in items_sorted])
            lam_array = np.full_like(u_vals, lam_val)

            color = colors[i % len(colors)]

            ax.plot(
                u_vals,
                transform(gamma_vals, lam_array),
                color=color,
                linestyle="-",
                label=f"Λ={lam_val}",
            )

        ax.set_xlabel("u")

    # ========================
    # оформление
    # ========================
    if normalized_selector.value:
        ax.set_ylabel(r"$\gamma / \gamma_{1Dz}^{num}$")
    else:
        ax.set_ylabel("gamma")

    ax.grid()

    if legend_position_selector.value == "inside":
        ax.legend()
    elif legend_position_selector.value == "right":
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    else:
        ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1))

    fig
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

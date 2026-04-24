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
def _(mo):
    param_mode_selector = mo.ui.radio(
        options=["vs_u", "vs_Lambda"],
        value="vs_u",
        label="increment plot mode",
    )

    param_mode_selector
    return (param_mode_selector,)


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
    increment_param_data,
    legend_position_selector,
    param_mode_selector,
    plt,
):
    plt.figure()

    if param_mode_selector.value == "vs_u":
        # группируем по Lambda
        from collections import defaultdict
        grouped_by_lambda = defaultdict(list)  # Changed name

        for item in increment_param_data:
            grouped_by_lambda[item["Lambda"]].append(item)

        for lambda_val, items in grouped_by_lambda.items():
            items_sorted = sorted(items, key=lambda x: x["u"])

            u_vals = [it["u"] for it in items_sorted]
            gamma_vals = [it["gamma"] for it in items_sorted]

            plt.plot(u_vals, gamma_vals, label=f"Λ={lambda_val}")

        plt.xlabel("u")

    else:
        # группируем по u
        from collections import defaultdict
        grouped_by_u = defaultdict(list)  # Changed name

        for item in increment_param_data:
            grouped_by_u[item["u"]].append(item)

        for current_u_val, items in grouped_by_u.items():  # Changed variable name
            items_sorted = sorted(items, key=lambda x: x["Lambda"])

            L_vals = [it["Lambda"] for it in items_sorted]
            gamma_vals = [it["gamma"] for it in items_sorted]

            plt.plot(L_vals, gamma_vals, label=f"u={current_u_val}")

        plt.xlabel("Lambda")

    plt.ylabel("gamma")
    plt.grid()

    if legend_position_selector.value == "inside":
        plt.legend()
    elif legend_position_selector.value == "right":
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    else:
        plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1))

    plt.gca()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

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
    return fixed_L_selector, fixed_u_selector


@app.cell
def _(mo):
    scale_selector = mo.ui.radio(options=["linear", "log"], value="linear")
    min_len_selector = mo.ui.slider(start=3, stop=50, value=5)
    min_len_selector, scale_selector
    return min_len_selector, scale_selector


@app.cell
def _(np):
    def find_increment_internal(t_arr_internal, y_arr_internal, min_len_internal):
        mask_internal = y_arr_internal > 0
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
def _(df_data, fixed_L_selector, fixed_u_selector, mode_selector, ts_selector):
    grouped_output = []

    if ts_selector.value is not None:
        df_tmp = df_data[df_data["ts_name"] == ts_selector.value]

        if mode_selector.value == "fixed_u":
            df_tmp = df_tmp[df_tmp["u"] == fixed_u_selector.value]
            grouped_iter = df_tmp.groupby("Lambda")
        else:
            df_tmp = df_tmp[df_tmp["Lambda"] == fixed_L_selector.value]
            grouped_iter = df_tmp.groupby("u")

        for key_local, df_group in grouped_iter:
            grouped_output.append(
                {
                    "label_local": str(key_local),
                    "t_array": df_group["t"].values,
                    "y_array": df_group["value"].values,
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
def _(find_increment_internal, grouped_output, min_len_selector):
    fit_output = []

    for entry_fit in grouped_output:
        result_fit = find_increment_internal(
            entry_fit["t_array"],
            entry_fit["y_array"],
            min_len_selector.value,
        )

        fit_output.append(
            {
                "label_local": entry_fit["label_local"],
                "t_array": entry_fit["t_array"],
                "y_array": entry_fit["y_array"],
                "fit": result_fit,
            }
        )
    return (fit_output,)


@app.cell
def _(grouped_output, plt, scale_selector):
    plt.figure()

    for entry_plot in grouped_output:
        plt.plot(entry_plot["t_array"], entry_plot["y_array"], label=entry_plot["label_local"])

    if scale_selector.value == "log":
        plt.yscale("log")

    plt.grid()
    plt.legend()
    plt.gca()
    return


@app.cell
def _(plt, slope_output):
    plt.figure()

    for entry_plot2 in slope_output:
        plt.plot(entry_plot2["t_array"], entry_plot2["slope_array"], label=entry_plot2["label_local"])

    plt.grid()
    plt.legend()
    plt.gca()
    return


@app.cell
def _(fit_output, plt):
    plt.figure()

    for entry_plot3 in fit_output:
        t_vals = entry_plot3["t_array"]
        y_vals = entry_plot3["y_array"]

        plt.plot(t_vals, y_vals, alpha=0.3)

        fit_val = entry_plot3["fit"]
        if fit_val is not None:
            slope_val, t_start_val, t_end_val = fit_val

            mask_val = (t_vals >= t_start_val) & (t_vals <= t_end_val)

            plt.plot(
                t_vals[mask_val],
                y_vals[mask_val],
                linewidth=4,
                label=f"{entry_plot3['label_local']} γ={slope_val:.3f}",
            )

    plt.grid()
    plt.legend()
    plt.gca()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

import marimo

__generated_with = "0.23.2"
app = marimo.App(layout_file="layouts/max_growth_rate_calc.grid.json")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from pde_analysis.db_analysis.load_timeseries import load_timeseries_dataframe
    from pde_analysis.db_analysis.preprocess import normalize_parameters

    return load_timeseries_dataframe, mo, normalize_parameters, np, plt


@app.cell(hide_code=True)
def _(mo):
    db_path = mo.ui.text(
        value="data/db/simulations.db",
        label="db path",
        full_width=True,
    )
    db_path
    return (db_path,)


@app.cell(hide_code=True)
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
            r.lambda_,
            r.u,
            r.tend,
            r.x0,
            r.range_x,
            r.range_y,
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


@app.cell(hide_code=True)
def _(experiment_ui, load_timeseries_dataframe, normalize_parameters):
    df_data = load_timeseries_dataframe(experiment_name=experiment_ui.value)
    df_data = normalize_parameters(df_data, ndigits=5)
    return (df_data,)


@app.cell(hide_code=True)
def _(df_data, visible_lambda_selector, visible_u_selector):
    df_filtered = df_data[
        (df_data["u"].isin(visible_u_selector.value)) &
        (df_data["lambda_"].isin(visible_lambda_selector.value))
    ]

    df_filtered
    return (df_filtered,)


@app.cell(hide_code=True)
def _(df_filtered):
    df_tmax = (
        df_filtered
        .groupby("run_id", as_index=False)
        .agg(
            u=("u", "first"),
            lambda_=("lambda_", "first"),
            tmax=("t", "max"),
        )
    )
    duplicates = (
        df_tmax
        .groupby(["u", "lambda_"])
        .filter(lambda x: len(x) > 1)
        .sort_values(["u", "lambda_", "tmax"])
    )
    duplicates
    return


@app.cell(hide_code=True)
def _(df_filtered):
    excluded_run_ids = [29, 17, 116, 20, 21,7,45,46,47, 48, 49, 50, 51]
    df_filtered_selected = df_filtered[
        ~df_filtered["run_id"].isin(excluded_run_ids)
    ]
    return (df_filtered_selected,)


@app.cell(hide_code=True)
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
def _(mode_selector):
    if mode_selector.value == "fixed_u":
        char_label = r"\Lambda"
    else:
        char_label = "u"
    return (char_label,)


@app.cell
def _(df_data, mo):
    u_vals_local = sorted(df_data["u"].dropna().unique())
    lambda_vals_local = sorted(df_data["lambda_"].dropna().unique())

    fixed_u_selector = mo.ui.dropdown(options=u_vals_local, value=u_vals_local[0])
    fixed_lambda_selector = mo.ui.dropdown(options=lambda_vals_local, value=lambda_vals_local[0])

    visible_u_selector = mo.ui.multiselect(options=u_vals_local, value=u_vals_local)
    visible_lambda_selector = mo.ui.multiselect(options=lambda_vals_local, value=lambda_vals_local)
    fixed_lambda_selector, fixed_u_selector, visible_u_selector, visible_lambda_selector
    return (
        fixed_lambda_selector,
        fixed_u_selector,
        visible_lambda_selector,
        visible_u_selector,
    )


@app.cell(hide_code=True)
def _(mo):
    scale_selector = mo.ui.radio(options=["linear", "log"], value="log")

    min_len_selector = mo.ui.slider(start=2, stop=50, value=3)
    max_len_selector = mo.ui.slider(start=4, stop=100, value=10)

    tmin_selector = mo.ui.number(value=0.0, step=0.1, label="t_min")
    tmax_selector = mo.ui.number(value=15.0, label="t_max")

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


@app.cell(hide_code=True)
def _(mo):
    t_amp_min_selector = mo.ui.number(value=0.0, step=0.1, label="t_amp_min")
    t_amp_max_selector = mo.ui.number(value=60.0, label="t_amp_max")
    t_amp_min_selector, t_amp_max_selector
    return t_amp_max_selector, t_amp_min_selector


@app.cell(hide_code=True)
def _(np):
    from scipy.signal import find_peaks

    def find_increment_internal(
        t_arr_internal,
        y_arr_internal,
        min_len_internal,
        max_len_internal,
        tmin_internal,
        tmax_internal,
        peak_number=2,  # New parameter: which peak to select (1=first, 2=second, etc.)
        half_width=0.1,  # Half-width around the peak for t_start and t_end
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
        dlog_y_internal = np.gradient(log_y_internal, t_arr_internal)

        # Find peaks in the derivative
        peaks, _ = find_peaks(dlog_y_internal)

        if len(peaks) > 0:

            sorted_peak_indices = peaks

            if peak_number <= len(sorted_peak_indices):
                peak_idx = int(sorted_peak_indices[peak_number - 1])
                slope = float(dlog_y_internal[peak_idx])
                t_peak = float(t_arr_internal[peak_idx])
                # Set t_start and t_end with half_width around the peak
                t_start = max(t_peak - half_width, t_arr_internal[0])
                t_end = min(t_peak + half_width, t_arr_internal[-1])
            else:
                # If not enough peaks, take global max
                peak_idx = int(np.nanargmax(dlog_y_internal))
                slope = float(dlog_y_internal[peak_idx])
                t_peak = float(t_arr_internal[peak_idx])
                t_start = max(t_peak - half_width, t_arr_internal[0])
                t_end = min(t_peak + half_width, t_arr_internal[-1])
        else:
            # If no peaks found, take global max
            peak_idx = int(np.nanargmax(dlog_y_internal))
            slope = float(dlog_y_internal[peak_idx])
            t_peak = float(t_arr_internal[peak_idx])
            t_start = max(t_peak - half_width, t_arr_internal[0])
            t_end = min(t_peak + half_width, t_arr_internal[-1])

        return (slope, t_start, t_end)

    return (find_increment_internal,)


@app.cell(hide_code=True)
def _(
    df_filtered_selected,
    fixed_lambda_selector,
    fixed_u_selector,
    mode_selector,
    ts_selector,
):
    grouped_output = []

    if ts_selector.value is not None:
        df_tmp_2 = df_filtered_selected[df_filtered_selected["ts_name"] == ts_selector.value]

        if mode_selector.value == "fixed_u":
            df_tmp_2 = df_tmp_2[df_tmp_2["u"] == fixed_u_selector.value]
            grouped_iter = df_tmp_2.groupby("lambda_")

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
            df_tmp_2 = df_tmp_2[df_tmp_2["lambda_"] == fixed_lambda_selector.value]
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


@app.cell(hide_code=True)
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

        # Remove duplicate time points (keep first occurrence)
        unique_t, unique_indices = np.unique(t_local, return_index=True)
        y_local = y_local[unique_indices]
        t_local = unique_t

        # Optional: sort by time (np.unique already sorts)
        # If you need to maintain original order and remove duplicates only:
        # _, unique_indices = np.unique(t_local, return_index=True)
        # unique_indices = np.sort(unique_indices)  # preserve order
        # t_local = t_local[unique_indices]
        # y_local = y_local[unique_indices]

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
    return slope_output, slope_vals


@app.cell(hide_code=True)
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
            peak_number=2
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


@app.cell(hide_code=True)
def _(
    char_label,
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
            entry_plot["t_array"],  # ось абсцисс: ν₀ t
            entry_plot["y_array"],  # ось ординат: ∫∫ dx dz |a|²
            label=fr"${char_label}={entry_plot['label_local']}$"
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
    plt.xlabel(r'$\nu_0 t$')  # подпись оси абсцисс
    plt.ylabel(r'$W_a$')  # подпись оси ординат

    if legend_position_selector.value == "inside":
        plt.legend()

    elif legend_position_selector.value == "right":
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    else:  # outside
        plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1))
    plt.gca()
    return


@app.cell(hide_code=True)
def _(
    char_label,
    fit_output,
    legend_position_selector,
    np,
    plt,
    slope_output,
    slope_vals,
):
    plt.figure()

    # Создаем словарь для быстрого доступа к fit данным по label
    fit_dict = {entry["label_local"]: entry["fit"] for entry in fit_output if "fit" in entry}
    slope_vals
    for entry_plot2 in slope_output:
        t_vals_slope = entry_plot2["t_array"]
        _slope_vals = entry_plot2["slope_array"]
        label = entry_plot2["label_local"]

        # Основная линия
        line_main_slope, = plt.plot(t_vals_slope, _slope_vals, label=fr"${char_label}={label}$", alpha=0.6)

        color_slope = line_main_slope.get_color()

        # Пытаемся получить fit данные из fit_output
        if label in fit_dict:
            fit_data = fit_dict[label]
            if fit_data is not None and len(fit_data) >= 3:
                slope_val_max, t_start_slope, t_end_slope = fit_data

                # Маска для выделенного участка
                mask_slope = (t_vals_slope >= t_start_slope) & (t_vals_slope <= t_end_slope)

                # Выделяем участок
                plt.plot(
                    t_vals_slope[mask_slope],
                    _slope_vals[mask_slope],
                    linewidth=4,
                    color=color_slope,
                )

        else:
            # Если fit данных нет, находим максимум из массива
            max_slope_idx = np.argmax(_slope_vals)
            max_slope_value = _slope_vals[max_slope_idx]

            # Выделяем окрестность максимума
            half_window = 1
            start_idx = max(0, max_slope_idx - half_window)
            end_idx = min(len(t_vals_slope), max_slope_idx + half_window + 1)

            mask_slope = np.zeros(len(t_vals_slope), dtype=bool)
            mask_slope[start_idx:end_idx] = True

            plt.plot(
                t_vals_slope[mask_slope],
                _slope_vals[mask_slope],
                linewidth=4,
                color=color_slope,
            )

    plt.grid()
    if legend_position_selector.value == "inside":
        plt.legend()
    elif legend_position_selector.value == "right":
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    else:
        plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1))

    plt.xlabel(r'$\nu_0 t$')
    plt.ylabel(r'$d(\ln W_a)/dt$')
    plt.gca()
    return


@app.cell(hide_code=True)
def _(
    char_label,
    fit_output,
    legend_position_selector,
    np,
    plt,
    scale_selector,
):
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
                label=fr"${char_label}={entry_plot3['label_local']} ,\; 2γ={slope_val_local:.3f}$",
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

    plt.xlabel(r'$\nu_0 t$')  # подпись оси абсцисс
    plt.ylabel(r'$W_a$')  # подпись оси ординат
    plt.gca()
    return


@app.cell(hide_code=True)
def _(df_filtered, np, t_amp_max_selector, t_amp_min_selector, ts_selector):
    wsat_param_data = []

    if ts_selector.value is not None:
        _df_tmp = df_filtered[df_filtered["ts_name"] == ts_selector.value]

        _grouped = _df_tmp.groupby(["u", "lambda_"])

        for (_u_val, _lambda_val), _df_group_2 in _grouped:
            _t_arr = _df_group_2["t"].values
            _y_arr = _df_group_2["value"].values

            _mask_amp = (
                (_t_arr >= t_amp_min_selector.value) &
                (_t_arr <= t_amp_max_selector.value)
            )
            _t_amp = _t_arr[_mask_amp]
            _y_amp = _y_arr[_mask_amp]

            if len(_t_amp) == 0:
                continue

            _wsat_val = float(np.nanmax(_y_amp))
            wsat_param_data.append(
                {
                    "u": _u_val,
                    "lambda_": _lambda_val,
                    "Wsat": _wsat_val,
                }
            )
    return (wsat_param_data,)


@app.cell(hide_code=True)
def _(legend_position_selector, np, plt, wsat_param_data):
    from collections import defaultdict as _defaultdict

    _fig, _ax = plt.subplots()
    _grouped_by_lambda = _defaultdict(list)

    for _item in wsat_param_data:
        _grouped_by_lambda[_item["lambda_"]].append(_item)

    for _i, (_lam_val, _items) in enumerate(sorted(_grouped_by_lambda.items(), key=lambda x: x[0])):
        _items_sorted = sorted(_items, key=lambda x: x["u"])
        _u_vals = np.array([it["u"] for it in _items_sorted], dtype=float)
        _wsat_vals = np.array([np.log(it["Wsat"]) for it in _items_sorted], dtype=float)
        _ax.plot(_u_vals, _wsat_vals, marker="o", label=f"Λ={_lam_val}")

    u_min = np.min([item["u"] for item in wsat_param_data])
    u_max = np.max([item["u"] for item in wsat_param_data])
    _u_ref = np.linspace(u_min, u_max, 200)
    _ref_level = np.log(np.exp(2 * np.pi / _u_ref ** 2))

    _ax.plot(_u_ref,_ref_level, color="black", linestyle="--", linewidth=2, label=r"$2\pi/u^2$")

    _ax.set_xlabel("u")
    _ax.set_ylabel(r"$\log(W_{sat})$")
    _ax.grid()

    if legend_position_selector.value == "inside":
        _ax.legend()
    elif legend_position_selector.value == "right":
        _ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    else:
        _ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1))
    _fig
    return u_max, u_min


@app.cell(hide_code=True)
def _(legend_position_selector, np, plt, u_max, u_min, wsat_param_data):
    from collections import defaultdict as _defaultdict

    _fig2, _ax2 = plt.subplots()
    _grouped_by_lambda = _defaultdict(list)

    for _item in wsat_param_data:
        _grouped_by_lambda[_item["lambda_"]].append(_item)

    for _i, (_lam_val, _items) in enumerate(sorted(_grouped_by_lambda.items(), key=lambda x: x[0])):
        _items_sorted = sorted(_items, key=lambda x: x["u"])
        _u_vals = np.array([it["u"] for it in _items_sorted], dtype=float)
        _wsat_vals = np.array([np.log(it["Wsat"]) for it in _items_sorted], dtype=float)

        # Новая координата X: 2π/u²
        _x_vals = 2 * np.pi / (_u_vals ** 2)

        _ax2.plot(_x_vals, _wsat_vals, marker="o", label=f"Λ={_lam_val}")

    # Эталонная прямая линия (y = x, так как log(exp(2π/u²)) = 2π/u²)
    _x_ref = np.linspace(
        2 * np.pi / (u_max ** 2),  # минимальное значение при максимальном u
        2 * np.pi / (u_min ** 2),  # максимальное значение при минимальном u
        200
    )
    _y_ref = _x_ref  # так как log(exp(2π/u²)) = 2π/u²

    _ax2.plot(_x_ref, _y_ref, color="black", linestyle="--", linewidth=2, label=r"$exp(2\pi Z)$")

    _ax2.set_xlabel(r"$2\pi Z$")
    _ax2.set_ylabel(r"$\log(W_{sat})$")
    _ax2.grid()

    if legend_position_selector.value == "inside":
        _ax2.legend()
    elif legend_position_selector.value == "right":
        _ax2.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    else:
        _ax2.legend(loc="upper left", bbox_to_anchor=(1.05, 1))

    _fig2
    return


@app.cell(hide_code=True)
def _(legend_position_selector, np, plt, wsat_param_data):
    from collections import defaultdict as _defaultdict

    _fig, _ax = plt.subplots()
    _grouped_by_u = _defaultdict(list)

    for _item in wsat_param_data:
        _grouped_by_u[_item["u"]].append(_item)

    for _i, (_u_val, _items) in enumerate(sorted(_grouped_by_u.items(), key=lambda x: x[0])):
        _items_sorted = sorted(_items, key=lambda x: x["lambda_"])
        _lam_vals = np.array([it["lambda_"] for it in _items_sorted], dtype=float)
        _wsat_vals = np.array([np.log(it["Wsat"]) for it in _items_sorted], dtype=float)

        # Основная линия с точками
        _line, = _ax.plot(_lam_vals, _wsat_vals, marker="o", label=f"u={_u_val}")

        # Горизонтальная пунктирная линия на уровне 2π/u²
        _level = np.exp(2 * np.pi / (_u_val ** 2))
        _ax.axhline(
            y=np.log(_level),  # логарифм, так как по оси Y логарифм
            linestyle="--",
            color=_line.get_color(),
            linewidth=2,
            alpha=0.7
        )

    _ax.set_xlabel("Λ")
    _ax.set_ylabel(r"$\log(W_{sat})$")
    _ax.grid()

    if legend_position_selector.value == "inside":
        _ax.legend()
    elif legend_position_selector.value == "right":
        _ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    else:
        _ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1))

    _fig
    return


@app.cell(hide_code=True)
def _(
    gamma_eff_mode_selector,
    gamma_x_interp,
    increment_param_data,
    legend_position_selector,
    np,
    plt,
    wsat_param_data,
):
    _fig, _ax = plt.subplots()

    _wsat_lookup = {
        (_item["u"], _item["lambda_"]): _item["Wsat"]
        for _item in wsat_param_data
    }

    _lambda_colors = {
        _lam_val: color
        for _lam_val, color in zip(
            sorted({item["lambda_"] for item in increment_param_data}),
            plt.cm.tab10(np.linspace(0, 1, max(1, len({item["lambda_"] for item in increment_param_data})))),
        )
    }
    _u_markers = {
        _u_val: marker
        for _u_val, marker in zip(
            sorted({item["u"] for item in increment_param_data}),
            ["o", "s", "^", "D", "P", "X", "v", ">", "<", "p", "*", "h"],
        )
    }

    _x_vals = []
    _y_vals = []

    for _item in increment_param_data:
        _u_val = _item["u"]
        _lam_val = _item["lambda_"]
        _gamma_val = _item["gamma"]
        _wsat_val = _wsat_lookup.get((_u_val, _lam_val), np.nan)

        if gamma_eff_mode_selector.value == "gamma_num_2D":
            _gamma_eff_val = _gamma_val / 2.0
        else:
            _gamma_eff_val = _gamma_val / gamma_x_interp(_u_val)

        _x_val = 2 * np.pi / (_u_val ** 2) * (_gamma_eff_val ** 2)
        _y_val = np.log(_wsat_val)

        if np.isfinite(_x_val) and np.isfinite(_y_val):
            _x_vals.append(_x_val)
            _y_vals.append(_y_val)
            _ax.plot(
                _x_val,
                _y_val,
                linestyle="None",
                marker=_u_markers[_u_val],
                color=_lambda_colors[_lam_val],
                markersize=6,
                alpha=0.8,
            )

    if _x_vals and _y_vals:
        _x_arr = np.array(_x_vals, dtype=float)
        _y_arr = np.array(_y_vals, dtype=float)
        _coef = np.polyfit(_x_arr, _y_arr, 1)
        _fit_slope = _coef[0]
        _fit_intercept = _coef[1]
        _fit_vals = _fit_slope * _x_arr + _fit_intercept
        _ss_res = np.sum((_y_arr - _fit_vals) ** 2)
        _ss_tot = np.sum((_y_arr - np.mean(_y_arr)) ** 2)
        _r_squared = 1.0 - _ss_res / _ss_tot if _ss_tot > 0 else np.nan
        _x_fit = np.linspace(np.min(_x_arr), np.max(_x_arr), 200)
        _y_fit = _fit_slope * _x_fit + _fit_intercept
        _ax.plot(_x_fit, _y_fit, color="black", linewidth=2, linestyle="--")
        _ax.text(
            0.02,
            0.98,
            (
                f"slope = {_fit_slope:.3f} \n"
                f"intercept = {_fit_intercept:.3f} \n"
                rf"$R^2$ = {_r_squared:.3f}"
            ),
            transform=_ax.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        )

    if gamma_eff_mode_selector.value == "gamma_num_2D":
        _x_label = r"$2\pi / u^2 \cdot \gamma^2_{2D} (u,\Lambda)$"
    else:
        _x_label = r"$2\pi / u^2 \cdot (\gamma_{2D}(u,\Lambda)/\gamma_{1Dx}(u))^2$"

    _ax.set_xlabel(_x_label)
    _ax.set_ylabel(r"$\log(W_{sat})$")
    _ax.grid(True)

    if legend_position_selector.value == "inside":
        _ax.legend()
    elif legend_position_selector.value == "right":
        _ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    else:
        _ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1))

    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    param_mode_selector = mo.ui.radio(
        options=["vs_u", "vs_Lambda"],
        value="vs_u",
        label="increment plot mode",
    )

    normalized_selector = mo.ui.checkbox(
        value=False,
        label="normalize by gamma_1D",
    )

    normalization_type_selector = mo.ui.radio(
        options=["1Dz", "1Dx", "1Dz*1Dx"],
        value="1Dz",
        label="Normalization type",
    )

    gamma_eff_mode_selector = mo.ui.radio(
        options=["gamma_num_2D", "gamma2D_per_gamma1Dx"],
        value="gamma_num_2D",
        label="gamma_eff mode",
    )
    normalized_selector, param_mode_selector, normalization_type_selector, gamma_eff_mode_selector
    return (
        gamma_eff_mode_selector,
        normalization_type_selector,
        normalized_selector,
        param_mode_selector,
    )


@app.cell(hide_code=True)
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
        _df_tmp = df_filtered[df_filtered["ts_name"] == ts_selector.value]

        _grouped = _df_tmp.groupby(["u", "lambda_"])

        for (_u_val, _lambda_val), _df_group_2 in _grouped:
            _t_arr = _df_group_2["t"].values
            _y_arr = _df_group_2["value"].values

            _res = find_increment_internal(
                _t_arr,
                _y_arr,
                min_len_selector.value,
                max_len_selector.value,
                tmin_selector.value,
                tmax_selector.value,
            )

            if _res is None:
                continue

            _slope_val, _, _ = _res

            increment_param_data.append(
                {
                    "u": _u_val,
                    "lambda_": _lambda_val,
                    "gamma": _slope_val,
                }
            )
    return (increment_param_data,)


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

    def gamma_interp(L):
        return np.interp(L, lambda_num, gamma_num)

    return lambda_num, gamma_interp, gamma_num


@app.cell(hide_code=True)
def _(
    lambda_num,
    gamma_interp,
    gamma_num,
    increment_param_data,
    legend_position_selector,
    normalization_type_selector,
    normalized_selector,
    np,
    param_mode_selector,
    plt,
):
    from scipy.interpolate import interp1d
    from collections import defaultdict as _defaultdict

    # ---- Данные для 1Dx ----
    gauss_w0_0p5 = [
        [np.float64(0.4), 1.927632926231027],
        [np.float64(0.6), 1.8618887456236592],
        [np.float64(0.8), 1.7634897985058942],
        [np.float64(1.0), 1.6292176902361817],
        [np.float64(1.2), 1.470791177616892],
        [np.float64(1.4), 1.3056809640564424],
        [np.float64(1.6), 1.1497878424167567],
        [np.float64(1.8), 1.0114144684646131],
        [np.float64(2.0), 0.8936880369139644],
        [np.float64(2.2), 0.7956294618296091],
        [np.float64(2.4), 0.7142589925935026],
        [np.float64(2.6), 0.6471517894360832],
    ]
    u_x = np.array([p[0] for p in gauss_w0_0p5])
    gamma_x = np.array([p[1] for p in gauss_w0_0p5])
    gamma_x_interp = interp1d(u_x, gamma_x, kind='linear', fill_value='extrapolate')

    # ---- Функция нормировки ----
    def get_norm_factor(gamma, u, lam):
        if not normalized_selector.value:
            return gamma
        if normalization_type_selector.value == "1Dz":
            return gamma / gamma_interp(lam)
        elif normalization_type_selector.value == "1Dx":
            return gamma / gamma_x_interp(u)
        elif normalization_type_selector.value == "1Dz*1Dx":
            return 2*gamma / (gamma_interp(lam) * gamma_x_interp(u))
        else:
            return gamma

    _fig, _ax = plt.subplots()
    _num_colors = 12
    _colors = plt.cm.plasma(np.linspace(0, 1, _num_colors))

    # ========================
    # vs Lambda
    # ========================
    if param_mode_selector.value == "vs_Lambda":
        _grouped_by_u = _defaultdict(list)
        for _item in increment_param_data:
            _grouped_by_u[_item["u"]].append(_item)

        # Добавляем счетчик i для индексации цвета
        for _i, (current_u_val, _items) in enumerate(sorted(_grouped_by_u.items(), key=lambda x: x[0])):
            _items_sorted = sorted(_items, key=lambda x: x["lambda_"])
            _lam = np.array([it["lambda_"] for it in _items_sorted])
            _gamma_vals = np.array([it["gamma"] for it in _items_sorted])
            _color = _colors[_i % len(_colors)]

            _ax.plot(
                _lam,
                get_norm_factor(_gamma_vals, current_u_val, _lam),
                color=_color,
                linestyle="-",
                label=f"u={current_u_val}",
            )

        _lam_min = np.min([it["lambda_"] for it in increment_param_data])
        _lam_max = np.max([it["lambda_"] for it in increment_param_data])

        if not normalized_selector.value:
            _lam_grid = np.linspace(_lam_min, _lam_max, 300)
            _gamma_approx = 2.0 / (1.0 + 2.0 * _lam_grid / np.pi)
            _ax.plot(_lam_grid, _gamma_approx, "k--", label=r"$2\gamma_{1Dz}^{approx}$")
            _mask = (lambda_num >= _lam_min) & (lambda_num <= _lam_max)
            _ax.plot(lambda_num[_mask], gamma_num[_mask], color="gray", linestyle="-", label=r"$2\gamma_{1Dz}^{num}$")

        _ax.set_xlabel("Λ")

    # ========================
    # vs u (СОРТИРОВКА ПО ВОЗРАСТАНИЮ LAMBDA)
    # ========================
    else:
        _grouped_by_lambda = _defaultdict(list)
        for _item in increment_param_data:
            _grouped_by_lambda[_item["lambda_"]].append(_item)

        # СОРТИРОВКА по возрастанию Lambda (ключ сортировки)
        for _i, (_lam_val, _items) in enumerate(sorted(_grouped_by_lambda.items(), key=lambda x: x[0])):
            _items_sorted = sorted(_items, key=lambda x: x["u"])
            _u_vals = np.array([it["u"] for it in _items_sorted])
            _gamma_vals = np.array([it["gamma"] for it in _items_sorted])
            _color = _colors[_i % len(_colors)]

            _ax.plot(
                _u_vals,
                get_norm_factor(_gamma_vals, _u_vals, _lam_val),
                color=_color,
                linestyle="-",
                label=f"Λ={_lam_val}",
            )

        _ax.set_xlabel("u")

    # ========================
    # оформление
    # ========================
    if normalized_selector.value:
        if normalization_type_selector.value == "1Dz":
            _ax.set_ylabel(r"$\gamma / \gamma_{1Dz}^{num}$")
        elif normalization_type_selector.value == "1Dx":
            _ax.set_ylabel(r"$\gamma / \gamma_{1Dx}^{num}$")
        elif normalization_type_selector.value == "1Dz*1Dx":
            _ax.set_ylabel(r"$\gamma / (\gamma_{1Dz}^{num} \gamma_{1Dx}^{num})$")
    else:
        _ax.set_ylabel(r"$2\gamma$")

    _ax.grid()

    if legend_position_selector.value == "inside":
        _ax.legend()
    elif legend_position_selector.value == "right":
        _ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    else:
        _ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1))
    _fig
    return (gamma_x_interp,)


if __name__ == "__main__":
    app.run()

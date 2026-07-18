import marimo

__generated_with = "0.23.14"
app = marimo.App()


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import sqlite3
    import pandas as pd
    import json

    return json, mo, np, pd, plt, sqlite3


@app.cell(hide_code=True)
def _(mo):
    db_path = mo.ui.text(
        value="data/db/simulations.db",
        label="Путь к БД",
        full_width=True,
    )
    db_path
    return (db_path,)


@app.cell(hide_code=True)
def _(db_path, mo, pd, sqlite3):
    def load_table(db_path, query):
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
            r.status
        FROM runs r
        JOIN experiments e ON r.experiment_id = e.experiment_id
        ORDER BY e.name, r.run_id
        """,
    )

    experiment_names_loaded = (
        sorted(runs_df["experiment_name"].dropna().unique().tolist())
        if not runs_df.empty
        else []
    )

    exp1_ui = mo.ui.dropdown(
        options=experiment_names_loaded,
        value=experiment_names_loaded[0] if len(experiment_names_loaded) > 0 else None,
        label="Серия 1",
    )
    exp2_ui = mo.ui.dropdown(
        options=experiment_names_loaded,
        value=experiment_names_loaded[1] if len(experiment_names_loaded) > 1 else None,
        label="Серия 2",
    )
    exp1_ui, exp2_ui
    return exp1_ui, exp2_ui


@app.cell(hide_code=True)
def _(db_path, exp1_ui, exp2_ui, json, pd, sqlite3):
    def load_timeseries_for_experiment(experiment_name, db_path):
        conn = sqlite3.connect(db_path)
        query = """
        SELECT
            r.run_id,
            r.u,
            r.lambda_,
            r.file_name,
            ts.name AS ts_name,
            ts.t_values,
            ts.y_values
        FROM runs r
        JOIN experiments e ON r.experiment_id = e.experiment_id
        JOIN timeseries ts ON ts.run_id = r.run_id
        WHERE e.name = ?
        """
        df = pd.read_sql_query(query, conn, params=(experiment_name,))
        conn.close()
        if df.empty:
            return df

        records = []
        for _, row in df.iterrows():
            t_vals = json.loads(row["t_values"])
            y_vals = json.loads(row["y_values"])
            for t, y in zip(t_vals, y_vals):
                records.append(
                    {
                        "run_id": row["run_id"],
                        "u": round(float(row["u"]), 5),
                        "lambda_": round(float(row["lambda_"]), 5),
                        "file_name": row["file_name"],
                        "ts_name": row["ts_name"],
                        "t": t,
                        "value": y,
                    }
                )
        return pd.DataFrame(records)

    df1 = load_timeseries_for_experiment(exp1_ui.value, db_path.value)
    df2 = load_timeseries_for_experiment(exp2_ui.value, db_path.value)
    return df1, df2


@app.cell(hide_code=True)
def _(df1, df2):
    params1 = set(zip(df1["u"], df1["lambda_"])) if not df1.empty else set()
    params2 = set(zip(df2["u"], df2["lambda_"])) if not df2.empty else set()
    common_params = sorted(params1 & params2)
    common_params
    return (common_params,)


@app.cell(hide_code=True)
def _(common_params, df1, df2, mo, pd):
    all_u = sorted(
        float(v)
        for v in pd.concat([df1["u"], df2["u"]]).dropna().unique()
    )
    all_lambda = sorted(
        float(v)
        for v in pd.concat([df1["lambda_"], df2["lambda_"]]).dropna().unique()
    )

    # Ограничиваем u/Λ до общих параметров (для Plot 1)
    common_u = sorted({float(p[0]) for p in common_params})
    common_lambda = sorted({float(p[1]) for p in common_params})

    all_ts = sorted(
        pd.concat([df1["ts_name"], df2["ts_name"]]).dropna().unique().tolist()
    )

    mode_selector = mo.ui.radio(
        options=["fixed_u", "fixed_Lambda"],
        value="fixed_u",
        label="Режим",
    )

    fixed_u_selector = mo.ui.dropdown(
        options=common_u, value=common_u[0] if common_u else None,
        label="Фикс. u (меняется Λ)",
    )
    fixed_lambda_selector = mo.ui.dropdown(
        options=common_lambda, value=common_lambda[0] if common_lambda else None,
        label="Фикс. Λ (меняется u)",
    )

    visible_u_selector = mo.ui.multiselect(
        options=common_u, value=common_u,
        label="Видимые u",
    )
    visible_lambda_selector = mo.ui.multiselect(
        options=common_lambda, value=common_lambda,
        label="Видимые Λ",
    )

    ts_selector = mo.ui.dropdown(
        options=all_ts,
        value=all_ts[1] if len(all_ts) > 1 else (all_ts[0] if all_ts else None),
        label="Временной ряд",
    )

    scale_selector = mo.ui.radio(
        options=["linear", "log"], value="log", label="Масштаб Y",
    )
    tmin_selector = mo.ui.number(value=0.0, step=0.1, label="t_min")
    tmax_selector = mo.ui.number(value=15.0, label="t_max")
    min_tmax_selector = mo.ui.number(
        value=5.0, step=0.5, label="min t_max (фильтр расчётов)"
    )

    legend_position_selector = mo.ui.radio(
        options=["inside", "right", "outside"],
        value="inside",
        label="Положение легенды",
    )

    t_amp_min_selector = mo.ui.number(value=0.0, step=0.1, label="t_amp_min (Wsat)")
    t_amp_max_selector = mo.ui.number(value=60.0, label="t_amp_max (Wsat)")

    gamma_eff_mode_selector = mo.ui.radio(
        options=["gamma_num_2D", "gamma2D_per_gamma1Dx", "gamma_1Dz"],
        value="gamma_1Dz",
        label="γ_eff mode (Plot 3)",
    )

    (
        mode_selector,
        fixed_lambda_selector,
        fixed_u_selector,
        visible_u_selector,
        visible_lambda_selector,
        ts_selector,
        scale_selector,
        tmin_selector,
        tmax_selector,
        min_tmax_selector,
        legend_position_selector,
        t_amp_min_selector,
        t_amp_max_selector,
        gamma_eff_mode_selector,
    )
    return (
        fixed_lambda_selector,
        fixed_u_selector,
        legend_position_selector,
        min_tmax_selector,
        mode_selector,
        scale_selector,
        t_amp_max_selector,
        t_amp_min_selector,
        ts_selector,
        visible_lambda_selector,
        visible_u_selector,
    )


@app.cell(hide_code=True)
def _(common_params, df1, df2, visible_lambda_selector, visible_u_selector):
    def filter_common(df, common_set):
        if df.empty:
            return df
        _mask = df.apply(
            lambda row: (row["u"], row["lambda_"]) in common_set, axis=1
        )
        return df[_mask]

    df1c = filter_common(df1, common_params)
    df2c = filter_common(df2, common_params)

    # Фильтр по видимым u/Λ
    _visible_u = set(visible_u_selector.value) if visible_u_selector.value else set()
    _visible_lam = (
        set(visible_lambda_selector.value) if visible_lambda_selector.value else set()
    )

    if _visible_u:
        df1c = df1c[df1c["u"].isin(_visible_u)]
        df2c = df2c[df2c["u"].isin(_visible_u)]
    if _visible_lam:
        df1c = df1c[df1c["lambda_"].isin(_visible_lam)]
        df2c = df2c[df2c["lambda_"].isin(_visible_lam)]
    return df1c, df2c


@app.cell(hide_code=True)
def _(df1c, df2c, min_tmax_selector):
    def filter_by_tmax(df, min_tmax):
        if df.empty:
            return df
        _tmax = df.groupby("run_id")["t"].max().reset_index()
        _keep = set(_tmax.loc[_tmax["t"] >= min_tmax, "run_id"])
        return df[df["run_id"].isin(_keep)]

    df1f = filter_by_tmax(df1c, min_tmax_selector.value)
    df2f = filter_by_tmax(df2c, min_tmax_selector.value)
    return df1f, df2f


@app.cell(hide_code=True)
def _(mode_selector):
    if mode_selector.value == "fixed_u":
        char_label = r"\Lambda"
    else:
        char_label = "u"
    return (char_label,)


@app.cell(hide_code=True)
def _(
    df1f,
    df2f,
    fixed_lambda_selector,
    fixed_u_selector,
    mode_selector,
    ts_selector,
):
    def group_for_plot(df, ts_name, mode, fixed_u, fixed_lambda):
        if df.empty or ts_name is None:
            return []
        _df = df[df["ts_name"] == ts_name]
        if _df.empty:
            return []

        out = []
        if mode == "fixed_u":
            _df = _df[_df["u"] == fixed_u]
            for key, grp in _df.groupby("lambda_"):
                out.append(
                    {
                        "label_local": str(round(float(key), 4)),
                        "t_array": grp["t"].values,
                        "y_array": grp["value"].values,
                    }
                )
        else:
            _df = _df[_df["lambda_"] == fixed_lambda]
            for key, grp in _df.groupby("u"):
                out.append(
                    {
                        "label_local": str(round(float(key), 4)),
                        "t_array": grp["t"].values,
                        "y_array": grp["value"].values,
                    }
                )
        return out

    groups1 = group_for_plot(
        df1f, ts_selector.value, mode_selector.value,
        fixed_u_selector.value, fixed_lambda_selector.value,
    )
    groups2 = group_for_plot(
        df2f, ts_selector.value, mode_selector.value,
        fixed_u_selector.value, fixed_lambda_selector.value,
    )
    return groups1, groups2


@app.cell(hide_code=True)
def _(
    char_label,
    fixed_u_selector,
    groups1,
    groups2,
    legend_position_selector,
    mode_selector,
    np,
    plt,
    scale_selector,
):
    plt.figure(figsize=(10, 6))

    # Собираем все ключи из обеих серий для согласованных цветов
    _all_keys = sorted({e["label_local"] for e in groups1 + groups2})
    _key_colors = {
        k: color
        for k, color in zip(
            _all_keys,
            plt.cm.tab10(np.linspace(0, 1, max(1, len(_all_keys)))),
        )
    }

    # Series 1 — сплошная, прозрачная
    for entry in groups1:
        plt.plot(
            entry["t_array"],
            entry["y_array"],
            linestyle="-",
            linewidth=1.5,
            alpha=0.7,
            color=_key_colors.get(entry["label_local"], "gray"),
            label=fr"${char_label}={entry['label_local']}$ (1)",
        )

    # Series 2 — пунктирная
    for entry in groups2:
        plt.plot(
            entry["t_array"],
            entry["y_array"],
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
            color=_key_colors.get(entry["label_local"], "gray"),
            label=fr"${char_label}={entry['label_local']}$ (2)",
        )

    # Теоретический уровень
    if mode_selector.value == "fixed_u" and fixed_u_selector.value is not None:
        _u_val = float(fixed_u_selector.value)
        _level = np.exp(2 * np.pi / (_u_val ** 2))
        plt.axhline(_level, linestyle=":", color="black", linewidth=2, alpha=0.5,
                    label=rf"$e^{{2\pi/{_u_val:.2f}^2}}$")

    if scale_selector.value == "log":
        plt.yscale("log")

    plt.grid(alpha=0.3)
    plt.xlabel(r"$\nu_0 t$")
    plt.ylabel(r"$W_a$")

    if legend_position_selector.value == "inside":
        plt.legend(fontsize=9)
    elif legend_position_selector.value == "right":
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=9)
    else:
        plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1), fontsize=9)

    plt.gca()
    return


@app.cell(hide_code=True)
def _(df1f, df2f, np, t_amp_max_selector, t_amp_min_selector, ts_selector):
    def compute_wsat(df, ts_name, t_min, t_max):
        if df.empty or ts_name is None:
            return []
        _df = df[df["ts_name"] == ts_name]
        if _df.empty:
            return []
        out = []
        for (_u, _lam), grp in _df.groupby(["u", "lambda_"]):
            _t = grp["t"].values
            _y = grp["value"].values
            mask = (_t >= t_min) & (_t <= t_max)
            _y_amp = _y[mask]
            if len(_y_amp) == 0:
                continue
            out.append({
                "u": float(_u),
                "lambda_": float(_lam),
                "Wsat": float(np.nanmax(_y_amp)),
            })
        return out

    wsat1 = compute_wsat(df1f, ts_selector.value,
                          t_amp_min_selector.value, t_amp_max_selector.value)
    wsat2 = compute_wsat(df2f, ts_selector.value,
                          t_amp_min_selector.value, t_amp_max_selector.value)
    return wsat1, wsat2


@app.cell(hide_code=True)
def _(legend_position_selector, np, plt, wsat1, wsat2):
    from collections import defaultdict as _defaultdict

    _fig, _ax = plt.subplots(figsize=(8, 6))

    # Согласованные цвета по Λ
    _all_lam_vals = sorted({_it["lambda_"] for _it in wsat1 + wsat2})
    _lam_colors = {
        _lam: color
        for _lam, color in zip(
            _all_lam_vals,
            plt.cm.tab10(np.linspace(0, 1, max(1, len(_all_lam_vals)))),
        )
    }

    # Series 1 — сплошные линии
    _g1 = _defaultdict(list)
    for _item1 in wsat1:
        _g1[_item1["lambda_"]].append(_item1)

    for _i, (_lam, _items_grp) in enumerate(sorted(_g1.items(), key=lambda x: x[0])):
        _items_sorted = sorted(_items_grp, key=lambda x: x["u"])
        _u = np.array([_it["u"] for _it in _items_sorted], dtype=float)
        _x = 2 * np.pi / (_u ** 2)
        _y = np.array([np.log(_it["Wsat"]) for _it in _items_sorted], dtype=float)
        _ax.plot(_x, _y, marker="o", linestyle="-",
                 color=_lam_colors[_lam],
                 label=rf"$\Lambda={_lam}$ (1)")

    # Series 2 — пунктирные линии
    _g2 = _defaultdict(list)
    for _item2 in wsat2:
        _g2[_item2["lambda_"]].append(_item2)

    for _i, (_lam, _items_grp) in enumerate(sorted(_g2.items(), key=lambda x: x[0])):
        _items_sorted = sorted(_items_grp, key=lambda x: x["u"])
        _u = np.array([_it["u"] for _it in _items_sorted], dtype=float)
        _x = 2 * np.pi / (_u ** 2)
        _y = np.array([np.log(_it["Wsat"]) for _it in _items_sorted], dtype=float)
        _ax.plot(_x, _y, marker="s", linestyle="--",
                 color=_lam_colors[_lam],
                 label=rf"$\Lambda={_lam}$ (2)")

    # Теоретическая линия: y = 2π/u²  (т.е. y = x)
    if wsat1 or wsat2:
        _all_u_vals = list({_it["u"] for _it in wsat1} | {_it["u"] for _it in wsat2})
        _u_min_p2 = min(_all_u_vals)
        _u_max_p2 = max(_all_u_vals)
        _x_ref = np.linspace(2 * np.pi / (_u_max_p2 ** 2),
                             2 * np.pi / (_u_min_p2 ** 2), 200)
        _ax.plot(_x_ref, _x_ref, color="black", linestyle=":",
                 linewidth=2, label=r"$2\pi/u^2$ (теория)")

    _ax.set_xlabel(r"$2\pi / u^2$")
    _ax.set_ylabel(r"$\log(W_{sat})$")
    _ax.grid(alpha=0.3)

    if legend_position_selector.value == "inside":
        _ax.legend(fontsize=8)
    elif legend_position_selector.value == "right":
        _ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=8)
    else:
        _ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1), fontsize=8)

    _fig
    return


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

    # 1Dx interpolation
    from scipy.interpolate import interp1d
    gauss_w0_0p5 = [
        [0.4, 1.927632926231027],
        [0.6, 1.8618887456236592],
        [0.8, 1.7634897985058942],
        [1.0, 1.6292176902361817],
        [1.2, 1.470791177616892],
        [1.4, 1.3056809640564424],
        [1.6, 1.1497878424167567],
        [1.8, 1.0114144684646131],
        [2.0, 0.8936880369139644],
        [2.2, 0.7956294618296091],
        [2.4, 0.7142589925935026],
        [2.6, 0.6471517894360832],
    ]
    u_x = np.array([p[0] for p in gauss_w0_0p5])
    gamma_x = np.array([p[1] for p in gauss_w0_0p5])
    gamma_x_interp = interp1d(u_x, gamma_x, kind='linear', fill_value='extrapolate')
    return (gamma_interp,)


@app.cell(hide_code=True)
def _(
    df1,
    df2,
    min_tmax_selector,
    np,
    t_amp_max_selector,
    t_amp_min_selector,
    ts_selector,
    visible_lambda_selector,
    visible_u_selector,
):
    # Фильтр только по видимым u/Λ (без common_params)
    _vu = set(visible_u_selector.value) if visible_u_selector.value else set()
    _vl = set(visible_lambda_selector.value) if visible_lambda_selector.value else set()

    def _filter_visible(df):
        if df.empty:
            return df
        if _vu:
            df = df[df["u"].isin(_vu)]
        if _vl:
            df = df[df["lambda_"].isin(_vl)]
        return df

    def _filter_tmax(df, min_tmax):
        if df.empty:
            return df
        _tmax = df.groupby("run_id")["t"].max().reset_index()
        _keep = set(_tmax.loc[_tmax["t"] >= min_tmax, "run_id"])
        return df[df["run_id"].isin(_keep)]

    def _compute_wsat(df, ts_name, t_min, t_max):
        if df.empty or ts_name is None:
            return []
        _df = df[df["ts_name"] == ts_name]
        if _df.empty:
            return []
        out = []
        for (_u, _lam), grp in _df.groupby(["u", "lambda_"]):
            _t = grp["t"].values
            _y = grp["value"].values
            mask = (_t >= t_min) & (_t <= t_max)
            _y_amp = _y[mask]
            if len(_y_amp) == 0:
                continue
            out.append({"u": float(_u), "lambda_": float(_lam), "Wsat": float(np.nanmax(_y_amp))})
        return out

    df1_all = _filter_tmax(_filter_visible(df1), min_tmax_selector.value)
    df2_all = _filter_tmax(_filter_visible(df2), min_tmax_selector.value)

    wsat1_all = _compute_wsat(df1_all, ts_selector.value,
                               t_amp_min_selector.value, t_amp_max_selector.value)
    wsat2_all = _compute_wsat(df2_all, ts_selector.value,
                               t_amp_min_selector.value, t_amp_max_selector.value)
    return wsat1_all, wsat2_all


@app.cell(hide_code=True)
def _(np, wsat1_all, wsat2_all):
    _x_all = []
    _y_all = []
    for _it in wsat1_all + wsat2_all:
        _g = np.interp(_it["lambda_"], 
                       np.array([0.01, 20.0]), 
                       np.array([1.99, 0.13])) / 2.0
        _x = 2 * np.pi / (_it["u"] ** 2) * (_g ** 2)
        _y = np.log(_it["Wsat"])
        if np.isfinite(_x) and np.isfinite(_y):
            _x_all.append(_x)
            _y_all.append(_y)

    _x_min_data = min(_x_all) if _x_all else 0.0
    _x_max_data = max(_x_all) if _x_all else 1.0
    _y_min_data = min(_y_all) if _y_all else 0.0
    _y_max_data = max(_y_all) if _y_all else 1.0

    # Немного отступа
    _x_pad = (_x_max_data - _x_min_data) * 0.05 or 0.1
    _y_pad = (_y_max_data - _y_min_data) * 0.05 or 0.1

    x_range_data = (_x_min_data - _x_pad, _x_max_data + _x_pad)
    y_range_data = (_y_min_data - _y_pad, _y_max_data + _y_pad)
    return x_range_data, y_range_data


@app.cell
def _(x_range_data):
    x_range_data
    return


@app.cell(hide_code=True)
def _(mo, x_range_data, y_range_data):
    _x_lo, _x_hi = x_range_data
    _y_lo, _y_hi = y_range_data
    _dx = (_x_hi - _x_lo)/ 100
    _dy = (_y_hi - _y_lo)/ 100

    x_range = mo.ui.range_slider(
        start=_x_lo - _dx, stop=_x_hi + _dx,
        step=_dx , value=[_x_lo, _x_hi], label="X диапазон",
    )
    y_range = mo.ui.range_slider(
        start=_y_lo - _dy,
        stop=_y_hi + _dy,
        step=_dy, 
        value=[_y_lo, _y_hi],
        label="Y диапазон",
    )

    x_range, y_range
    return x_range, y_range


@app.cell(hide_code=True)
def _(gamma_interp, np, plt, wsat1_all, wsat2_all, x_range, y_range):
    from matplotlib.lines import Line2D

    _fig, _ax = plt.subplots(figsize=(9, 7))

    # Собираем все Λ и u из обеих серий
    _all_lam = sorted({_it["lambda_"] for _it in wsat1_all + wsat2_all})
    _all_u = sorted({_it["u"] for _it in wsat1_all + wsat2_all})

    _lambda_colors = {
        _lam: color
        for _lam, color in zip(
            _all_lam,
            plt.cm.tab10(np.linspace(0, 1, max(1, len(_all_lam)))),
        )
    }
    _u_markers = {
        _u_val: marker
        for _u_val, marker in zip(
            _all_u,
            ["o", "s", "^", "D", "P", "X", "v", ">", "<", "p", "*", "h"],
        )
    }

    # Series 1 — закрашенные маркеры
    for _it1 in wsat1_all:
        _g1 = gamma_interp(_it1["lambda_"]) / 2.0
        _x1 = 2 * np.pi / (_it1["u"] ** 2) * (_g1 ** 2)
        _y1 = np.log(_it1["Wsat"])
        if np.isfinite(_x1) and np.isfinite(_y1):
            _ax.plot(
                _x1, _y1,
                linestyle="None",
                marker=_u_markers.get(_it1["u"], "o"),
                color=_lambda_colors.get(_it1["lambda_"], "gray"),
                markersize=8, alpha=0.8,
                markeredgecolor="black", markeredgewidth=0.5,
            )

    # Series 2 — пустые маркеры
    for _it2 in wsat2_all:
        _g2 = gamma_interp(_it2["lambda_"]) / 2.0
        _x2 = 2 * np.pi / (_it2["u"] ** 2) * (_g2 ** 2)
        _y2 = np.log(_it2["Wsat"])
        if np.isfinite(_x2) and np.isfinite(_y2):
            _ax.plot(
                _x2, _y2,
                linestyle="None",
                marker=_u_markers.get(_it2["u"], "o"),
                color=_lambda_colors.get(_it2["lambda_"], "gray"),
                markersize=8, alpha=0.8,
                markerfacecolor="none", markeredgewidth=1.5,
            )

    # Fit для каждой серии отдельно: slope = 1 (fixed)
    def _fit_series(data, color, label, y_pos):
        _x = []
        _y = []
        for _it in data:
            _g = gamma_interp(_it["lambda_"]) / 2.0
            _xv = 2 * np.pi / (_it["u"] ** 2) * (_g ** 2)
            _yv = np.log(_it["Wsat"])
            if np.isfinite(_xv) and np.isfinite(_yv):
                _x.append(_xv)
                _y.append(_yv)
        _x = np.array(_x, dtype=float)
        _y = np.array(_y, dtype=float)
        if len(_x) > 2:
            _ss_tot = np.sum((_y - np.mean(_y)) ** 2)
            _n = len(_x)
            _intercept = np.mean(_y - _x)
            _ss_res = np.sum((_y - (_x + _intercept)) ** 2)
            _r2 = 1.0 - _ss_res / _ss_tot if _ss_tot > 0 else np.nan
            _adj_r2 = 1.0 - (1.0 - _r2) * (_n - 1) / (_n - 2) if _ss_tot > 0 and _n > 2 else np.nan
            _x_fit = np.linspace(np.min(_x), np.max(_x), 200)
            _ax.plot(_x_fit, _x_fit + _intercept, color=color, linewidth=2, linestyle="--", alpha=0.8)
            _ax.text(
                0.02, y_pos,
                (f"{label}\n"
                 f"intercept = {_intercept:.3f}\n"
                 rf"$R^2_{{adj}}$ = {_adj_r2:.3f}"),
                transform=_ax.transAxes, va="top", ha="left", fontsize=10,
                color=color,
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
            )
        return _x, _y

    _x1, _y1 = _fit_series(wsat1_all, "black", "Серия 1", 0.98)
    _x2, _y2 = _fit_series(wsat2_all, "gray", "Серия 2", 0.82)

    _ax.set_xlabel(r"$2\pi / u^2 \cdot \gamma_{1Dz}(\Lambda)^2$")
    _ax.set_ylabel(r"$\log(W_{sat})$")
    _ax.grid(alpha=0.3)

    # Пояснение: закрашенные = серия 1, пустые = серия 2
    _series_handles = [
        Line2D([0], [0], marker="o", color="gray", linestyle="None",
               markerfacecolor="gray", markersize=8, label="Серия 1 (закрашенные)"),
        Line2D([0], [0], marker="o", color="gray", linestyle="None",
               markerfacecolor="none", markeredgewidth=1.5, markersize=8, label="Серия 2 (пустые)"),
    ]

    # Двойная легенда (Λ)
    _lambda_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=color,
               markersize=8, label=rf"$\Lambda={lam}$")
        for lam, color in sorted(_lambda_colors.items())
    ]
    _col_lam = 2 if len(_lambda_handles) > 6 else 1
    _first_leg = _ax.legend(
        handles=_lambda_handles + _series_handles, loc="upper left",
        bbox_to_anchor=(1.02, 1), title=r"$\Lambda$",
        ncol=_col_lam, framealpha=0.9,
    )
    _ax.add_artist(_first_leg)

    # u маркеры
    _u_handles = [
        Line2D([0], [0], marker=marker, color="gray", linestyle="None",
               markersize=8, label=rf"$u={u}$")
        for u, marker in sorted(_u_markers.items())
    ]
    _col = 2 if len(_u_handles) > 6 else 1
    _ax.legend(
        handles=_u_handles, loc="lower left",
        bbox_to_anchor=(1.02, 0), title=r"$u$",
        ncol=_col, framealpha=0.9,
    )

    # Применяем диапазон из ползунков
    _ax.set_xlim(x_range.value[0], x_range.value[1])
    _ax.set_ylim(y_range.value[0], y_range.value[1])

    _fig
    return


if __name__ == "__main__":
    app.run()

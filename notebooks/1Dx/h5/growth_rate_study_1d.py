import marimo

__generated_with = "0.23.2"
app = marimo.App(layout_file="layouts/growth_rate_study_1d.grid.json")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import re


    from pathlib import Path
    from pde_analysis.load_h5_solver_file import load_h5_solver_file_1d
    from pde_analysis.analysis import compute_energy_1d
    from scipy.signal import find_peaks

    return (
        Path,
        compute_energy_1d,
        find_peaks,
        load_h5_solver_file_1d,
        mo,
        np,
        plt,
        re,
    )


@app.cell(hide_code=True)
def _(mo):
    folder_ui = mo.ui.text(
        value=r"C:\YandexDisk\Yandex.Disk\Ioffe\workspace\one_decay\1Dx\data\HDF5\auto\w0x=0p5",
        label="data folder",
        full_width=True,
    )

    folder_ui
    return (folder_ui,)


@app.cell(hide_code=True)
def _(Path, folder_ui):
    folder_path = Path(folder_ui.value)
    files = sorted([str(p) for p in folder_path.glob("*.h5")])
    return (files,)


@app.cell(hide_code=True)
def _(files, mo):
    files_checkbox = mo.ui.multiselect(
        options=files,
        value=files[:1] if files else [],
        label="data files",
    )

    files_checkbox
    return (files_checkbox,)


@app.cell
def _(mo):
    t_sat = mo.ui.slider(
        start=0.0,
        stop=50.0,
        step=0.1,
        value=20.0,
        label="t_sat (saturation time)"
    )
    t_sat
    return (t_sat,)


@app.cell(hide_code=True)
def _(
    compute_energy_1d,
    files_checkbox,
    find_peaks,
    load_h5_solver_file_1d,
    np,
    t_sat,
    tmax_selector,
    tmin_selector,
):
    results = []
    t_range_min = min(tmin_selector.value, tmax_selector.value)
    t_range_max = max(tmin_selector.value, tmax_selector.value)
    peak_number = 1  # Default: first maximum, can be changed manually later

    for filepath in files_checkbox.value:
        data = load_h5_solver_file_1d(filepath)
        Wa = compute_energy_1d(data, "a")

        t = np.asarray(Wa["t"])
        W = np.asarray(Wa["W"])
        positive = W > 0
        t = t[positive]
        W = W[positive]

        if len(t) < 3:
            continue

        logW = np.log(W)
        dlogW = np.gradient(logW, t)
        in_range = (t >= t_range_min) & (t <= t_range_max)
        dlogW_in_range = dlogW[in_range]
        t_in_range = t[in_range]

        if len(dlogW_in_range) > 0 and np.isfinite(dlogW_in_range).any():
            # Find peaks in the range
            peaks, _ = find_peaks(dlogW_in_range)

            if len(peaks) > 0:
                # Sort peaks by height (descending) and take the peak_number-th (1-indexed)
                peak_heights = dlogW_in_range[peaks]
                sorted_peak_indices = peaks[np.argsort(peak_heights)[::-1]]  # Sort descending

                if peak_number <= len(sorted_peak_indices):
                    growth_rate_idx = int(sorted_peak_indices[peak_number - 1])
                    growth_rate = float(dlogW_in_range[growth_rate_idx])
                    t_growth_rate_max = float(t_in_range[growth_rate_idx])
                else:
                    # If not enough peaks, take global max
                    growth_rate_idx = int(np.nanargmax(dlogW_in_range))
                    growth_rate = float(dlogW_in_range[growth_rate_idx])
                    t_growth_rate_max = float(t_in_range[growth_rate_idx])
            else:
                # If no peaks found, take global max
                growth_rate_idx = int(np.nanargmax(dlogW_in_range))
                growth_rate = float(dlogW_in_range[growth_rate_idx])
                t_growth_rate_max = float(t_in_range[growth_rate_idx])
        else:
            growth_rate_idx = int(np.nanargmax(dlogW))
            growth_rate = float(dlogW[growth_rate_idx])
            t_growth_rate_max = float(t[growth_rate_idx])

        predicted_growth = float(np.exp(growth_rate * (t[-1] - t[0])))
        actual_growth = float(W[-1] / W[0])

        # Find Wa at t_sat
        t_sat_value = t_sat.value
        idx_sat = np.argmin(np.abs(t - t_sat_value))
        Wa_sat = float(W[idx_sat])

        params = data.get("parameters", {})
        u_value = params.get("u")

        results.append(
            {
                "file": filepath,
                "label": filepath.split("\\")[-1],
                "u": u_value,
                "t": t,
                "Wa": W,
                "logW": logW,
                "dlogW": dlogW,
                "growth_rate": growth_rate,
                "t_growth_rate_max": t_growth_rate_max,
                "predicted_growth": predicted_growth,
                "actual_growth": actual_growth,
                "Wa_sat": Wa_sat,
            }
        )
    return (results,)


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo, results):
    labels = [item["label"] for item in results]
    visible_ui = mo.ui.multiselect(
        options=labels,
        value=labels,
        label="visible files",
    )

    visible_ui
    return (visible_ui,)


@app.cell(hide_code=True)
def _(mo, results):
    selected_label_ui = mo.ui.radio(
        options=[item["label"] for item in results],
        value=results[0]["label"] if results else None,
        label="detail file",
    )

    selected_label_ui
    return (selected_label_ui,)


@app.cell(hide_code=True)
def _(mo):
    tmin_selector = mo.ui.number(value=0.0, step=0.01, label="t_min for d/dt plot")
    tmax_selector = mo.ui.number(value=10.0, step=0.01, label="t_max for d/dt plot")

    ymin_selector = mo.ui.number(value=-1.0, label="y_min for d/dt plot")
    ymax_selector = mo.ui.number(value=2.0, label="y_max for d/dt plot")

    scale_selector = mo.ui.radio(
        options=["linear", "log"],
        value="linear",
        label="scale for Wa plot",
    )

    legend_position_selector = mo.ui.radio(
        options=["inside", "right", "below", "outside"],
        value="inside",
        label="legend position",
    )

    c1_selector = mo.ui.number(value=1.6, step=0.01, label="C1 for level")
    c2_selector = mo.ui.number(value=1.6, step=0.01, label="C2 for level")

    tmin_selector, tmax_selector, ymin_selector, ymax_selector, scale_selector, legend_position_selector, c1_selector, c2_selector
    return (
        c1_selector,
        legend_position_selector,
        scale_selector,
        tmax_selector,
        tmin_selector,
        ymax_selector,
        ymin_selector,
    )


@app.cell(hide_code=True)
def _(
    legend_position_selector,
    plt,
    results,
    selected_label_ui,
    tmax_selector,
    tmin_selector,
    visible_ui,
    ymax_selector,
    ymin_selector,
):
    import re as _re

    fig_logs, ax_logs = plt.subplots(figsize=(8, 5))

    for _item in results:
        if _item["label"] not in visible_ui.value:
            continue

        # Parse filename to extract u and w0x (convert p to .)
        _filename = _item["label"]
        _u_match = _re.search(r'u=([\d.p]+)', _filename)
        _w_match = _re.search(r'w0x=([\d.p]+)', _filename)

        _u_val_parsed = _u_match.group(1).replace('p', '.') if _u_match else "?"
        _w_val_parsed = _w_match.group(1).replace('p', '.') if _w_match else "?"

        _new_label = f"u={_u_val_parsed}, w₀={_w_val_parsed}"

        mask = (_item["t"] >= tmin_selector.value) & (_item["t"] <= tmax_selector.value)
        ax_logs.plot(_item["t"][mask], _item["dlogW"][mask], label=_new_label)

    selected = next((item for item in results if item["label"] == selected_label_ui.value), None)
    if selected is not None:
        # Parse selected label too
        _filename_sel = selected["label"]
        _u_match_sel = _re.search(r'u=([\d.p]+)', _filename_sel)
        _w_match_sel = _re.search(r'w0x=([\d.p]+)', _filename_sel)

        _u_val_parsed_sel = _u_match_sel.group(1).replace('p', '.') if _u_match_sel else "?"
        _w_val_parsed_sel = _w_match_sel.group(1).replace('p', '.') if _w_match_sel else "?"
        _selected_new_label = f"u={_u_val_parsed_sel}, w₀={_w_val_parsed_sel}"

        ax_logs.scatter(
            [selected["t_growth_rate_max"]],
            [selected["growth_rate"]],
            color="red",
            marker="x",
            s=80,
            label=f"growth rate {_selected_new_label}",
        )

    ax_logs.set_ylabel("d/dt log Wa")
    ax_logs.set_xlabel(r"$\nu_0 t$")
    ax_logs.grid(True)
    ax_logs.set_ylim(ymin_selector.value, ymax_selector.value)

    if legend_position_selector.value == "inside":
        ax_logs.legend()
    elif legend_position_selector.value == "right":
        ax_logs.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    elif legend_position_selector.value == "below":
        ax_logs.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    else:
        ax_logs.legend(loc="upper left", bbox_to_anchor=(1.05, 1))

    if legend_position_selector.value == "below":
        fig_logs.tight_layout(rect=(0, 0.12, 1, 1))
    else:
        fig_logs.tight_layout()

    fig_logs
    return


@app.cell(hide_code=True)
def _(plt, results, visible_ui):
    fig_vs_u, ax_vs_u = plt.subplots(figsize=(8, 5))

    scatter_data = [_item for _item in results if _item["label"] in visible_ui.value and _item["u"] is not None]
    scatter_data.sort(key=lambda x: x["u"])

    if scatter_data:
        u_values = [_item["u"] for _item in scatter_data]
        growth_rates = [_item["growth_rate"] for _item in scatter_data]
        ax_vs_u.plot(u_values, growth_rates, marker="o", linestyle="-", label=r"$2\gamma/\nu_0$")

    ax_vs_u.set_xlabel("u")
    ax_vs_u.set_ylabel(r"$2\gamma/\nu_0$")
    ax_vs_u.grid(True)
    ax_vs_u.legend()
    fig_vs_u.tight_layout()

    print([[u, g] for u, g in zip(u_values, growth_rates)])
    fig_vs_u
    return


@app.cell(hide_code=True)
def _(
    c1_selector,
    legend_position_selector,
    np,
    plt,
    re,
    results,
    scale_selector,
    visible_ui,
):
    fig_with_level, ax_with_level = plt.subplots(figsize=(8, 5))

    curve_colors = {}
    for _item in results:
        if _item["label"] not in visible_ui.value:
            continue

        # Parse filename to extract u and w0x (convert p to .)
        filename = _item["label"]
        u_match = re.search(r'u=([\d.p]+)', filename)
        w_match = re.search(r'w0x=([\d.p]+)', filename)

        u_val_parsed = u_match.group(1).replace('p', '.') if u_match else "?"
        w_val_parsed = w_match.group(1).replace('p', '.') if w_match else "?"

        new_label = f"u={u_val_parsed}, w₀={w_val_parsed}"

        curve_line, = ax_with_level.plot(_item["t"], _item["Wa"], label=new_label)

        # Draw level if u is available
        if _item["u"] is not None:
            u_val = _item["u"]
            curve_color = curve_colors.get(u_val)
            if curve_color is None:
                curve_color = curve_line.get_color()
                curve_colors[u_val] = curve_color
            else:
                curve_line.set_color(curve_color)

            level_u_val = np.exp(2 * np.pi / (u_val ** 2) + c1_selector.value)
            ax_with_level.axhline(
                level_u_val,
                color=curve_color,
                linestyle="--",
                alpha=0.6,
            )

    ax_with_level.set_xlabel(r"$\nu_0 t$")
    ax_with_level.set_ylabel("Wa")

    # Add title template
    title_text = "dx=0.005"
    ax_with_level.set_title(title_text, fontsize=10)

    if scale_selector.value == "log":
        ax_with_level.set_yscale("log")

    ax_with_level.grid(True)

    if legend_position_selector.value == "inside":
        ax_with_level.legend()
    elif legend_position_selector.value == "right":
        ax_with_level.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    elif legend_position_selector.value == "below":
        ax_with_level.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    else:
        ax_with_level.legend(loc="upper left", bbox_to_anchor=(1.05, 1))

    if legend_position_selector.value == "below":
        fig_with_level.tight_layout(rect=(0, 0.12, 1, 1))
    else:
        fig_with_level.tight_layout()
    fig_with_level 
    return


@app.cell
def _(np, results):
    # Create table of u and Wa_sat for all files
    data_table = []
    for _item in results:
        _u_val = _item["u"] if _item["u"] is not None else "N/A"
        data_table.append([_u_val, np.log(_item["Wa_sat"])])
    data_table
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Логарифм уровня насыщения
    """)
    return


@app.cell
def _():
    gauss_sat_w0_0p5 = [
      [
        0.6,
        18.140116403004352
      ],
      [
        0.8,
        10.602046782172613
      ],
      [
        1.2,
        4.6280666529594345
      ],
      [
        1.4,
        3.6656025879181002
      ],
      [
        1.6,
        3.0356334593598286
      ],
      [
        1.8,
        2.575687682681905
      ],
      [
        1.0,
        6.511413031397413
      ],
      [
        2.2,
        1.9433207964495842
      ],
      [
        2.4,
        1.7169975531773678
      ],
      [
        2.6,
        1.5303836361740362
      ],
      [
        2.0,
        2.222702294977362
      ]
    ]

    gauss_sat_w0_0p1 = [
      [
        0.6,
        15.1925911507745
      ],
      [
        0.8,
        7.9686218534980675
      ],
      [
        1.2,
        2.621942824571425
      ],
      [
        1.4,
        1.5942229044393141
      ],
      [
        1.6,
        0.9254216996924053
      ],
      [
        1.8,
        0.45040515902158457
      ],
      [
        1.0,
        4.419518157033542
      ],
      [
        2.2,
        -0.1741956748669994
      ],
      [
        2.4,
        -0.38881073239125
      ],
      [
        2.6,
        -0.561234059550923
      ],
      [
        2.0,
        0.09330118664918909
      ]
    ]

    gauss_sat_w0_0p05 = [
      [
        0.6,
        13.81584292759961
      ],
      [
        0.8,
        6.619291511151065
      ],
      [
        1.2,
        1.4054774214939003
      ],
      [
        1.4,
        0.443741479836081
      ],
      [
        1.6,
        -0.16000296916538126
      ],
      [
        1.8,
        -0.5716466367938834
      ],
      [
        1.0,
        3.1355232144591603
      ],
      [
        2.2,
        -1.089425454443182
      ],
      [
        2.4,
        -1.2767438687251125
      ],
      [
        2.6,
        -1.4079758505392888
      ],
      [
        2.0,
        -0.8737186851544905
      ]
    ]

    gauss_sat_w0_1 = [
      [
        0.6,
        19.021970026510296
      ],
      [
        0.8,
        11.437571992978805
      ],
      [
        1.2,
        5.206963288745076
      ],
      [
        1.4,
        4.464098965003791
      ],
      [
        1.6,
        3.920967265995989
      ],
      [
        1.8,
        3.495672902943778
      ],
      [
        1.0,
        6.87218512731246
      ],
      [
        2.2,
        2.881291736272489
      ],
      [
        2.0,
        3.1563727553728125
      ]
    ]

    gauss_sat_x0_0 = [
      [
        0.6,
        13.841496953848015
      ],
      [
        0.8,
        6.544843736213469
      ],
      [
        1.2,
        1.4817327446498139
      ],
      [
        1.4,
        0.4444830316988758
      ],
      [
        1.6,
        -0.20016148240295253
      ],
      [
        1.8,
        -0.6363965414494771
      ],
      [
        1.0,
        3.2399573115009175
      ],
      [
        2.2,
        -1.1856901843394638
      ],
      [
        2.0,
        -0.953910963357522
      ]
    ]


    gauss_sat_x0_10 = [
      [
        0.6,
        13.796843130813746
      ],
      [
        0.8,
        6.672939421564022
      ],
      [
        1.2,
        1.4233325027371324
      ],
      [
        1.4,
        0.42314343291730167
      ],
      [
        1.6,
        -0.16108706300257095
      ],
      [
        1.8,
        -0.555329791209162
      ],
      [
        1.0,
        3.300954448999917
      ],
      [
        2.2,
        -1.0644132746125243
      ],
      [
        2.0,
        -0.846011139344839
      ]
    ]

    delta_sat_w_0p5 = [
      [
        0.5,
        24.439200962732478
      ],
      [
        0.6,
        16.767069747298795
      ],
      [
        0.8,
        9.335289176002735
      ],
      [
        1.0,
        5.922748233272939
      ],
      [
        1.2,
        4.060715582862744
      ]
    ]

    delta_sat_w_0p1 = [
      [
        0.5,
        24.439200962732478
      ],
      [
        0.6,
        16.767069747298795
      ],
      [
        0.8,
        9.335289176002735
      ],
      [
        1.0,
        5.922748233272939
      ],
      [
        1.2,
        4.060715582862744
      ]
    ]

    delta_sat_w_0p05 = [
      [
        0.5,
        25.912751007700436
      ],
      [
        0.6,
        18.453748691290574
      ],
      [
        0.8,
        11.129755074368255
      ],
      [
        1.0,
        7.80696895867281
      ],
      [
        1.2,
        6.02957586863489
      ]
    ]

    delta_sat_w_0p01 = [
      [
        0.5,
        25.928037812524835
      ],
      [
        0.6,
        18.477341378061933
      ],
      [
        0.8,
        11.175648940844695
      ],
      [
        1.0,
        7.880955623720084
      ],
      [
        1.2,
        6.136645830661121
      ]
    ]
    return (
        delta_sat_w_0p01,
        delta_sat_w_0p05,
        delta_sat_w_0p1,
        delta_sat_w_0p5,
        gauss_sat_w0_0p05,
        gauss_sat_w0_0p1,
        gauss_sat_w0_0p5,
        gauss_sat_w0_1,
        gauss_sat_x0_0,
        gauss_sat_x0_10,
    )


@app.cell
def _(np):
    from sklearn.linear_model import LinearRegression

    def calculate_r2(x_values, y_values):
        """
        Calculate R^2 for linear regression.
        """
        x = np.array(x_values).reshape(-1, 1)
        y = np.array(y_values)
        model = LinearRegression().fit(x, y)
        r2 = model.score(x, y)
        return r2

    def sort_by_first_element(data_list):
        return sorted(data_list, key=lambda x: x[0])



    return calculate_r2, sort_by_first_element


@app.cell
def _(
    calculate_r2,
    delta_sat_w_0p01,
    delta_sat_w_0p05,
    delta_sat_w_0p1,
    delta_sat_w_0p5,
    np,
    plt,
    sort_by_first_element,
):
    _fig1, _ax1 = plt.subplots(figsize=(8, 5))

    # 1. Delta function with different w
    _delta_datasets = {
        "w₀=0.01": sort_by_first_element(delta_sat_w_0p01),
        "w₀=0.05": sort_by_first_element(delta_sat_w_0p05),
        "w₀=0.1": sort_by_first_element(delta_sat_w_0p1),
        "w₀=0.5": sort_by_first_element(delta_sat_w_0p5),
    }

    for _label, _data in _delta_datasets.items():
        _u_values = [_point[0] for _point in _data]
        _x_values = [2 * np.pi / (u ** 2) for u in _u_values]
        _logWa_values = [_point[1] for _point in _data]

        # Calculate R^2
        _r2 = calculate_r2(_x_values, _logWa_values)
        _label_with_r2 = f"{_label} (R²={_r2:.5f})"

        _ax1.plot(_x_values, _logWa_values, marker='o', label=_label_with_r2)
    _ax1.set_box_aspect(1)   
    _ax1.set_xticks(np.arange(0, max(_ax1.get_xlim()) + 2.5, 2.5))
    _ax1.set_yticks(np.arange(0, max(_ax1.get_ylim()) + 2.5, 2.5))

    _ax1.set_xlabel(r"$2\pi/u^2$")
    _ax1.set_ylabel(r"$\log(W_a)$")
    _ax1.set_title("Delta function evolution: saturation vs w₀")
    _ax1.grid(True)
    _ax1.legend()
    _fig1.tight_layout()
    _fig1
    return


@app.cell
def _(
    calculate_r2,
    gauss_sat_w0_0p05,
    gauss_sat_w0_0p1,
    gauss_sat_w0_0p5,
    gauss_sat_w0_1,
    np,
    plt,
    sort_by_first_element,
):
    _fig2, _ax2 = plt.subplots(figsize=(8, 5))

    # 2. Gauss evolution with different w, x0=5
    _gauss_datasets = {
        "w₀=0.05": sort_by_first_element(gauss_sat_w0_0p05),
        "w₀=0.1": sort_by_first_element(gauss_sat_w0_0p1),
        "w₀=0.5": sort_by_first_element(gauss_sat_w0_0p5),
        "w₀=1.0": sort_by_first_element(gauss_sat_w0_1),
    }

    for _label, _data in _gauss_datasets.items():
        _u_values = [_point[0] for _point in _data]
        _x_values = [2 * np.pi / (u ** 2) for u in _u_values]
        _logWa_values = [_point[1] for _point in _data]

        # Calculate R^2
        _r2 = calculate_r2(_x_values, _logWa_values)
        _label_with_r2 = f"{_label} (R²={_r2:.5f})"

        _ax2.plot(_x_values, _logWa_values, marker='o', label=_label_with_r2)
    _ax2.set_xticks(np.arange(0, max(_ax2.get_xlim()) + 2.5, 2.5))
    _ax2.set_yticks(np.arange(0, max(_ax2.get_ylim()) + 2.5, 2.5))

    _ax2.set_xlabel(r"$2\pi/u^2$")
    _ax2.set_ylabel(r"$\log(W_a)$")
    _ax2.set_title("Gauss evolution (x₀=5): saturation vs w₀")
    _ax2.grid(True)
    _ax2.legend()
    _ax2.set_aspect('equal')
    _fig2.tight_layout()
    _fig2
    return


@app.cell
def _(
    calculate_r2,
    gauss_sat_w0_0p05,
    gauss_sat_x0_0,
    gauss_sat_x0_10,
    np,
    plt,
    sort_by_first_element,
):
    _fig3, _ax3 = plt.subplots(figsize=(8, 5))

    # 3. Compare w0=0.05, x0 = 0, 5, 10
    _x0_datasets = {
        "x₀=0": sort_by_first_element(gauss_sat_x0_0),
        "x₀=5": sort_by_first_element(gauss_sat_w0_0p05),
        "x₀=10": sort_by_first_element(gauss_sat_x0_10),
    }

    for _label, _data in _x0_datasets.items():
        _u_values = [_point[0] for _point in _data]
        _x_values = [2 * np.pi / (u ** 2) for u in _u_values]
        _logWa_values = [_point[1] for _point in _data]

        # Calculate R^2
        _r2 = calculate_r2(_x_values, _logWa_values)
        _label_with_r2 = f"{_label} (R²={_r2:.5f})"

        _ax3.plot(_x_values, _logWa_values, marker='o', label=_label_with_r2)
    _ax3.set_box_aspect(1)   
    _ax3.set_xticks(np.arange(0, max(_ax3.get_xlim()) + 2.5, 2.5))
    _ax3.set_yticks(np.arange(0, max(_ax3.get_ylim()) + 2.5, 2.5))

    _ax3.set_xlabel(r"$2\pi/u^2$")
    _ax3.set_ylabel(r"$\log(W_a)$")
    _ax3.set_title("Gauss evolution (w₀=0.05): saturation vs x₀")
    _ax3.grid(True)
    _ax3.legend()
    _fig3.tight_layout()
    _fig3
    return


@app.cell
def _(np, plt):
    picked_data_w_0p05 = [
        [0, 2],
        [np.float64(0.6), 1.7448231306716089],
        [np.float64(0.8), 1.5550525678418161],
        [np.float64(1.0), 1.283402333661524],
        [np.float64(1.2), 0.9691163819206601],
        [np.float64(1.4), 0.6859949325346566],
        [np.float64(1.6), 0.478948184707793],
        [np.float64(1.8), 0.3431621053689149],
        [np.float64(2.0), 0.25648421159802837],
        [np.float64(2.2), 0.20180857309470923],
        [np.float64(2.4), 0.1614733634596064],
        [np.float64(2.6), 0.13635387040818614]
    ]

    # Данные для w = 0.5
    picked_data_w_0p5 = [
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
        [np.float64(2.6), 0.6471517894360832]
    ]

    # Третий набор: w = 0.05, x0 = 10 
    picked_data_w_0p05_x0_10 = [
        [np.float64(0.6), 1.7464403066455922],
        [np.float64(0.8), 1.5555665410760544],
        [np.float64(1.0), 1.281173878707527],
        [np.float64(1.2), 0.9629130200310074],
        [np.float64(1.4), 0.6784667436218115],
        [np.float64(1.6), 0.4703941371814331],
        [np.float64(1.8), 0.3367577702557565],
        [np.float64(2.0), 0.2504684697508459],
        [np.float64(2.2), 0.194926990679134]      
    ]

    # Извлечение x и y для w=0.05
    x1, gamma2_1 = zip(*picked_data_w_0p05)
    gamma1_1 = np.array(gamma2_1) / 2

    # Извлечение x и y для w=0.5
    x2, gamma2_2 = zip(*picked_data_w_0p5)
    gamma1_2 = np.array(gamma2_2) / 2

    # Извлечение x и y для третьего набора
    x3, gamma2_3 = zip(*picked_data_w_0p05_x0_10)
    gamma1_3 = np.array(gamma2_3) / 2

    # Построение графиков
    plt.figure(figsize=(10, 6))
    plt.plot(x1, gamma1_1, 'o-', linewidth=2, markersize=6, label='w = 0.05 (x0 = 5)')
    plt.plot(x2, gamma1_2, 's-', linewidth=2, markersize=6, label='w = 0.5 (x0 = 5)')
    plt.plot(x3, gamma1_3, '^-', linewidth=2, markersize=6, label='w = 0.05 (x0 = 10)')

    # Оформление
    plt.xlabel('u_values')
    plt.ylabel('growth_rates')
    plt.title('График зависимости growth_rates от u_values')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Данные для эволюции дельта функции
    """)
    return


@app.cell
def _(np, plt):
    delta_w0_0p01 = [[np.float64(0.4), 1.861616235609759], [np.float64(0.5), 1.8130577533297156], [np.float64(0.6), 1.7608288187366057], [np.float64(0.8), 1.655252916818629], [np.float64(1.0), 1.571040012628858]]

    delta_w0_0p05 = [[np.float64(0.4), 1.8615656670863174], [np.float64(0.5), 1.8130488411299623], [np.float64(0.6), 1.7609930733318464], [np.float64(0.8), 1.6566415230067584], [np.float64(1.0), 1.5759652401597801]]

    delta_w0_0p1 = [[np.float64(0.4), 1.8613971637251652], [np.float64(0.5), 1.8128515802620626], [np.float64(0.6), 1.7610154534064684], [np.float64(0.8), 1.6582838540358225], [np.float64(1.0), 1.5864466223768696]]

    delta_w0_0p5 = [[np.float64(0.4), 1.8624345700420264], [np.float64(0.5), 1.8122961860411984], [np.float64(0.6), 1.762724022469257]]


    _fig, _ax = plt.subplots(figsize=(8, 5))

    # Define datasets with their labels
    _datasets = {
        "w₀=0.01": delta_w0_0p01,
        "w₀=0.05": delta_w0_0p05,
        "w₀=0.1": delta_w0_0p1,
        "w₀=0.5": delta_w0_0p5,
    }

    for _label, _data in _datasets.items():
        _u_values = [_point[0] for _point in _data]
        _gamma_nu0_values = [_point[1] for _point in _data]
        _ax.plot(_u_values, _gamma_nu0_values, marker='o', label=_label)

    _ax.set_title("Delta function evolution")
    _ax.set_xlabel("u")
    _ax.set_ylabel(r"$2\gamma/\nu_0$")
    _ax.grid(True)
    _ax.legend()
    _fig.tight_layout()
    _fig
    return (delta_w0_0p05,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Данные для эволюции двух гауссов
    """)
    return


@app.cell
def _(np, plt):
    # x0 = 5
    gauss_w0_0p05 = [[np.float64(0.4), 1.8625464287709406], [np.float64(0.6), 1.7448231306716089], [np.float64(0.8), 1.5550525678418161], [np.float64(1.0), 1.283402333661524], [np.float64(1.2), 0.9691163819206601], [np.float64(1.4), 0.6859949325346566], [np.float64(1.6), 0.478948184707793], [np.float64(1.8), 0.3431621053689149], [np.float64(2.0), 0.25648421159802837], [np.float64(2.2), 0.20180857309470923], [np.float64(2.4), 0.1614733634596064], [np.float64(2.6), 0.13635387040818614]]
    gauss_w0_0p5_first = [[np.float64(0.4), 1.3788411240662146], [np.float64(0.6), 1.0758704992326167], [np.float64(0.8), 0.8641129089552038], [np.float64(1.0), 0.7149524744417085], [np.float64(1.2), 0.6060744370519366], [np.float64(1.4), 0.5259662940482241], [np.float64(1.6), 0.4614495010255077], [np.float64(1.8), 0.41202259560924637], [np.float64(2.0), 0.3662547267155065], [np.float64(2.2), 0.33636612674037536], [np.float64(2.4), 0.30840876466282807], [np.float64(2.6), 0.282261191282408]]
    gauss_w0_0p5_second = [[np.float64(0.4), 1.927632926231027], [np.float64(0.6), 1.8618887456236592], [np.float64(0.8), 1.7634897985058942], [np.float64(1.0), 1.6292176902361817], [np.float64(1.2), 1.470791177616892], [np.float64(1.4), 1.3056809640564424], [np.float64(1.6), 1.1497878424167567], [np.float64(1.8), 1.0114144684646131], [np.float64(2.0), 0.8936880369139644], [np.float64(2.2), 0.7956294618296091], [np.float64(2.4), 0.7142589925935026], [np.float64(2.6), 0.6471517894360832]]

    gauss_w0_0p1 = [[np.float64(0.4), 1.8661548728715616], [np.float64(0.6), 1.756810363739973], [np.float64(0.8), 1.5953916784603877], [np.float64(1.0), 1.376519055851647], [np.float64(1.2), 1.1241669881307126], [np.float64(1.4), 0.8809084063063413], [np.float64(1.6), 0.67878250646854], [np.float64(1.8), 0.5273862294097604], [np.float64(2.0), 0.4181278096990635], [np.float64(2.2), 0.34058950893712314], [np.float64(2.4), 0.28307119478334997], [np.float64(2.6), 0.24123563461985675]]

    gauss_w0_1_second = [[np.float64(0.6), 1.927326647846229], [np.float64(0.8), 1.8642606355774909], [np.float64(1.0), 1.7571150047014186], [np.float64(1.2), 1.6260268153618416], [np.float64(1.4), 1.517597081071317], [np.float64(1.6), 1.4772442772821215], [np.float64(1.8), 1.492916223346679], [np.float64(2.0), 1.531478861782336], [np.float64(2.2), 1.5779128982329471]]

    gauss_w0_1_first =[[np.float64(0.6), 1.1095912435957942], [np.float64(0.8), 0.8921705835354659], [np.float64(1.0), 0.7274267659527838], [np.float64(1.2), 0.6166811119638158], [np.float64(1.4), 0.5111209177954943], [np.float64(1.6), 0.45652111444032517], [np.float64(1.8), 0.4026069242726926], [np.float64(2.0), 0.3509039806417449], [np.float64(2.2), 0.3026085809790109]]

    # w0 = 0.05, x0 =...
    gauss_x0_0 = [[np.float64(0.6), 1.746312902997989], [np.float64(0.8), 1.5646976650325284], [np.float64(1.0), 1.307335458571494], [np.float64(1.2), 1.008972223015002], [np.float64(1.4), 0.7342005525422186], [np.float64(1.6), 0.5275757735248003], [np.float64(1.8), 0.387997662232376], [np.float64(2.0), 0.29467201143500965], [np.float64(2.2), 0.2328078098879942]]

    gauss_x0_10 = [[np.float64(0.6), 1.7464403066455922], [np.float64(0.8), 1.5555665410760544], [np.float64(1.0), 1.281173878707527], [np.float64(1.2), 0.9629130200310074], [np.float64(1.4), 0.6784667436218115], [np.float64(1.6), 0.4703941371814331], [np.float64(1.8), 0.3367577702557565], [np.float64(2.0), 0.2504684697508459], [np.float64(2.2), 0.194926990679134]]

    _fig, _ax = plt.subplots(figsize=(8, 5))

    # Define datasets with their labels
    _datasets = {
        "w₀=0.05, x₀=5": gauss_w0_0p05,
        "w₀=0.1, x₀=5": gauss_w0_0p1,
        "w₀=0.5, x₀=5 (1st peak)": gauss_w0_0p5_first,
        "w₀=0.5, x₀=5 (2nd peak)": gauss_w0_0p5_second,
        "w₀=1.0, x₀=5 (1st peak)": gauss_w0_1_first,
        "w₀=1.0, x₀=5 (2nd peak)": gauss_w0_1_second,
        "w₀=0.05, x₀=0": gauss_x0_0,
        "w₀=0.05, x₀=10": gauss_x0_10,
    }

    for _label, _data in _datasets.items():
        _u_values = [_point[0] for _point in _data]
        _gamma_nu0_values = [_point[1] for _point in _data]
        _ax.plot(_u_values, _gamma_nu0_values, marker='o', label=_label)

    _ax.set_xlabel("u")
    _ax.set_ylabel(r"$2\gamma/\nu_0$")
    _ax.grid(True)
    _ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    _fig.tight_layout()
    _fig
    return (gauss_w0_0p05,)


@app.cell
def _(delta_w0_0p05, gauss_w0_0p05, plt):
    _fig, _ax = plt.subplots(figsize=(8, 5))

    # Define datasets with their labels
    _datasets = {
        "delta function evolution": delta_w0_0p05,
        "gauss function evolution": gauss_w0_0p05,
    }

    for _label, _data in _datasets.items():
        _u_values = [_point[0] for _point in _data]
        _gamma_nu0_values = [_point[1] for _point in _data]
        _ax.plot(_u_values, _gamma_nu0_values, marker='o', label=_label)

    _ax.set_title(r"$w_0 = 0.05$")
    _ax.set_xlabel("u")
    _ax.set_ylabel(r"$2\gamma/\nu_0$")
    _ax.grid(True)
    _ax.legend()
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(gauss_w0_0p05, np, plt):
    _fig, _ax = plt.subplots(figsize=(8, 5))

    # Define function
    def sech_squared_fit(u, k):
        return 2 / (np.cosh(k * u))**2

    # Data
    _u_values = [point[0] for point in gauss_w0_0p05]
    _gamma_values = [point[1] for point in gauss_w0_0p05]

    # Plot data
    _ax.plot(_u_values, _gamma_values, marker='o', label="Data: w₀=0.05, x₀=5")

    # Fit parameter k (adjust manually or fit automatically)
    k = 0.8  # You can change this value

    # Generate smooth curve for the fit
    _u_fit = np.linspace(min(_u_values), max(_u_values), 100)
    _gamma_fit = sech_squared_fit(_u_fit, k)

    _ax.plot(_u_fit, _gamma_fit, linestyle='--', label=f'2/(cosh(k·u))², k={k:.2f}')

    _ax.set_xlabel("u")
    _ax.set_ylabel(r"$2\gamma/\nu_0$")
    _ax.set_title(r"$w_0 = 0.05, x_0 = 5$")
    _ax.grid(True)
    _ax.legend()
    _fig.tight_layout()
    _fig
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

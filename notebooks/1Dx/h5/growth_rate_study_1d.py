import marimo

__generated_with = "0.23.2"
app = marimo.App(layout_file="layouts/growth_rate_study_1d.grid.json")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from pathlib import Path
    from pde_analysis.load_h5_solver_file import load_h5_solver_file_1d
    from pde_analysis.analysis import compute_energy_1d

    return Path, compute_energy_1d, load_h5_solver_file_1d, mo, np, plt


@app.cell
def _(mo):
    folder_ui = mo.ui.text(
        value=r"C:\YandexDisk\Yandex.Disk\Ioffe\workspace\one_decay\1Dx\data\HDF5\auto\w0x=0p5",
        label="data folder",
        full_width=True,
    )

    folder_ui
    return (folder_ui,)


@app.cell
def _(Path, folder_ui):
    folder_path = Path(folder_ui.value)
    files = sorted([str(p) for p in folder_path.glob("*.h5")])
    return (files,)


@app.cell
def _(files, mo):
    files_checkbox = mo.ui.multiselect(
        options=files,
        value=files[:1] if files else [],
        label="data files",
    )

    files_checkbox
    return (files_checkbox,)


@app.cell
def _():
    return


@app.cell
def _(
    compute_energy_1d,
    files_checkbox,
    load_h5_solver_file_1d,
    np,
    tmax_selector,
    tmin_selector,
):
    results = []
    t_range_min = min(tmin_selector.value, tmax_selector.value)
    t_range_max = max(tmin_selector.value, tmax_selector.value)

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
            growth_rate_idx = int(np.nanargmax(dlogW_in_range))
            growth_rate = float(dlogW_in_range[growth_rate_idx])
            t_growth_rate_max = float(t_in_range[growth_rate_idx])
        else:
            growth_rate_idx = int(np.nanargmax(dlogW))
            growth_rate = float(dlogW[growth_rate_idx])
            t_growth_rate_max = float(t[growth_rate_idx])

        predicted_growth = float(np.exp(growth_rate * (t[-1] - t[0])))
        actual_growth = float(W[-1] / W[0])

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
            }
        )
    return (results,)


@app.cell
def _(mo, results):
    labels = [item["label"] for item in results]
    visible_ui = mo.ui.multiselect(
        options=labels,
        value=labels,
        label="visible files",
    )

    visible_ui
    return (visible_ui,)


@app.cell
def _(mo, results):
    selected_label_ui = mo.ui.radio(
        options=[item["label"] for item in results],
        value=results[0]["label"] if results else None,
        label="detail file",
    )

    selected_label_ui
    return (selected_label_ui,)


@app.cell
def _(mo):
    tmin_selector = mo.ui.number(value=0.0, step=0.1, label="t_min for d/dt plot")
    tmax_selector = mo.ui.number(value=10.0, step=0.1, label="t_max for d/dt plot")

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
        c2_selector,
        legend_position_selector,
        scale_selector,
        tmax_selector,
        tmin_selector,
        ymax_selector,
        ymin_selector,
    )


@app.cell
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
    fig_logs, ax_logs = plt.subplots(figsize=(8, 5))

    for _item in results:
        if _item["label"] not in visible_ui.value:
            continue
        mask = (_item["t"] >= tmin_selector.value) & (_item["t"] <= tmax_selector.value)
        ax_logs.plot(_item["t"][mask], _item["dlogW"][mask], label=_item["label"])

    selected = next((item for item in results if item["label"] == selected_label_ui.value), None)
    if selected is not None:
        ax_logs.scatter(
            [selected["t_growth_rate_max"]],
            [selected["growth_rate"]],
            color="red",
            marker="x",
            s=80,
            label=f"growth rate {selected['label']}",
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


@app.cell
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

    fig_vs_u
    return


@app.cell
def _(
    c1_selector,
    c2_selector,
    legend_position_selector,
    np,
    plt,
    results,
    scale_selector,
    visible_ui,
):
    fig_with_level, ax_with_level = plt.subplots(figsize=(8, 5))

    curve_colors = {}
    for _item in results:
        if _item["label"] not in visible_ui.value:
            continue
        curve_line, = ax_with_level.plot(_item["t"], _item["Wa"], label=_item["label"])

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
                label=f"level u={u_val:.3f}, C1",
            )

            selected_growth_rate_value = _item["growth_rate"]
            gamma = selected_growth_rate_value / 2.0
            level_gamma_val = np.exp(2 * np.pi * (gamma ** 2) + c2_selector.value)
            ax_with_level.axhline(
                level_gamma_val,
                color=curve_color,
                linestyle="-.",
                alpha=0.6,
                label=f"level gamma u={u_val:.3f}, C2",
            )

    ax_with_level.set_xlabel(r"$\nu_0 t$")
    ax_with_level.set_ylabel("Wa")

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
def _():
    return


if __name__ == "__main__":
    app.run()

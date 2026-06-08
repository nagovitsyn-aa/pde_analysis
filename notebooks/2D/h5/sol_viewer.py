import marimo

__generated_with = "0.22.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt

    from pde_analysis.load_h5_solver_file import load_h5_solver_file
    from pde_analysis.analysis import (
        compute_energy,
        find_nearest_index,
        get_profile_at_tz,
        prepare_heatmap,
        prepare_surface_xz,
        prepare_xt_surface,
    )

    return (
        compute_energy,
        find_nearest_index,
        get_profile_at_tz,
        load_h5_solver_file,
        mo,
        plt,
        prepare_heatmap,
        prepare_surface_xz,
        prepare_xt_surface,
    )


@app.cell
def _(mo):
    folder_ui = mo.ui.text(
        value=r"C:\YandexDisk\Ioffe\workspace\one_decay\2D\data\sol\comparsion",
        label="data folder",
    )

    folder_ui
    return (folder_ui,)


@app.cell
def _(folder_ui):
    from pathlib import Path

    folder_path = Path(folder_ui.value)

    files = sorted(
        [str(p) for p in folder_path.glob("*.h5")]
    )
    return (files,)


@app.cell
def _(files, mo):
    file_select = mo.ui.dropdown(
        options=files,
        value=files[0] if files else None,
        label="file",
    )

    file_select
    return (file_select,)


@app.cell
def _(file_select, load_h5_solver_file):
    data_obj = load_h5_solver_file(file_select.value)
    return (data_obj,)


@app.cell
def _(data_obj, mo):
    t_vals_local = data_obj["grid"]["t"]
    z_vals_local = data_obj["grid"]["y"]

    t_slider_ui = mo.ui.slider(
        float(t_vals_local[0]),
        float(t_vals_local[-1]),
        value=float(t_vals_local[0]),
        step=float(t_vals_local[1] - t_vals_local[0]),
        label=f"t [{t_vals_local[0]:.2f}, {t_vals_local[-1]:.2f}]",
    )

    z_slider_ui = mo.ui.slider(
        float(z_vals_local[0]),
        float(z_vals_local[-1]),
        value=float(z_vals_local[0]),
        step=float(z_vals_local[1] - z_vals_local[0]),
        label=f"z [{z_vals_local[0]:.2f}, {z_vals_local[-1]:.2f}]",
    )

    # быстрые кнопки
    quick_values = [0, 3, 6, 9, 12, 15]

    quick_buttons = [
        mo.ui.button(label=f"t={v}", value=v) for v in quick_values
    ]

    quick_row = mo.hstack(quick_buttons)

    mo.vstack([t_slider_ui, quick_row, z_slider_ui])
    return quick_buttons, t_slider_ui, t_vals_local, z_slider_ui, z_vals_local


@app.cell
def _(quick_buttons, t_slider_ui):
    t_value = t_slider_ui.value


    for btn in quick_buttons:

        if btn.value:

            t_value = btn.value
    return (t_value,)


@app.cell
def _(find_nearest_index, t_vals_local, t_value, z_slider_ui, z_vals_local):
    t_idx_val = find_nearest_index(t_vals_local, t_value)
    z_idx_val = find_nearest_index(z_vals_local, z_slider_ui.value)
    return t_idx_val, z_idx_val


@app.cell
def _(data_obj, mo, t_vals_local):
    x_vals_local = data_obj["grid"]["x"]

    xmin_ui = mo.ui.slider(
        float(x_vals_local[0]),
        float(x_vals_local[-1]),
        value=float(x_vals_local[0]),
        label=f"xmin [{x_vals_local[0]:.1f}, {x_vals_local[-1]:.1f}]",
    )

    xmax_ui = mo.ui.slider(
        float(x_vals_local[0]),
        float(x_vals_local[-1]),
        value=float(x_vals_local[-1]),
        label=f"xmax [{x_vals_local[0]:.1f}, {x_vals_local[-1]:.1f}]",
    )

    tmax_ui = mo.ui.slider(
        float(t_vals_local[0]),
        float(t_vals_local[-1]),
        value=float(t_vals_local[-1]),
        label=f"tmax [{t_vals_local[0]:.1f}, {t_vals_local[-1]:.1f}]",
    )

    mo.vstack([xmin_ui, xmax_ui, tmax_ui])
    return tmax_ui, xmax_ui, xmin_ui


@app.cell
def _(mo):
    scale_ui = mo.ui.radio(
        options=["linear", "log"],
        value="linear",
        label="energy scale",
    )

    scale_ui
    return (scale_ui,)


@app.function
def make_profile_plot(data_obj, get_profile_at_tz, plt, t_idx_val, z_idx_val, xmin, xmax):
    res_a = get_profile_at_tz(data_obj, "a", t_idx_val, z_idx_val)
    res_b = get_profile_at_tz(data_obj, "b", t_idx_val, z_idx_val)

    fig = plt.figure()
    plt.plot(res_a["x"], res_a["profile"], label="|a|")
    plt.plot(res_b["x"], res_b["profile"], label="|b|")

    plt.xlim(xmin, xmax)
    plt.legend()

    return fig


@app.function
def make_surface_plot(data_obj, prepare_surface_xz, plt, t_idx_val, xmin, xmax):
    X_a, Z_a, A = prepare_surface_xz(data_obj, "a", t_idx_val)
    X_b, Z_b, B = prepare_surface_xz(data_obj, "b", t_idx_val)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(X_a, Z_a, A, alpha=0.5)
    ax.plot_surface(X_b, Z_b, B, alpha=0.5)

    ax.set_xlim(xmin, xmax)

    return fig


@app.function
def make_heatmap_plot(data_obj, prepare_heatmap, plt, t_idx_val, xmin, xmax):
    heat_a, extent = prepare_heatmap(data_obj, "a", t_idx_val)

    fig = plt.figure()
    plt.imshow(heat_a, origin="lower", aspect="auto", extent=extent)
    plt.xlim(xmin, xmax)
    plt.colorbar()

    return fig


@app.function
def make_xt_plot(data_obj, prepare_xt_surface, plt, z_idx_val, xmin, xmax):
    X, T, A = prepare_xt_surface(data_obj, "a", z_idx_val)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(X, T, A, alpha=0.7)
    ax.set_xlim(xmin, xmax)

    return fig


@app.function
def make_energy_plot(data_obj, compute_energy, plt, tmax, scale):
    Wa = compute_energy(data_obj, "a")
    Wb = compute_energy(data_obj, "b")

    mask = Wa["t"] <= tmax

    fig = plt.figure()

    plt.plot(Wa["t"][mask], Wa["W"][mask], label="Wa")
    plt.plot(Wb["t"][mask], Wb["W"][mask], label="Wb")

    if scale == "log":
        plt.yscale("log")

    plt.legend()
    return fig


@app.cell
def _(
    compute_energy,
    data_obj,
    get_profile_at_tz,
    mo,
    plt,
    prepare_heatmap,
    prepare_surface_xz,
    prepare_xt_surface,
    scale_ui,
    t_idx_val,
    tmax_ui,
    xmax_ui,
    xmin_ui,
    z_idx_val,
):
    tabs = mo.ui.tabs(
        {
            "energy": make_energy_plot(
                data_obj,
                compute_energy,
                plt,
                tmax_ui.value,
                scale_ui.value,
            ),
            "profile": make_profile_plot(
                data_obj,
                get_profile_at_tz,
                plt,
                t_idx_val,
                z_idx_val,
                xmin_ui.value,
                xmax_ui.value,
            ),
            "surface": make_surface_plot(
                data_obj,
                prepare_surface_xz,
                plt,
                t_idx_val,
                xmin_ui.value,
                xmax_ui.value,
            ),
            "heatmap": make_heatmap_plot(
                data_obj,
                prepare_heatmap,
                plt,
                t_idx_val,
                xmin_ui.value,
                xmax_ui.value,
            ),
            "evolution": make_xt_plot(
                data_obj,
                prepare_xt_surface,
                plt,
                z_idx_val,
                xmin_ui.value,
                xmax_ui.value,
            ),
        }
    )

    tabs
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

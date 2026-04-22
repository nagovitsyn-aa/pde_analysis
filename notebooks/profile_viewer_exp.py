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
    file = mo.ui.text(
        value=r"C:\YandexDisk\Ioffe\workspace\one_decay\2D\data\sol\comparsion\LSODA_Wed-W15_dx=0.02_Λ=0p1_u=1p.h5",
        label="file path",
    )

    file
    return (file,)


@app.cell
def _(file, load_h5_solver_file):
    data = load_h5_solver_file(file.value)
    return (data,)


@app.cell
def _(data, mo):
    t_vals = data["grid"]["t"]
    z_vals = data["grid"]["y"]

    t_slider = mo.ui.slider(
        float(t_vals[0]),
        float(t_vals[-1]),
        value=float(t_vals[0]),
        step=float(t_vals[1] - t_vals[0]),
        label="t",
    )

    z_slider = mo.ui.slider(
        float(z_vals[0]),
        float(z_vals[-1]),
        value=float(z_vals[0]),
        step=float(z_vals[1] - z_vals[0]),
        label="z",
    )

    mo.vstack([t_slider, z_slider])
    return t_slider, t_vals, z_slider, z_vals


@app.cell
def _(find_nearest_index, t_slider, t_vals, z_slider, z_vals):
    t_index = find_nearest_index(t_vals, t_slider.value)
    z_index = find_nearest_index(z_vals, z_slider.value)
    return t_index, z_index


@app.cell
def _(data, get_profile_at_tz, plt, t_index, z_index):
    res_a = get_profile_at_tz(data, "a", t_index, z_index)
    res_b = get_profile_at_tz(data, "b", t_index, z_index)

    plt.figure()
    plt.plot(res_a["x"], res_a["profile"], label="|a|")
    plt.plot(res_b["x"], res_b["profile"], label="|b|")

    plt.xlabel("x")
    plt.ylabel("amplitude")
    plt.title(f"t={res_a['t']:.3f}, z={res_a['z']:.3f}")
    plt.legend()

    plt.gca()
    return


@app.cell
def _(compute_energy, data):
    Wa = compute_energy(data, "a")
    Wb = compute_energy(data, "b")
    return Wa, Wb


@app.cell
def _(Wa, Wb, plt):
    plt.figure()

    plt.plot(Wa["t"], Wa["W"], label="Wa")
    plt.plot(Wb["t"], Wb["W"], "--", label="Wb")

    plt.xlabel("t")
    plt.ylabel("energy")
    plt.title("Energy vs time")
    plt.legend()

    plt.gca()
    return


@app.cell
def _(data, plt, prepare_surface_xz, t_index):
    X_a, Z_a, A_surface = prepare_surface_xz(data, "a", t_index)
    X_b, Z_b, B_surface = prepare_surface_xz(data, "b", t_index)

    fig_surface = plt.figure()
    ax_surface = fig_surface.add_subplot(111, projection="3d")

    ax_surface.plot_surface(X_a, Z_a, A_surface, alpha=0.5)
    ax_surface.plot_surface(X_b, Z_b, B_surface, alpha=0.5)

    ax_surface.set_xlabel("x")
    ax_surface.set_ylabel("z")
    ax_surface.set_zlabel("amplitude")

    plt.gca()
    return


@app.cell
def _(data, plt, prepare_heatmap, t_index):
    heatmap_a, extent_a = prepare_heatmap(data, "a", t_index)
    heatmap_b, extent_b = prepare_heatmap(data, "b", t_index)

    fig_a = plt.figure()
    plt.imshow(heatmap_a, origin="lower", aspect="auto", extent=extent_a)
    plt.title("|a|")
    plt.colorbar()

    fig_b = plt.figure()
    plt.imshow(heatmap_b, origin="lower", aspect="auto", extent=extent_b)
    plt.title("|b|")
    plt.colorbar()

    plt.gca()
    return


@app.cell
def _(data, plt, prepare_xt_surface, z_index):
    X_xt, T_xt, A_xt = prepare_xt_surface(data, "a", z_index)

    fig_xt = plt.figure()
    ax_xt = fig_xt.add_subplot(111, projection="3d")

    ax_xt.plot_surface(X_xt, T_xt, A_xt, alpha=0.7)

    ax_xt.set_xlabel("x")
    ax_xt.set_ylabel("t")
    ax_xt.set_zlabel("|a|")

    plt.gca()
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

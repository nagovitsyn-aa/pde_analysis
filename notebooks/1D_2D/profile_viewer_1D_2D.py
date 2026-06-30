import marimo

__generated_with = "0.23.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from pde_analysis.load_h5_solver_file import (
        load_h5_solver_file,
        load_h5_solver_file_1d,
    )

    from pde_analysis.analysis import get_profile_at_tz

    return (
        get_profile_at_tz,
        load_h5_solver_file,
        load_h5_solver_file_1d,
        mo,
        np,
        plt,
    )


@app.cell
def _(mo):
    file_1d = mo.ui.text(
        value=r"",
        label="1D file",
        full_width=True,
    )

    file_2d = mo.ui.text(
        value=r"",
        label="2D file",
        full_width=True,
    )

    mo.vstack([file_1d, file_2d])
    return file_1d, file_2d


@app.cell
def _(file_1d, file_2d, load_h5_solver_file, load_h5_solver_file_1d):
    data_1d = load_h5_solver_file_1d(file_1d.value)
    data_2d = load_h5_solver_file(file_2d.value)
    return data_1d, data_2d


@app.cell
def _(data_1d, data_2d, mo):
    t1 = data_1d["grid"]["t"]
    t2 = data_2d["grid"]["t"]

    t_max = min(t1[-1], t2[-1])

    t_ui = mo.ui.slider(
        start=0.0,
        stop=float(t_max),
        step=float(t_max / 500),
        value=0.0,
        label="time",
    )

    field_ui = mo.ui.radio(
        options=["a", "b"],
        value="a",
        label="field",
    )

    z_vals = data_2d["grid"]["y"]

    z_ui = mo.ui.slider(
        0,
        len(z_vals) - 1,
        value=len(z_vals) // 2,
        label="z index",
    )

    x = data_1d["grid"]["x"].squeeze()

    xmin_ui = mo.ui.number(
        value=float(x[0]),
        label="xmin",
    )

    xmax_ui = mo.ui.number(
        value=float(x[-1]),
        label="xmax",
    )

    mo.vstack(
        [
            field_ui,
            t_ui,
            z_ui,
            mo.hstack([xmin_ui, xmax_ui]),
        ]
    )
    return field_ui, t_ui, x, xmax_ui, xmin_ui, z_ui


@app.cell
def _(data_1d, data_2d, field_ui, get_profile_at_tz, np, t_ui, z_ui):
    t_selected = t_ui.value

    t_grid_1d = data_1d["grid"]["t"]
    idx1 = np.argmin(np.abs(t_grid_1d - t_selected))

    t_grid_2d = data_2d["grid"]["t"]
    idx2 = np.argmin(np.abs(t_grid_2d - t_selected))

    field = field_ui.value

    if field == "a":
        re = data_1d["solution"]["aRe"][idx1]
        im = data_1d["solution"]["aIm"][idx1]
    else:
        re = data_1d["solution"]["bRe"][idx1]
        im = data_1d["solution"]["bIm"][idx1]

    profile_1d = np.sqrt(re**2 + im**2)

    profile_2d = get_profile_at_tz(
        data_2d,
        field,
        idx2,
        z_ui.value,
    )
    return idx1, profile_1d, profile_2d


@app.cell
def _(data_1d, idx1, plt, profile_1d, profile_2d, x, xmax_ui, xmin_ui):
    plt.figure(figsize=(8, 4.5))

    plt.plot(
        x,
        profile_1d,
        lw=2,
        label="1D",
    )

    plt.plot(
        profile_2d["x"],
        profile_2d["profile"],
        "--",
        lw=2,
        label="2D",
    )

    plt.xlim(
        xmin_ui.value,
        xmax_ui.value,
    )

    plt.xlabel("x")
    plt.ylabel("Amplitude")

    plt.title(
        (
            f"t(1D)={data_1d['grid']['t'][idx1]:.4f}, "
            f"t(2D)={profile_2d['t']:.4f}, "
            f"z={profile_2d['z']:.4f}"
        )
    )

    plt.grid(True)
    plt.legend()

    plt.gca()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

import marimo

__generated_with = "0.23.2"
app = marimo.App()


@app.cell
def _():
    file = r"C:\YandexDisk\Ioffe\workspace\one_decay\2D\data\sol\comparsion_gauss\LSODA_Tue-W27_dx=0.05_Λ=0p1_u=1.h5"

    dat_3t = r"C:\YandexDisk\Ioffe\workspace\one_decay\2D\data\sol\comparsion_gauss\centr_a_b_3t_1u_1e-1D.dat"
    dat_6t = r"C:\YandexDisk\Ioffe\workspace\one_decay\2D\data\sol\comparsion_gauss\centr_a_b_6t_1u_1e-1D.dat"
    dat_Wa = r"C:\YandexDisk\Ioffe\workspace\one_decay\2D\data\sol\comparsion_gauss\centr_Wa_Wb_1u_1e-1D.dat"

    xmin, xmax = -10, 10

    t_val_3 = 3.0
    t_val_6 = 6.0
    z_val = 0.0

    t_min, t_max = 0, 15
    return (
        dat_3t,
        dat_6t,
        dat_Wa,
        file,
        t_max,
        t_min,
        t_val_3,
        t_val_6,
        xmax,
        xmin,
        z_val,
    )


@app.cell
def _(file):
    from pde_analysis.load_h5_solver_file import load_h5_solver_file

    data = load_h5_solver_file(file)
    return (data,)


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    from pde_analysis.analysis import get_profile_at_tz, crop_x, compute_energy
    from pde_analysis.io_dat import load_ab_dat, load_Wa_Wb
    import numpy as np


    return (
        compute_energy,
        crop_x,
        get_profile_at_tz,
        load_Wa_Wb,
        load_ab_dat,
        mo,
        np,
        plt,
    )


@app.cell
def _(crop_x, get_profile_at_tz, load_ab_dat, np, plt):

    def find_nearest_index(grid, value):
        grid = np.asarray(grid)
        return int(np.argmin(np.abs(grid - value)))


    def plot_profile(data, t_value, z_value, dat_file, xmin, xmax, title):
        t_grid = data["grid"]["t"]
        z_grid = data["grid"]["y"]

        t_index = find_nearest_index(t_grid, t_value)
        z_index = find_nearest_index(z_grid, z_value)

        res_a = get_profile_at_tz(data, "a", t_index, z_index)
        res_b = get_profile_at_tz(data, "b", t_index, z_index)

        x_a, a = crop_x(res_a["x"], res_a["profile"], xmin, xmax)
        x_b, b = crop_x(res_b["x"], res_b["profile"], xmin, xmax)

        dat = load_ab_dat(dat_file)

        x_dat, a_dat = crop_x(dat["x"], dat["b"], xmin, xmax)
        _, b_dat = crop_x(dat["x"], dat["a"], xmin, xmax)

        plt.figure()

        plt.plot(x_a, a, alpha=0.6, label="|a| A")
        plt.plot(x_b, b, alpha=0.6, label="|b| A")

        plt.plot(x_dat, a_dat, "--", label="|a| S")
        plt.plot(x_dat, b_dat, "--", label="|b| S")

        plt.xlim(xmin, xmax)
        plt.legend()

        plt.title(f"{title} (t={t_grid[t_index]:.2f})")

        return plt.gcf()

    return (plot_profile,)


@app.cell
def _(dat_3t, data, plot_profile, t_val_3, xmax, xmin, z_val):
    plot_profile(data, t_val_3, z_val, dat_3t, xmin, xmax, "")
    return


@app.cell
def _(dat_6t, data, plot_profile, t_val_6, xmax, xmin, z_val):
    plot_profile(data, t_val_6, z_val, dat_6t, xmin, xmax, "")
    return


@app.cell
def _(mo):
    energy_field = mo.ui.radio(
        options=[
            "Wa",
            "Wb",
            "Wa + Wb",
        ],
        value="Wa",
        label="Energy",
    )

    normalize = mo.ui.switch(
        value=True,
        label="Normalize",
    )

    mo.vstack([
        energy_field,
        normalize,
    ])
    return energy_field, normalize


@app.cell
def _(compute_energy, dat_Wa, data, load_Wa_Wb, t_max, t_min):
    Wa_model = compute_energy(data, "a", t_min, t_max)
    Wb_model = compute_energy(data, "b", t_min, t_max)

    energy_dat = load_Wa_Wb(dat_Wa)
    return Wa_model, Wb_model, energy_dat


@app.cell
def _(Wa_model, Wb_model, energy_dat, energy_field, normalize, np, plt):

    def maybe_normalize(values):
        if not normalize.value:
            return values

        if np.isclose(values[0], 0.0):
            print("Warning: first value is zero, normalization skipped.")
            return values

        return values / values[0]

    plt.figure()

    mode = energy_field.value

    if mode in ("Wa", "Wa + Wb"):
        plt.plot(
            Wa_model["t"],
            2*maybe_normalize(Wa_model["W"]),
            alpha=0.8,
            label="Wa A",
        )

        plt.plot(
            energy_dat["t"],
            maybe_normalize(energy_dat["Wa"]),
            "--",
            label="Wa S",
        )

    if mode in ("Wb", "Wa + Wb"):
        plt.plot(
            Wb_model["t"],
            2*maybe_normalize(Wb_model["W"]),
            alpha=0.6,
            label="Wb A",
        )

        plt.plot(
            energy_dat["t"],
            maybe_normalize(energy_dat["Wb"]),
            "--",
            label="Wb S",
        )

    ylabel = "Energy"

    if normalize.value:
        ylabel += " / first value"

    plt.xlabel("t")
    plt.ylabel(ylabel)
    plt.title(mode)
    plt.legend()

    plt.gca()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

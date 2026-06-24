import marimo

__generated_with = "0.23.2"
app = marimo.App()


@app.cell
def _():
    file = r"C:\\YandexDisk\\Ioffe\\workspace\\one_decay\\2D\\data\\sol\\comparsion\\fullRange_TD_LSODA_Mon-W17_dx=0.02_Λ=0p1_u=1p.h5"

    dat_3t = r"C:\YandexDisk\Ioffe\workspace\tempdata\a_b_3t.dat"
    dat_6t = r"C:\YandexDisk\Ioffe\workspace\tempdata\a_b_6t.dat"
    dat_Wa = r"C:\YandexDisk\Ioffe\workspace\tempdata\Wa(t).dat"

    xmin, xmax = -20, 20

    t_val_3 = 3.0
    t_val_6 = 6.0
    z_val = 0.1

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
    import matplotlib.pyplot as plt
    from pde_analysis.analysis import get_profile_at_tz, crop_x, compute_energy
    from pde_analysis.io_dat import load_ab_dat, load_Wa
    import numpy as np


    return (
        compute_energy,
        crop_x,
        get_profile_at_tz,
        load_Wa,
        load_ab_dat,
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

        plt.plot(x_a, a, label="|a| A")
        plt.plot(x_b, b, label="|b| A")

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
def _(compute_energy, dat_Wa, data, load_Wa, t_max, t_min):
    Wa_model = compute_energy(data, "a", t_min, t_max)
    Wa_dat = load_Wa(dat_Wa)
    return Wa_dat, Wa_model


@app.cell
def _(Wa_dat, Wa_model, plt):

    W_model_norm = Wa_model["W"] / Wa_model["W"][0]
    Wa_dat_norm = Wa_dat["Wa"] / Wa_dat["Wa"][0]

    plt.figure()
    plt.plot(Wa_model["t"], W_model_norm, label="A (norm)")
    plt.plot(Wa_dat["t"], Wa_dat_norm, "--", label="S (norm)")

    plt.xlabel("t")
    plt.ylabel("Wa / Wa(t0)")  
    plt.legend()
    plt.title("Energy comparison")

    # plt.yscale('log') 
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

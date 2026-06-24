import marimo

__generated_with = "0.23.2"
app = marimo.App()


@app.cell
def _():
    file_1d = r"C:\YandexDisk\Yandex.Disk\Ioffe\workspace\one_decay\comparsion\u=1_w0=0.5_1D\Mon-W26_u=1p_w0x=0p5.h5"

    dat_1d_3t = r"C:\YandexDisk\Yandex.Disk\Ioffe\workspace\one_decay\comparsion\u=1_w0=0.5_1D\a_b_3t_1d.dat"
    dat_1d_6t = r"C:\YandexDisk\Yandex.Disk\Ioffe\workspace\one_decay\comparsion\u=1_w0=0.5_1D\a_b_6t_1d.dat"

    file_2d_lmbd001 = r"C:\YandexDisk\Yandex.Disk\Ioffe\workspace\one_decay\comparsion\u=1_w0=0.5_2D\LSODA_Mon-W26_dx=0.01_Λ=0p01_u=1p.h5"
    file_2d_lmbd01 = r"C:\YandexDisk\Yandex.Disk\Ioffe\workspace\one_decay\comparsion\u=1_w0=0.5_2D\LSODA_Mon-W26_dx=0.01_Λ=0p1_u=1p.h5"

    dat_2d_3t = r"C:\YandexDisk\Yandex.Disk\Ioffe\workspace\one_decay\comparsion\u=1_w0=0.5_2D\S, Lambda=0.1\a_b_3t.dat"
    dat_2d_6t = r"C:\YandexDisk\Yandex.Disk\Ioffe\workspace\one_decay\comparsion\u=1_w0=0.5_2D\S, Lambda=0.1\a_b_6t.dat"

    xmin, xmax = -20, 20

    t_val_3 = 3.0
    t_val_6 = 6.0
    z_val = 0.1
    return (
        dat_1d_3t,
        dat_1d_6t,
        dat_2d_3t,
        dat_2d_6t,
        file_1d,
        file_2d_lmbd001,
        file_2d_lmbd01,
        t_val_3,
        t_val_6,
        xmax,
        xmin,
        z_val,
    )


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np

    from pde_analysis.load_h5_solver_file import (
        load_h5_solver_file,
        load_h5_solver_file_1d,
    )
    from pde_analysis.analysis import get_profile_at_tz, crop_x
    from pde_analysis.io_dat import load_ab_dat

    return (
        crop_x,
        get_profile_at_tz,
        load_ab_dat,
        load_h5_solver_file,
        load_h5_solver_file_1d,
        np,
        plt,
    )


@app.cell
def _(file_1d, load_h5_solver_file_1d):
    data_1d = load_h5_solver_file_1d(file_1d)
    return (data_1d,)


@app.cell
def _(file_2d_lmbd001, load_h5_solver_file):
    data_2d_lmbd001 = load_h5_solver_file(file_2d_lmbd001)
    return (data_2d_lmbd001,)


@app.cell
def _(file_2d_lmbd01, load_h5_solver_file):
    data_2d_lmbd01 = load_h5_solver_file(file_2d_lmbd01)
    return (data_2d_lmbd01,)


@app.cell
def _(crop_x, get_profile_at_tz, load_ab_dat, np, plt):
    def find_nearest_index(grid, value):
        grid = np.asarray(grid)
        return int(np.argmin(np.abs(grid - value)))


    def get_1d_a_profile(data_1d, t_value, xmin, xmax):
        t_grid = np.asarray(data_1d["grid"]["t"]).squeeze()
        x_grid = np.asarray(data_1d["grid"]["x"]).squeeze()

        t_index = find_nearest_index(t_grid, t_value)

        a_re = np.asarray(data_1d["solution"]["bRe"][t_index]).squeeze()
        a_im = np.asarray(data_1d["solution"]["bIm"][t_index]).squeeze()

        a_abs = np.sqrt(a_re**2 + a_im**2)

        x_crop, a_crop = crop_x(x_grid, a_abs, xmin, xmax)

        return {
            "x": x_crop,
            "a": a_crop,
            "t": t_grid[t_index],
            "t_index": t_index,
        }

    def get_2d_a_profile(data_2d, t_value, z_value, xmin, xmax):
        t_grid = np.asarray(data_2d["grid"]["t"]).squeeze()
        z_grid = np.asarray(data_2d["grid"]["y"]).squeeze()

        t_index = find_nearest_index(t_grid, t_value)
        z_index = find_nearest_index(z_grid, z_value)

        res_a = get_profile_at_tz(data_2d, "b", t_index, z_index)

        x_vals = np.asarray(res_a["x"]).squeeze()
        a_vals = np.asarray(res_a["profile"]).squeeze()

        x_crop, a_crop = crop_x(x_vals, a_vals, xmin, xmax)

        return {
            "x": x_crop,
            "a": a_crop,
            "t": t_grid[t_index],
            "t_index": t_index,
            "z_index": z_index,
        }


    def load_1d_dat_profile(dat_file, xmin, xmax):
        dat = np.loadtxt(dat_file)

        x_dat = np.asarray(dat[:, 0]).squeeze()
        a_dat = np.asarray(dat[:, 1]).squeeze()

        x_crop, a_crop = crop_x(x_dat, a_dat, xmin, xmax)

        return {
            "x": x_crop,
            "a": a_crop,
        }


    def load_2d_s_a_profile(dat_file, xmin, xmax):
        dat = load_ab_dat(dat_file)

        x_dat = np.asarray(dat["x"]).squeeze()
        a_dat = np.asarray(dat["b"]).squeeze()

        x_crop, a_crop = crop_x(x_dat, a_dat, xmin, xmax)

        return {
            "x": x_crop,
            "a": a_crop,
        }


    def plot_a_comparison(
        data_1d,
        data_2d_lmbd001,
        data_2d_lmbd01,
        t_value,
        z_value,
        dat_1d_file,
        dat_2d_file,
        xmin,
        xmax,
        title="",
    ):
        prof_1d = get_1d_a_profile(data_1d, t_value, xmin, xmax)
        prof_2d_lmbd001 = get_2d_a_profile(data_2d_lmbd001, t_value, z_value, xmin, xmax)
        prof_2d_lmbd01 = get_2d_a_profile(data_2d_lmbd01, t_value, z_value, xmin, xmax)

        prof_1d_s = load_1d_dat_profile(dat_1d_file, xmin, xmax)
        prof_2d_s = load_2d_s_a_profile(dat_2d_file, xmin, xmax)

        plt.figure(figsize=(8, 5))

        plt.plot(
            prof_1d["x"],
            prof_1d["a"],
            color="gray",
            alpha=0.5,
            linewidth=2,
            label="1D A",
        )

        plt.plot(
            prof_1d_s["x"],
            prof_1d_s["a"],
            "--",
            color="black",
            linewidth=2,
            label="1D S",
        )

        plt.plot(
            prof_2d_lmbd001["x"],
            prof_2d_lmbd001["a"],
            color="blue",
            alpha=0.8,
            linewidth=2,
            label=r"2D A, $\Lambda=0.01$",
        )

        plt.plot(
            prof_2d_lmbd01["x"],
            prof_2d_lmbd01["a"],
            alpha=0.5,
            color="red",
            linewidth=2,
            label=r"2D A, $\Lambda=0.1$",
        )

        plt.plot(
            prof_2d_s["x"],
            prof_2d_s["a"],
            "--",
            color="red",
            linewidth=2,
            label=r"2D S, $\Lambda=0.1$",
        )

        plt.xlim(xmin, xmax)
        plt.xlabel("x")
        plt.ylabel(r"$|a|$")
        plt.legend()
        plt.title(f"{title} (t={prof_2d_lmbd01['t']:.2f})")

        return plt.gcf()


    return (plot_a_comparison,)


@app.cell
def _(
    dat_1d_3t,
    dat_2d_3t,
    data_1d,
    data_2d_lmbd001,
    data_2d_lmbd01,
    plot_a_comparison,
    t_val_3,
    xmax,
    xmin,
    z_val,
):
    plot_a_comparison(
        data_1d=data_1d,
        data_2d_lmbd001=data_2d_lmbd001,
        data_2d_lmbd01=data_2d_lmbd01,
        t_value=t_val_3,
        z_value=z_val,
        dat_1d_file=dat_1d_3t,
        dat_2d_file=dat_2d_3t,
        xmin=xmin,
        xmax=xmax,
        title="u=1, w0x=w0z=0.5",
    )
    return


@app.cell
def _(
    dat_1d_6t,
    dat_2d_6t,
    data_1d,
    data_2d_lmbd001,
    data_2d_lmbd01,
    plot_a_comparison,
    t_val_6,
    xmax,
    xmin,
    z_val,
):
    plot_a_comparison(
        data_1d=data_1d,
        data_2d_lmbd001=data_2d_lmbd001,
        data_2d_lmbd01=data_2d_lmbd01,
        t_value=t_val_6,
        z_value=z_val,
        dat_1d_file=dat_1d_6t,
        dat_2d_file=dat_2d_6t,
        xmin=xmin,
        xmax=xmax,
        title="u=1, w0x=w0z=0.5",
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

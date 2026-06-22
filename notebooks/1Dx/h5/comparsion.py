import marimo

__generated_with = "0.23.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from pde_analysis.load_h5_solver_file import load_h5_solver_file_1d
    from pde_analysis.analysis import compute_energy_1d

    return load_h5_solver_file_1d, mo, np, plt


@app.cell
def _(mo):
    file = mo.ui.text(
        value=r"C:\YandexDisk\Yandex.Disk\Ioffe\workspace\one_decay\1Dx\data\delta vs zero\HDF5\comparsion\Fri-W25_u=0p316228_w0x=0p05.h5",
        label="file path",
        full_width=True,
    )

    file 
    return (file,)


@app.cell
def _(file, load_h5_solver_file_1d):
    data = load_h5_solver_file_1d(file.value)
    return (data,)


@app.cell
def _(a_im_center, a_re_center, np):
    np.sqrt(a_re_center**2 + a_im_center**2)
    return


@app.cell
def _(data, np):
    x = data["grid"]["x"]
    t = data["grid"]["t"]

    center_idx = np.argmin(np.abs(x))

    a_re_center = data["solution"]["aRe"][:, center_idx]
    a_im_center = data["solution"]["aIm"][:, center_idx]

    b_re_center = data["solution"]["bRe"][:, center_idx]
    b_im_center = data["solution"]["bIm"][:, center_idx]

    a_abs_center = np.sqrt(
        a_re_center**2 + a_im_center**2
    )

    b_abs_center = np.sqrt(
        b_re_center**2 + b_im_center**2
    )
    return a_abs_center, a_im_center, a_re_center, b_abs_center, t


@app.cell
def _(mo):
    dat_file = mo.ui.text(
        value=r"C:\YandexDisk\Yandex.Disk\Ioffe\workspace\one_decay\1Dx\data\delta vs zero\HDF5\comparsion\lmbd10_8192_1e-2dt_5e-2wx_ros73.dat",
        label="dat file",
        full_width=True,
    )

    dat_file
    return (dat_file,)


@app.cell
def _(dat_file, np):
    dat_data = np.loadtxt(dat_file.value)

    t_ref = dat_data[:, 0]
    amp_ref = dat_data[:, 1]
    return amp_ref, t_ref


@app.cell
def _(a_abs_center, amp_ref, b_abs_center, plt, t, t_ref):
    plt.figure(figsize=(8, 5))

    plt.plot(
        t,
        b_abs_center,
        label="|a(0,t)|"
    )
    plt.plot(
        t,
        a_abs_center,
        label="|b(0,t)|"
    )



    plt.plot(
        t_ref,
        amp_ref,
        "--",
        linewidth=2,
        label="reference"
    )

    plt.xlabel("t")
    plt.ylabel("amplitude")
    plt.title("Central amplitude comparison")
    plt.grid(True)
    plt.legend()

    plt.gca()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

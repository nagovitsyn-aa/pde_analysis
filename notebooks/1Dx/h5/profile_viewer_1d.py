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

    return compute_energy_1d, load_h5_solver_file_1d, mo, np, plt


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
def _(data, mo):
    t_vals = data["grid"]["t"]

    t_index = mo.ui.slider(0, len(t_vals) - 1, value=0, label="t index")

    t_index
    return (t_index,)


@app.cell
def _(data, np, plt, t_index):
    t_idx = t_index.value

    x = data["grid"]["x"]
    t = data["grid"]["t"][t_idx]

    a_re = data["solution"]["aRe"][t_idx]
    a_im = data["solution"]["aIm"][t_idx]
    b_re = data["solution"]["bRe"][t_idx]
    b_im = data["solution"]["bIm"][t_idx]

    a_abs = np.sqrt(a_re**2 + a_im**2)
    b_abs = np.sqrt(b_re**2 + b_im**2)

    plt.figure()
    plt.plot(x, a_abs, label="|a|")
    plt.plot(x, b_abs, label="|b|")

    plt.xlabel("x")
    plt.ylabel("amplitude")
    plt.title(f"t = {t:.3f}")
    plt.legend()

    plt.gca()
    return t, x


@app.cell
def _(compute_energy_1d, data):
    Wa = compute_energy_1d(data, "a")
    Wb = compute_energy_1d(data, "b")
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
def _(np, x):
    center_idx = np.argmin(np.abs(x))

    center_idx
    return (center_idx,)


@app.cell
def _(t):
    t
    return


@app.cell
def _(a_abs_center):
    a_abs_center.shape
    return


@app.cell
def _(a_im_center, a_re_center, np):
    np.sqrt(a_re_center**2 + a_im_center**2)
    return


@app.cell
def _(center_idx, data, np, plt, t):
    t_full = data["grid"]["t"]

    a_re_center = data["solution"]["aRe"][:, center_idx]
    a_im_center = data["solution"]["aIm"][:, center_idx]

    b_re_center = data["solution"]["bRe"][:, center_idx]
    b_im_center = data["solution"]["bIm"][:, center_idx]

    a_abs_center = np.sqrt(a_re_center**2 + a_im_center**2)
    b_abs_center = np.sqrt(b_re_center**2 + b_im_center**2)

    print(t.shape)
    print(a_abs_center.shape)

    plt.figure()

    plt.plot(t_full, a_abs_center, label="|a(0,t)|")
    plt.plot(t_full, b_abs_center, label="|b(0,t)|")

    plt.xlabel("t")
    plt.ylabel("amplitude")
    plt.title("Central amplitude vs time")
    plt.legend()

    plt.gca()
    return a_abs_center, a_im_center, a_re_center, b_abs_center, t_full


@app.cell
def _(t_full):
    t_full
    return


@app.cell
def _(b_abs_center):
    b_abs_center
    return


@app.cell
def _(center_idx, data):
    x0 = data["grid"]["x"][center_idx]
    print(f"Central node: x = {x0}")
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

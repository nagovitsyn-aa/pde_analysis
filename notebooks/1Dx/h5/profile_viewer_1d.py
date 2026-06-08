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
        value=r"C:\YandexDisk\Yandex.Disk\Ioffe\workspace\one_decay\1Dx\data\HDF5\Sun-W20_u=0p6.h5",
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
    return


@app.cell
def _(compute_energy_1d, data):
    Wa = compute_energy_1d(data, "a")
    Wb = compute_energy_1d(data, "a")
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

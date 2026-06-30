import marimo

__generated_with = "0.23.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt

    from pde_analysis.load_h5_solver_file import load_h5_solver_file
    from pde_analysis.analysis import get_profile_at_tz, compute_energy

    return compute_energy, get_profile_at_tz, load_h5_solver_file, mo, plt


@app.cell
def _(mo):
    file = mo.ui.text(
        value=r"C:\YandexDisk\Ioffe\workspace\one_decay\2D\data\sol\comparsion_gauss\LSODA_Wed-W26_dx=0.05_Λ=0p01_u=0p316228.h5",
        label="file path",
        full_width=True,
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

    t_index = mo.ui.slider(0, len(t_vals) - 1, value=0, label="t index")
    z_index = mo.ui.slider(0, len(z_vals) - 1, value=0, label="z index")

    mo.vstack([t_index, z_index])  # ВАЖНО
    return t_index, z_index


@app.cell
def _(data, get_profile_at_tz, plt, t_index, z_index):
    res_a = get_profile_at_tz(data, "a", t_index.value, z_index.value)
    res_b = get_profile_at_tz(data, "b", t_index.value, z_index.value)

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
    plt.plot(Wb["t"], Wb["W"], "--",label="Wb")

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


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

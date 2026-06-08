import marimo

__generated_with = "0.23.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from pathlib import Path
    from pde_analysis.load_h5_solver_file import load_h5_solver_file_1d
    from pde_analysis.analysis import compute_energy_1d

    return Path, compute_energy_1d, load_h5_solver_file_1d, mo, np, plt


@app.cell
def _(mo):
    folder_ui = mo.ui.text(
        value=r"C:\YandexDisk\Yandex.Disk\Ioffe\workspace\one_decay\1Dx\data\HDF5\w0x=1",
        label="data folder",
        full_width=True,
    )

    folder_ui
    return (folder_ui,)


@app.cell
def _(Path, folder_ui):
    folder_path = Path(folder_ui.value)
    files = sorted([str(p) for p in folder_path.glob("*.h5")])
    return (files,)


@app.cell
def _(files, mo):
    files_checkbox = mo.ui.multiselect(
        options=files,
        value=files[:1] if files else [],
        label="data files",
    )

    files_checkbox
    return (files_checkbox,)


@app.cell
def _(compute_energy_1d, files_checkbox, load_h5_solver_file_1d, np):
    results = []

    for filepath in files_checkbox.value:
        data = load_h5_solver_file_1d(filepath)
        Wa = compute_energy_1d(data, "a")

        t = np.asarray(Wa["t"])
        W = np.asarray(Wa["W"])
        positive = W > 0
        t = t[positive]
        W = W[positive]

        if len(t) < 3:
            continue

        logW = np.log(W)
        dlogW = np.gradient(logW, t)
        max_idx = int(np.nanargmax(dlogW))
        max_increment = float(dlogW[max_idx])
        t_max_increment = float(t[max_idx])
        predicted_growth = float(np.exp(max_increment * (t[-1] - t[0])))
        actual_growth = float(W[-1] / W[0])

        params = data.get("parameters", {})
        u_value = params.get("u")

        results.append(
            {
                "file": filepath,
                "label": filepath.split("\\")[-1],
                "u": u_value,
                "t": t,
                "logW": logW,
                "dlogW": dlogW,
                "max_increment": max_increment,
                "t_max_increment": t_max_increment,
                "predicted_growth": predicted_growth,
                "actual_growth": actual_growth,
            }
        )
    return (results,)


@app.cell
def _(mo, results):
    labels = [item["label"] for item in results]
    visible_ui = mo.ui.multiselect(
        options=labels,
        value=labels,
        label="visible files",
    )

    visible_ui
    return (visible_ui,)


@app.cell
def _(mo, results):
    selected_label_ui = mo.ui.radio(
        options=[item["label"] for item in results],
        value=results[0]["label"] if results else None,
        label="detail file",
    )

    selected_label_ui
    return (selected_label_ui,)


@app.cell
def _(plt, results, selected_label_ui, visible_ui):
    _fig, _axs = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    for item in results:
        if item["label"] not in visible_ui.value:
            continue
        _axs[0].plot(item["t"], item["logW"], label=item["label"])
        _axs[1].plot(item["t"], item["dlogW"], label=item["label"])

    selected = next((item for item in results if item["label"] == selected_label_ui.value), None)
    if selected is not None:
        _axs[1].scatter(
            [selected["t_max_increment"]],
            [selected["max_increment"]],
            color="red",
            marker="x",
            s=80,
            label=f"max increment {selected['label']}",
        )

    _axs[0].set_ylabel("log Wa")
    _axs[1].set_ylabel("d/dt log Wa")
    _axs[1].set_xlabel("t")

    _axs[0].grid(True)
    _axs[1].grid(True)
    _axs[0].legend()
    _axs[1].legend()
    _fig.tight_layout()

    _fig
    return


@app.cell
def _(plt, results, visible_ui):
    _fig, _ax = plt.subplots(figsize=(8, 5))

    scatter_data = [item for item in results if item["label"] in visible_ui.value and item["u"] is not None]
    scatter_data.sort(key=lambda x: x["u"])

    if scatter_data:
        u_values = [item["u"] for item in scatter_data]
        increments = [item["max_increment"] for item in scatter_data]
        _ax.plot(u_values, increments, marker="o", linestyle="-", label="max increment")

    _ax.set_xlabel("u")
    _ax.set_ylabel("max increment")
    _ax.grid(True)
    _ax.legend()
    _fig.tight_layout()

    _fig
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

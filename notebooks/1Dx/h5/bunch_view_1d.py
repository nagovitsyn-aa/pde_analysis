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
    value=r"C:\YandexDisk\Yandex.Disk\Ioffe\workspace\one_decay\1Dx\data\HDF5\w0x=0p05",
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
        label="files",
    )

    files_checkbox
    return (files_checkbox,)


@app.cell
def _(compute_energy_1d, files_checkbox, load_h5_solver_file_1d, np):
    energy_data = []

    for filepath in files_checkbox.value:
        data_obj = load_h5_solver_file_1d(filepath)
        Wa = compute_energy_1d(data_obj, "a")
        Wb = compute_energy_1d(data_obj, "b")

        def _get_centers(d):
            x = np.asarray(d["grid"]["x"])
            center_idx = len(x) // 2

            a_re = np.asarray(d["solution"]["aRe"])[:, center_idx]
            a_im = np.asarray(d["solution"]["aIm"])[:, center_idx]
            a_c = np.sqrt(a_re**2 + a_im**2)

            b_re = np.asarray(d["solution"]["bRe"])[:, center_idx]
            b_im = np.asarray(d["solution"]["bIm"])[:, center_idx]
            b_c = np.sqrt(b_re**2 + b_im**2)

            return a_c, b_c

        _centers = _get_centers(data_obj)

        energy_data.append(
            {
                "file": filepath,
                "t": Wa["t"],
                "Wa": Wa["W"],
                "Wb": Wb["W"],
                "a_center": _centers[0],
                "b_center": _centers[1],
            }
        )
    return (energy_data,)


@app.cell
def _(energy_data, mo):
    labels = [item["file"].split("\\")[-1] for item in energy_data]

    visibility_ui = mo.ui.multiselect(
        options=labels,
        value=labels,
        label="visible curves",
    )

    visibility_ui
    return (visibility_ui,)


@app.cell
def _(mo):
    scale_ui = mo.ui.radio(
        options=["linear", "log"],
        value="linear",
        label="scale",
    )

    scale_ui
    return (scale_ui,)


@app.cell
def _(mo):
    field_ui = mo.ui.radio(
        options=["Wa", "Wb", "both"],
        value="both",
        label="field to plot",
    )

    field_ui
    return (field_ui,)


@app.cell
def _(energy_data, field_ui, plt, scale_ui, visibility_ui):
    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    for item in energy_data:
        label = item["file"].split("\\")[-1]

        if label not in visibility_ui.value:
            continue

        t = item["t"]

        # Energy plot (top)
        if field_ui.value in ("Wa", "both"):
            first_value = item["Wa"][0]
            if first_value == 0:
                print(f"Warning: first value is zero for {label} (Wa), skipping normalization")
                W_norm_a = item["Wa"]
            else:
                W_norm_a = item["Wa"] / first_value
            axs[0].plot(t, W_norm_a, label=f"{label} Wa")
        if field_ui.value in ("Wb", "both"):
            first_value_b = item["Wb"][0]
            if first_value_b == 0:
                print(f"Warning: first value is zero for {label} (Wb), skipping normalization")
                W_norm_b = item["Wb"]
            else:
                W_norm_b = item["Wb"] / first_value_b
            axs[0].plot(t, W_norm_b, label=f"{label} Wb", linestyle="--")

        # Central amplitude plot (bottom)
        if field_ui.value in ("Wa", "both"):
            _a_center = item["a_center"]
            _first_a = _a_center[0]
            if _first_a == 0:
                _a_norm = _a_center
            else:
                _a_norm = _a_center / _first_a
            axs[1].plot(t, _a_norm, label=f"{label} |a| center")
        if field_ui.value in ("Wb", "both"):
            _b_center = item["b_center"]
            _first_b = _b_center[0]
            if _first_b == 0:
                _b_norm = _b_center
            else:
                _b_norm = _b_center / _first_b
            axs[1].plot(t, _b_norm, label=f"{label} |b| center", linestyle="--")

    if scale_ui.value == "log":
        axs[0].set_yscale("log")
        axs[1].set_yscale("log")

    axs[1].set_xlabel("t")
    axs[0].set_ylabel("Energy (normalized)")
    axs[1].set_ylabel("Central amplitude (norm)")

    axs[0].legend()
    axs[1].legend()
    plt.tight_layout()
    return (fig,)


@app.cell
def _(fig):
    fig
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

import marimo

__generated_with = "0.23.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt

    from pathlib import Path

    from pde_analysis.load_h5_solver_file import load_h5_solver_file
    from pde_analysis.analysis import compute_energy

    return Path, compute_energy, load_h5_solver_file, mo, plt


@app.cell
def _(mo):
    folder_ui = mo.ui.text(
        value=r"C:\YandexDisk\Ioffe\workspace\one_decay\2D\data\sol\comparsion_gauss",
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
def _(compute_energy, field_ui, files_checkbox, load_h5_solver_file):
    energy_data = []

    for filepath in files_checkbox.value:
        data_obj = load_h5_solver_file(filepath)
        energy = compute_energy(data_obj, field_ui.value)

        energy_data.append(
            {
                "file": filepath,
                "t": energy["t"],
                "W": energy["W"],
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
        options=["a", "b"],
        value="a",
        label="field",
    )

    field_ui
    return (field_ui,)


@app.cell
def _(energy_data, field_ui, plt, scale_ui, visibility_ui):
    fig = plt.figure()

    for item in energy_data:
        label = item["file"].split("\\")[-1]

        if label not in visibility_ui.value:
            continue

        # Нормировка на первое значение
        first_value = item["W"][0]
        # Проверка на ноль, чтобы избежать ошибки
        if first_value == 0:
            print(f"Warning: first value is zero for {label}, skipping normalization (using original)")
            W_norm = item["W"]
        else:
            W_norm = item["W"] / first_value

        plt.plot(item["t"], W_norm, label=label)

    if scale_ui.value == "log":
        plt.yscale("log")

    plt.xlabel("t")
    plt.ylabel(f"W{field_ui.value} (normalized to first point)")
    plt.legend()
    plt.gca()  
    return


if __name__ == "__main__":
    app.run()

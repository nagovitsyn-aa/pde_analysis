import marimo

__generated_with = "0.22.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt

    from pathlib import Path

    from pde_analysis.load_h5_solver_file import load_h5_solver_file, load_h5_solver_file_with_timederivative
    from pde_analysis.residuals import compute_residuals, compute_residuals_from_grid

    return (
        Path,
        compute_residuals,
        load_h5_solver_file_with_timederivative,
        mo,
        plt,
    )


@app.cell
def _(mo):
    folder_ui = mo.ui.text(
        value=r"C:\YandexDisk\Ioffe\workspace\one_decay\2D\data\sol\comparsion",
        label="data folder",
        full_width=True,
    )

    folder_ui
    return (folder_ui,)


@app.cell
def _(Path, folder_ui):
    folder_path = Path(folder_ui.value)

    files = sorted([str(p) for p in folder_path.glob("*.h5")])
    files
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


@app.cell(disabled=True, hide_code=True)
def _(mo):
    mo.md(r"""
    residual_data = []

    for filepath in files_checkbox.value:
        data_obj = load_h5_solver_file(filepath)

        t, err1, err2, err3, err4 = compute_residuals_from_grid(data_obj)

        residual_data.append(
            {
                "file": filepath,
                "t": t,
                "err1": err1,
                "err2": err2,
                "err3": err3,
                "err4": err4,
            }
        )

    residual_data
    """)
    return


@app.cell
def _(
    compute_residuals,
    files_checkbox,
    load_h5_solver_file_with_timederivative,
):
    residual_data = []

    for filepath in files_checkbox.value:
        data_obj = load_h5_solver_file_with_timederivative(filepath)

        t, err1, err2, err3, err4 = compute_residuals(data_obj)

        residual_data.append(
            {
                "file": filepath,
                "t": t,
                "err1": err1,
                "err2": err2,
                "err3": err3,
                "err4": err4,
            }
        )

    residual_data
    return (residual_data,)


@app.cell
def _(mo, residual_data):
    labels = [item["file"].split("\\")[-1] for item in residual_data]

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
def _(plt, residual_data, scale_ui, visibility_ui):
    import numpy as np

    plt.figure()

    linestyles = [
        "-", "--", "-.", ":",
        (0, (1, 1)),
        (0, (3, 1, 1, 1)),
        (0, (5, 2)),
        (0, (2, 2, 5, 2))
    ]

    for i, item in enumerate(residual_data):
        label = item["file"].split("\\")[-1]

        if label not in visibility_ui.value:
            continue

        err_total = np.sqrt(
            item["err1"]**2 +
            item["err2"]**2 +
            item["err3"]**2 +
            item["err4"]**2
        )

        linestyle = linestyles[i % len(linestyles)]

        plt.plot(
            item["t"],
            err_total,
            label=label,
            linestyle=linestyle,
            alpha=0.8,   # <-- прозрачность
            linewidth=2
        )

    if scale_ui.value == "log":
        plt.yscale("log")

    plt.xlabel("t")
    plt.ylabel("total residual")
    plt.grid()
    plt.legend()

    plt.gca()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

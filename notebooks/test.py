import marimo

__generated_with = "0.22.0"
app = marimo.App()


@app.cell
def _():
    import pde_analysis

    return (pde_analysis,)


@app.cell
def _(pde_analysis):
    print("Module loaded:", pde_analysis)
    return


@app.cell
def _():
    file = r"C:\YandexDisk\Ioffe\workspace\one_decay\2D\data\sol\comparsion\RK_Sun-W14_dx=0.02_Λ=0p1_u=1p.h5"
    return (file,)


@app.cell
def _(file):
    with open(file, "rb") as f:
        print(f.read(1000))
    return


@app.cell
def _():
    from pde_analysis.load_h5_solver_file import load_h5_solver_file
    from pde_analysis.print_summary import print_summary

    return load_h5_solver_file, print_summary


@app.cell
def _(file, load_h5_solver_file):
    data = load_h5_solver_file(file)
    return (data,)


@app.cell
def _(data, print_summary):
    print_summary(data)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

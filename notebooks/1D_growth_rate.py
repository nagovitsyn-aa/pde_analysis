import marimo

__generated_with = "0.23.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import re

    return mo, np, plt, re


@app.cell
def _(np):
    incrDataDispRel = np.array([
        [0.1,1.947653543147013],[0.3,1.7696332562665407],[0.5,1.6007014838169693],
        [0.7,1.456112583735515],[0.9,1.3337523787040317],[1.1,1.2296783409467655],
        [1.3,1.1403754753220876],[1.5,1.0630322982681704],[1.7,0.9954538675820535],
        [1.9,0.9359281627588089],[2.1,0.8831105220747468],[2.3,0.8359345846317141],
        [2.5,0.7935459123739559],[2.7,0.7552528640930768],[2.9,0.7204901179474229],
        [3.1,0.6887913780832551],[3.3,0.6597687596316428],[3.5,0.6330970613640707],
        [3.7,0.6085016465759362],[3.9,0.5857490132783538],[4.1,0.564639388517043],
        [4.3,0.5450008609520227],[4.5,0.5266846934398998],[4.7,0.5095615489385183],
        [4.9,0.49351842937077545],[5.1,0.4784561755526281]
    ])

    incrDataStepPump = np.array([
        [0.001,1.99],[0.2,1.9054325514345585],[0.4,1.718679339261633],
        [0.6,1.5549246790949707],[0.8,1.4165136779308098],[1.0,1.2995172548521543],
        [1.2,1.1999062403591658],[1.4,1.1141280245321938],[1.6,1.0396967553428207],
        [1.8,0.9744050275132695],[2.0,0.9168130216908025],[2.5,0.7985188012310396],
        [3.0,0.7069988553439283],[3.5,0.6340002402401451],[4.0,0.5744454696457597],
        [4.5,0.5248936861851411],[5.0,0.48300144463895084]
    ])

    incrDataExp = np.array([
        [0.01,1.9928050716967245],[0.1,1.8161333200993417],[0.2,1.6849609053565109],
        [0.3,1.5799951885209726],[0.4,1.4911154516411378],[0.5,1.413789849033274],
        [0.6,1.345388182960182],[0.7,1.2841702745086352],[0.8,1.22889470081888],
        [0.9,1.178629086948092],[1.0,1.132649643809105],[1.1,1.090379883854399],
        [1.2,1.0513513768786171],[1.3,1.0151789562253077],[1.4,0.9815368066318896],
        [1.5,0.9501532130414946],[1.6,0.9207952676389994],[1.7,0.8932490300534381],
        [1.8,0.8673683523181209],[1.9,0.8429883944856915],[2.0,0.8199766323645665],
        [2.5,0.7218010496439926],[3.0,0.6448274730878004],[3.5,0.5826444868025907],
        [4.0,0.5313354941995087],[4.5,0.48820702277786343],[5.0,0.4514127004981953]
    ])

    incrDataDispRel, incrDataStepPump, incrDataExp
    return incrDataDispRel, incrDataExp, incrDataStepPump


@app.cell
def _(mo):
    lambda_range_ui = mo.ui.range_slider(
        start=0.0,
        stop=10.0,
        step=0.1,
        value=(0.0, 5.0),
        label="Lambda range",
    )

    scale_ui = mo.ui.radio(
        options=["linear", "log"],
        value="linear",
        label="y-scale",
    )

    legend_position_ui = mo.ui.radio(
        options=["inside", "right"],
        value="right",
        label="legend position",
    )

    approx_ui = mo.ui.text_area(
        value="1/(1+2*x/np.pi)",
        label="approx formulas (use x, numpy; one per line)",
        full_width=True,
    )

    lambda_range_ui, scale_ui, legend_position_ui, approx_ui
    return approx_ui, lambda_range_ui, legend_position_ui, scale_ui


@app.cell
def _(mo):
    visible_ui = mo.ui.multiselect(
        options=["exp", "step", "disp", "approx"],
        value=["exp", "step", "disp", "approx"],
        label="visible curves",
    )

    visible_ui
    return (visible_ui,)


@app.cell
def _(re):
    def to_latex(expr: str):
        s = expr

        s = s.replace("np.pi", r"\pi")
        s = s.replace("x", r"\Lambda")
        s = re.sub(r"\*\*", r"^", s)
        s = re.sub(r"(\w+)\^(\d+)", r"{\1}^{\2}", s)
        s = s.replace("*", " ")

        # простая дробь
        if "/" in s:
            parts = s.split("/")
            if len(parts) == 2:
                s = r"\frac{" + parts[0] + "}{" + parts[1] + "}"

        return "$" + s + "$"

    return (to_latex,)


@app.cell
def _(
    approx_ui,
    incrDataDispRel,
    incrDataExp,
    incrDataStepPump,
    lambda_range_ui,
    legend_position_ui,
    np,
    plt,
    scale_ui,
    to_latex,
    visible_ui,
):
    fig, ax = plt.subplots()

    lam_min, lam_max = lambda_range_ui.value
    x = np.linspace(lam_min, lam_max, 400)

    def filter_data(data):
        mask = (data[:,0] >= lam_min) & (data[:,0] <= lam_max)
        return data[mask]

    if "exp" in visible_ui.value:
        d = filter_data(incrDataExp)
        ax.plot(d[:,0], 0.5*d[:,1], linewidth=3, label="exp")

    if "step" in visible_ui.value:
        d = filter_data(incrDataStepPump)
        ax.plot(d[:,0], 0.5*d[:,1], linewidth=3, alpha=0.7, label="step")

    if "disp" in visible_ui.value:
        d = filter_data(incrDataDispRel)
        ax.plot(d[:,0], 0.5*d[:,1], linestyle="--", linewidth=3, label="disp rel")

    if "approx" in visible_ui.value:
        formulas = [line.strip() for line in approx_ui.value.splitlines() if line.strip()]

        for f in formulas:
            try:
                y = eval(f, {"np": np, "x": x})
                ax.plot(
                    x,
                    y,
                    linestyle=":",
                    linewidth=3,
                    label=to_latex(f),
                )
            except Exception:
                pass

    if scale_ui.value == "log":
        ax.set_yscale("log")

    ax.set_xlim(lam_min, lam_max)
    ax.set_xlabel("Lambda")
    ax.set_ylabel("gamma/2")
    ax.grid()

    if legend_position_ui.value == "inside":
        ax.legend()
    else:
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
        fig.tight_layout()

    fig
    return


@app.cell
def _(mo):
    fit_model_ui = mo.ui.radio(
        options=["a/(1+b*x)", "a/(1+b*x)^c"],
        value="a/(1+b*x)",
        label="fit model",
    )

    fit_model_ui
    return (fit_model_ui,)


@app.cell
def _(fit_model_ui, incrDataExp, np):
    x_data = incrDataExp[:, 0]
    y_data = incrDataExp[:, 1]

    if fit_model_ui.value == "a/(1+b*x)":
        # линейная регрессия для 1/y = (1/a) + (b/a)x
        Y = 1.0 / y_data
        A = np.vstack([np.ones_like(x_data), x_data]).T
        coeffs, *_ = np.linalg.lstsq(A, Y, rcond=None)

        A0, B0 = coeffs
        a = 1.0 / A0
        b = B0 / A0

        def fit_func(x):
            return a / (1.0 + b * x)

        params = {"a": a, "b": b}

    else:
        # грубый grid search для a/(1+b*x)^c
        a_vals = np.linspace(1.5, 2.5, 40)
        b_vals = np.linspace(0.1, 2.0, 40)
        c_vals = np.linspace(0.5, 2.0, 30)

        best_err = np.inf
        best_params = None

        for a in a_vals:
            for b in b_vals:
                for c in c_vals:
                    y_pred = a / (1.0 + b * x_data) ** c
                    err = np.mean((y_pred - y_data) ** 2)

                    if err < best_err:
                        best_err = err
                        best_params = (a, b, c)

        a, b, c = best_params

        def fit_func(x):
            return a / (1.0 + b * x) ** c

        params = {"a": a, "b": b, "c": c}

    fit_func, params
    return fit_func, params


@app.cell
def _(
    approx_ui,
    fit_func,
    incrDataExp,
    lambda_range_ui,
    legend_position_ui,
    np,
    params,
    plt,
    scale_ui,
    to_latex,
):
    def _():
        fig, ax = plt.subplots()

        lam_min, lam_max = lambda_range_ui.value
        x = np.linspace(lam_min, lam_max, 400)

        # данные
        mask = (incrDataExp[:,0] >= lam_min) & (incrDataExp[:,0] <= lam_max)
        d = incrDataExp[mask]

        ax.plot(
            d[:,0],
            0.5*d[:,1],
            linewidth=3,
            label="data (incrDataExp)",
        )

        # базовая аппроксимация (первая строка textarea)
        base_formula = approx_ui.value.splitlines()[0].strip()
        try:
            y_base = eval(base_formula, {"np": np, "x": x})
            ax.plot(
                x,
                y_base,
                linestyle="--",
                linewidth=3,
                label="base " + to_latex(base_formula),
            )
        except Exception:
            pass

        # fitted
        y_fit = fit_func(x)
        param_str = ", ".join([f"{k}={v:.3f}" for k,v in params.items()])

        ax.plot(
            x,
            0.5*y_fit,
            linestyle="-.",
            linewidth=3,
            label=f"fit ({param_str})",
        )

        if scale_ui.value == "log":
            ax.set_yscale("log")

        ax.set_xlim(lam_min, lam_max)
        ax.set_xlabel("Lambda")
        ax.set_ylabel("gamma")
        ax.grid()

        if legend_position_ui.value == "inside":
            ax.legend()
        else:
            ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
            fig.tight_layout()
        return fig


    _()
    return


@app.cell
def _():
    1.861156653468705/2
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

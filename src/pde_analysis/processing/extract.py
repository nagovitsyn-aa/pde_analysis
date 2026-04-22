import h5py
import numpy as np
from pde_analysis.load_h5_solver_file import _read_array, _read_value, _read_string, _read_string_array


def extract_params(h5_path):
    with h5py.File(h5_path, "r") as f:

        # --- Parameters ---
        params = {}
        for key in f["/Parameters"]:
            ds = f[f"/Parameters/{key}"]
            params[key] = _read_value(ds)

        # --- Step ---
        step = {
            "dx": float(f["/StepSize/dx"][()]),
            "dy": float(f["/StepSize/dy"][()])
        }

        return {
            "Parameters": params,
            "Step": step
        }

def extract_metadata(h5_path):
    with h5py.File(h5_path, "r") as f:

        metadata = {}

        metadata["CalculationStatus"] = _read_string(f["/Metadata/CalculationStatus"])
        metadata["Timing"] = float(f["/Metadata/Timing"][()])

        method_ds = f["/Metadata/Method"]

        if method_ds.shape == ():
            metadata["Method"] = _read_string(method_ds)
        else:
            metadata["Method"] = _read_string_array(method_ds)

        return metadata
    

def extract_timeseries(h5_path):
    with h5py.File(h5_path, "r") as f:

        from scipy.integrate import trapezoid

        # Чтение данных
        aRe = f["/Solution/aRe"][()]
        aIm = f["/Solution/aIm"][()]
        bRe = f["/Solution/bRe"][()]
        bIm = f["/Solution/bIm"][()]

        t = _read_array(f["/Grid/t"])   
        x = _read_array(f["/Grid/x"])   
        y = _read_array(f["/Grid/y"])   

        dx = float(f["/StepSize/dx"][()])
        dy = float(f["/StepSize/dy"][()])

        # |a|^2 и |b|^2
        a2 = aRe**2 + aIm**2
        b2 = bRe**2 + bIm**2


        if a2.shape[-1] == len(x):
            axis_x = -1
        elif a2.shape[-2] == len(x):
            axis_x = -2
        else:
            raise ValueError("Не удалось определить ось x в массиве a2")

        # Интегрируем по x
        int_x_a = trapezoid(a2, dx=dx, axis=axis_x)
        int_x_b = trapezoid(b2, dx=dx, axis=axis_x)

        # Интегрируем по y (после удаления оси x оставшаяся пространственная ось становится последней)
        Ea = trapezoid(int_x_a, dx=dy, axis=-1)
        Eb = trapezoid(int_x_b, dx=dy, axis=-1)


        # --- максимум ---
        max_a = np.max(np.sqrt(a2), axis=(1, 2))
        max_b = np.max(np.sqrt(b2), axis=(1, 2))

        # Нормировка на первое значение
        Ea = Ea / Ea[0]
        Eb = Eb / Eb[0]
        max_a = max_a / max_a[0]
        max_b = max_b / max_b[0]

        return {
            "E_a": (t, Ea),
            "E_b": (t, Eb),
            "max_a": (t, max_a),
            "max_b": (t, max_b)
        }
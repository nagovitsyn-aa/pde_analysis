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
    """Извлекает из .h5 временные ряды в сыром виде (без нормировки).

    Возвращает словарь:
      "E_a"       — полная энергия |a|^2, интегрированная по x, y
      "E_b"       — полная энергия |b|^2, интегрированная по x, y
      "max_a"     — максимум |a| по пространству на каждом срезе
      "max_b"     — максимум |b| по пространству на каждом срезе
      "a_center"  — |a| в центре пучка (ближайшая к (0,0) точка сетки)
      "b_center"  — |b| в центре пучка (ближайшая к (0,0) точка сетки)
    """
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

        # Определяем, по какой оси x
        if a2.shape[-1] == len(x):
            axis_x = -1
        elif a2.shape[-2] == len(x):
            axis_x = -2
        else:
            raise ValueError("Не удалось определить ось x в массиве a2: "
                             f"a2.shape={a2.shape}, len(x)={len(x)}")

        # --- Полная энергия (сырая, без нормировки) ---
        # Интегрируем по x
        int_x_a = trapezoid(a2, dx=dx, axis=axis_x)
        int_x_b = trapezoid(b2, dx=dx, axis=axis_x)

        # Интегрируем по y (после удаления оси x оставшаяся пространственная ось — последняя)
        E_a = trapezoid(int_x_a, dx=dy, axis=-1)
        E_b = trapezoid(int_x_b, dx=dy, axis=-1)

        # --- Максимум |a|, |b| по пространству (сырой) ---
        max_a = np.max(np.sqrt(a2), axis=(1, 2))
        max_b = np.max(np.sqrt(b2), axis=(1, 2))

        # --- Амплитуда в центре пучка ---
        # Находим индекс сетки, ближайший к (0, 0)
        ix_center = int(np.argmin(np.abs(x)))
        iy_center = int(np.argmin(np.abs(y)))

        a_center = np.sqrt(
            aRe[:, ix_center, iy_center]**2 +
            aIm[:, ix_center, iy_center]**2
        )
        b_center = np.sqrt(
            bRe[:, ix_center, iy_center]**2 +
            bIm[:, ix_center, iy_center]**2
        )

        return {
            "E_a":        (t, E_a),
            "E_b":        (t, E_b),
            "max_a":      (t, max_a),
            "max_b":      (t, max_b),
            "a_center":   (t, a_center),
            "b_center":   (t, b_center),
        }

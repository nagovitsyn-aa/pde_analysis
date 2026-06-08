import h5py
import numpy as np


def _read_array(ds):
    return np.array(ds)


def _read_scalar(ds):
    arr = np.array(ds)
    if arr.size != 1:
        raise ValueError(f"Expected scalar, got shape {arr.shape}")
    return arr.item()


def _read_string(ds):
    val = ds[()]
    if isinstance(val, bytes):
        return val.decode("utf-8")
    return str(val)


def _read_string_array(ds):
    arr = ds[:]
    return [v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in arr]


def _read_value(ds):
    """Универсальный reader для параметров"""
    val = ds[()]

    if isinstance(val, bytes):
        return val.decode("utf-8")

    if isinstance(val, np.ndarray):
        if val.size == 1:
            return val.item()
        return val

    return val


def load_h5_solver_file(filepath):
    data = {}

    with h5py.File(filepath, "r") as f:

        # --- Grid ---
        data["grid"] = {
            "t": _read_array(f["/Grid/t"]),
            "x": _read_array(f["/Grid/x"]),
            "y": _read_array(f["/Grid/y"]),
        }

        # --- Solution ---
        data["solution"] = {
            "aRe": _read_array(f["/Solution/aRe"]),
            "aIm": _read_array(f["/Solution/aIm"]),
            "bRe": _read_array(f["/Solution/bRe"]),
            "bIm": _read_array(f["/Solution/bIm"]),
        }

        # --- Step size ---
        data["step"] = {
            "dx": _read_scalar(f["/StepSize/dx"]),
            "dy": _read_scalar(f["/StepSize/dy"]),
        }

        # --- Metadata ---
        metadata = {}

        metadata["CalculationStatus"] = _read_string(f["/Metadata/CalculationStatus"])
        metadata["Timing"] = _read_scalar(f["/Metadata/Timing"])

        method_ds = f["/Metadata/Method"]

        # scalar string или массив строк
        if method_ds.shape == ():
            metadata["Method"] = _read_string(method_ds)
        else:
            metadata["Method"] = _read_string_array(method_ds)

        data["metadata"] = metadata

        # --- Parameters ---
        params = {}

        for key in f["/Parameters"]:
            ds = f[f"/Parameters/{key}"]
            params[key] = _read_value(ds)

        data["parameters"] = params

    return data


def load_h5_solver_file_with_timederivative(filepath):
    data = {}

    with h5py.File(filepath, "r") as f:

        # --- Grid ---
        data["grid"] = {
            "t": _read_array(f["/Grid/t"]),
            "x": _read_array(f["/Grid/x"]),
            "y": _read_array(f["/Grid/y"]),
        }

        # --- Solution ---
        data["solution"] = {
            "aRe": _read_array(f["/Solution/aRe"]),
            "aIm": _read_array(f["/Solution/aIm"]),
            "bRe": _read_array(f["/Solution/bRe"]),
            "bIm": _read_array(f["/Solution/bIm"]),
        }

        # --- TimeDerivative ---
        data["timeDerivative"] = {
            "aRe": _read_array(f["/TimeDerivative/aRe"]),
            "aIm": _read_array(f["/TimeDerivative/aIm"]),
            "bRe": _read_array(f["/TimeDerivative/bRe"]),
            "bIm": _read_array(f["/TimeDerivative/bIm"]),
        }

        # --- Step size ---
        data["step"] = {
            "dx": _read_scalar(f["/StepSize/dx"]),
            "dy": _read_scalar(f["/StepSize/dy"]),
        }

        # --- Metadata ---
        metadata = {}

        metadata["CalculationStatus"] = _read_string(f["/Metadata/CalculationStatus"])
        metadata["Timing"] = _read_scalar(f["/Metadata/Timing"])

        method_ds = f["/Metadata/Method"]

        # scalar string или массив строк
        if method_ds.shape == ():
            metadata["Method"] = _read_string(method_ds)
        else:
            metadata["Method"] = _read_string_array(method_ds)

        data["metadata"] = metadata

        # --- Parameters ---
        params = {}

        for key in f["/Parameters"]:
            ds = f[f"/Parameters/{key}"]
            params[key] = _read_value(ds)

        data["parameters"] = params

    return data


def load_h5_solver_file_1d(filepath):
    data = {}

    with h5py.File(filepath, "r") as f:

        # --- Grid ---
        data["grid"] = {
            "t": _read_array(f["/Grid/t"]),
            "x": _read_array(f["/Grid/x"]),
        }

        # --- Solution ---
        data["solution"] = {
            "aRe": np.squeeze(_read_array(f["/Solution/aRe"])),
            "aIm": np.squeeze(_read_array(f["/Solution/aIm"])),
            "bRe": np.squeeze(_read_array(f["/Solution/bRe"])),
            "bIm": np.squeeze(_read_array(f["/Solution/bIm"])),
        }

        # --- Step size ---
        data["step"] = {
            "dx": _read_scalar(f["/StepSize/dx"]),
        }

        # --- Metadata ---
        metadata = {}

        metadata["CalculationStatus"] = _read_string(f["/Metadata/CalculationStatus"])
        metadata["Timing"] = _read_scalar(f["/Metadata/Timing"])

        method_ds = f["/Metadata/Method"]

        # scalar string или массив строк
        if method_ds.shape == ():
            metadata["Method"] = _read_string(method_ds)
        else:
            metadata["Method"] = _read_string_array(method_ds)

        data["metadata"] = metadata

        # --- Parameters ---
        params = {}

        for key in f["/Parameters"]:
            ds = f[f"/Parameters/{key}"]
            params[key] = _read_value(ds)

        data["parameters"] = params

    return data


def load_h5_solver_file_1d_with_timederivative(filepath):
    data = {}

    with h5py.File(filepath, "r") as f:

        # --- Grid ---
        data["grid"] = {
            "t": _read_array(f["/Grid/t"]),
            "x": _read_array(f["/Grid/x"]),
        }

        # --- Solution ---
        data["solution"] = {
            "aRe": np.squeeze(_read_array(f["/Solution/aRe"])),
            "aIm": np.squeeze(_read_array(f["/Solution/aIm"])),
            "bRe": np.squeeze(_read_array(f["/Solution/bRe"])),
            "bIm": np.squeeze(_read_array(f["/Solution/bIm"])),
        }

        # --- TimeDerivative ---
        data["timeDerivative"] = {
            "aRe": np.squeeze(_read_array(f["/TimeDerivative/aRe"])),
            "aIm": np.squeeze(_read_array(f["/TimeDerivative/aIm"])),
            "bRe": np.squeeze(_read_array(f["/TimeDerivative/bRe"])),
            "bIm": np.squeeze(_read_array(f["/TimeDerivative/bIm"])),
        }

        # --- Step size ---
        data["step"] = {
            "dx": _read_scalar(f["/StepSize/dx"]),
        }

        # --- Metadata ---
        metadata = {}

        metadata["CalculationStatus"] = _read_string(f["/Metadata/CalculationStatus"])
        metadata["Timing"] = _read_scalar(f["/Metadata/Timing"])

        method_ds = f["/Metadata/Method"]

        # scalar string или массив строк
        if method_ds.shape == ():
            metadata["Method"] = _read_string(method_ds)
        else:
            metadata["Method"] = _read_string_array(method_ds)

        data["metadata"] = metadata

        # --- Parameters ---
        params = {}

        for key in f["/Parameters"]:
            ds = f[f"/Parameters/{key}"]
            params[key] = _read_value(ds)

        data["parameters"] = params

    return data
import numpy as np


def compute_abs(re, im):
    return np.sqrt(re**2 + im**2)


def get_profile_at_tz(data, field, t_index=0, z_index=0):
    """
    Возвращает профиль по x для фиксированных t и z.

    field: "a" или "b"
    """

    if field == "a":
        re = data["solution"]["aRe"]
        im = data["solution"]["aIm"]
    elif field == "b":
        re = data["solution"]["bRe"]
        im = data["solution"]["bIm"]
    else:
        raise ValueError("field must be 'a' or 'b'")

    abs_val = compute_abs(re, im)

    # предполагаем (t, x, y)
    profile = abs_val[t_index, :, z_index]

    x = data["grid"]["x"]
    t = data["grid"]["t"]
    z = data["grid"]["y"]  # переименование на уровне анализа

    return {
        "x": x,
        "profile": profile,
        "t": t[t_index],
        "z": z[z_index],
    }


def crop_x(x, y, xmin, xmax):
    mask = (x >= xmin) & (x <= xmax)
    return x[mask], y[mask]


def find_nearest_index(arr, value):
    return int(np.argmin(np.abs(arr - value)))


from scipy.integrate import simpson


def compute_energy(data, field, t_min=None, t_max=None):
    """
    field: "a" или "b"
    """

    if field == "a":
        re = data["solution"]["aRe"]
        im = data["solution"]["aIm"]
    elif field == "b":
        re = data["solution"]["bRe"]
        im = data["solution"]["bIm"]
    else:
        raise ValueError("field must be 'a' or 'b'")

    abs_sq = re**2 + im**2

    x = data["grid"]["x"]
    z = data["grid"]["y"]
    t = data["grid"]["t"]

    int_x = simpson(abs_sq, x, axis=1)
    energy = simpson(int_x, z, axis=1)

    # фильтрация по времени
    if t_min is not None or t_max is not None:
        mask = np.ones_like(t, dtype=bool)

        if t_min is not None:
            mask &= t >= t_min
        if t_max is not None:
            mask &= t <= t_max

        t = t[mask]
        energy = energy[mask]

    return {"t": t, "W": energy}


def compute_energy_1d(data, field, t_min=None, t_max=None):
    """
    Compute 1D energy from a 1D solution file.

    field: "a" or "b"
    """

    if field == "a":
        re = data["solution"]["aRe"]
        im = data["solution"]["aIm"]
    elif field == "b":
        re = data["solution"]["bRe"]
        im = data["solution"]["bIm"]
    else:
        raise ValueError("field must be 'a' or 'b'")

    re = np.asarray(re)
    im = np.asarray(im)
    x = np.asarray(data["grid"]["x"]).ravel()
    t = np.asarray(data["grid"]["t"]).ravel()

    abs_sq = re**2 + im**2
    if abs_sq.ndim != 2:
        raise ValueError(f"Expected 2D data for 1D energy, got shape {abs_sq.shape}")

    energy = simpson(abs_sq, x, axis=1)

    if t_min is not None or t_max is not None:
        mask = np.ones(t.shape, dtype=bool)
        if t_min is not None:
            mask &= t >= t_min
        if t_max is not None:
            mask &= t <= t_max

        t = t[mask]
        energy = energy[mask]

    return {"t": t, "W": energy}


import numpy as np


def compute_abs_field(data, field, t_index):
    if field == "a":
        re = data["solution"]["aRe"][t_index]
        im = data["solution"]["aIm"][t_index]
    elif field == "b":
        re = data["solution"]["bRe"][t_index]
        im = data["solution"]["bIm"][t_index]
    else:
        raise ValueError

    return np.sqrt(re**2 + im**2)


def get_2d_field(data, field, t_index):
    field_abs = compute_abs_field(data, field, t_index)

    x = data["grid"]["x"]
    z = data["grid"]["y"]

    return x, z, field_abs


def get_xt_evolution(data, field, z_index):
    if field == "a":
        re = data["solution"]["aRe"][:, :, z_index]
        im = data["solution"]["aIm"][:, :, z_index]
    elif field == "b":
        re = data["solution"]["bRe"][:, :, z_index]
        im = data["solution"]["bIm"][:, :, z_index]
    else:
        raise ValueError

    A = np.sqrt(re**2 + im**2)

    x = data["grid"]["x"]
    t = data["grid"]["t"]

    return x, t, A


def prepare_surface_xz(data, field, t_index):
    abs_field = compute_abs_field(data, field, t_index)

    x = data["grid"]["x"]
    z = data["grid"]["y"]

    X, Z = np.meshgrid(x, z, indexing="ij")

    return X, Z, abs_field


def prepare_heatmap(data, field, t_index):
    abs_field = compute_abs_field(data, field, t_index)

    x = data["grid"]["x"]
    z = data["grid"]["y"]

    extent = [x[0], x[-1], z[0], z[-1]]

    return abs_field.T, extent


def prepare_xt_surface(data, field, z_index):
    if field == "a":
        re = data["solution"]["aRe"][:, :, z_index]
        im = data["solution"]["aIm"][:, :, z_index]
    elif field == "b":
        re = data["solution"]["bRe"][:, :, z_index]
        im = data["solution"]["bIm"][:, :, z_index]
    else:
        raise ValueError

    abs_field = np.sqrt(re**2 + im**2)

    x = data["grid"]["x"]
    t = data["grid"]["t"]

    T, X = np.meshgrid(t, x, indexing="ij")

    return X, T, abs_field
import numpy as np


def dt_central(u, t):
    dudt = np.zeros_like(u, dtype=np.float64)
    dudt[1:-1] = (u[2:] - u[:-2]) / (t[2:] - t[:-2])[:, None, None]
    dudt[0] = (u[1] - u[0]) / (t[1] - t[0])
    dudt[-1] = (u[-1] - u[-2]) / (t[-1] - t[-2])
    return dudt




def dx_central(u, dx):
    dudx = np.zeros_like(u)
    dudx[:, 1:-1, :] = (u[:, 2:, :] - u[:, :-2, :]) / (2 * dx)
    dudx[:, 0, :] = (u[:, 1, :] - u[:, 0, :]) / dx
    dudx[:, -1, :] = (u[:, -1, :] - u[:, -2, :]) / dx
    return dudx


def dx_4th(u, dx):
    dudx = np.zeros_like(u)

    # центральная 4-го порядка
    dudx[:, 2:-2, :] = (
        -u[:, 4:, :] + 8*u[:, 3:-1, :] - 8*u[:, 1:-3, :] + u[:, :-4, :]
    ) / (12 * dx)

    # fallback 2-го порядка (границы)
    dudx[:, 1, :] = (u[:, 2, :] - u[:, 0, :]) / (2 * dx)
    dudx[:, -2, :] = (u[:, -1, :] - u[:, -3, :]) / (2 * dx)

    # край
    dudx[:, 0, :] = (u[:, 1, :] - u[:, 0, :]) / dx
    dudx[:, -1, :] = (u[:, -1, :] - u[:, -2, :]) / dx

    return dudx


def d2y_central(u, dy):
    d2udy2 = np.zeros_like(u)
    d2udy2[:, :, 1:-1] = (
        u[:, :, 2:] - 2 * u[:, :, 1:-1] + u[:, :, :-2]
    ) / (dy**2)

    d2udy2[:, :, 0] = d2udy2[:, :, 1]
    d2udy2[:, :, -1] = d2udy2[:, :, -2]
    return d2udy2

def d2y_4th(u, dy):
    d2udy2 = np.zeros_like(u)

    # центральная 4-го порядка
    d2udy2[:, :, 2:-2] = (
        -u[:, :, 4:]
        + 16*u[:, :, 3:-1]
        - 30*u[:, :, 2:-2]
        + 16*u[:, :, 1:-3]
        - u[:, :, :-4]
    ) / (12 * dy**2)

    # fallback 2-го порядка
    d2udy2[:, :, 1] = (
        u[:, :, 2] - 2*u[:, :, 1] + u[:, :, 0]
    ) / (dy**2)

    d2udy2[:, :, -2] = (
        u[:, :, -1] - 2*u[:, :, -2] + u[:, :, -3]
    ) / (dy**2)

    # край
    d2udy2[:, :, 0] = d2udy2[:, :, 1]
    d2udy2[:, :, -1] = d2udy2[:, :, -2]

    return d2udy2


# --- main ---
def compute_residuals(data):
    aRe = data["solution"]["aRe"]
    aIm = data["solution"]["aIm"]
    bRe = data["solution"]["bRe"]
    bIm = data["solution"]["bIm"]

    t = data["grid"]["t"]
    x = data["grid"]["x"]
    y = data["grid"]["y"]

    dx = data["step"]["dx"]
    dy = data["step"]["dy"]

    u = data["parameters"]["u"]
    Lambda = data["parameters"]["Lambda"]
    nu0 = data["parameters"]["nu0"]

    # --- производные ---
    dtaRe = data["timeDerivative"]["aRe"]
    dtaIm = data["timeDerivative"]["aIm"]
    dtbRe = data["timeDerivative"]["bRe"]
    dtbIm = data["timeDerivative"]["bIm"]

    dxaRe = dx_4th(aRe, dx)
    dxaIm = dx_4th(aIm, dx)
    dxbRe = dx_4th(bRe, dx)
    dxbIm = dx_4th(bIm, dx)

    d2yaRe = d2y_4th(aRe, dy)
    d2yaIm = d2y_4th(aIm, dy)
    d2ybRe = d2y_4th(bRe, dy)
    d2ybIm = d2y_4th(bIm, dy)
    
    print(aRe.shape, dtaRe.shape)
    
    # --- phase ---
    phi = x**2 / 2
    cos_phi = np.cos(phi)[None, :, None]
    sin_phi = np.sin(phi)[None, :, None]

    exp_y = np.exp(-y**2 / 2)[None, None, :]

    # --- RHS coupling ---
    rhs_aRe = nu0 * exp_y * (bRe * cos_phi - bIm * sin_phi)
    rhs_aIm = nu0 * exp_y * (bRe * sin_phi + bIm * cos_phi)

    rhs_bRe = nu0 * exp_y * (aRe * cos_phi + aIm * sin_phi)
    rhs_bIm = nu0 * exp_y * (-aRe * sin_phi + aIm * cos_phi)

    # --- residuals ---
    eq1 = dtaRe - u * dxaRe - Lambda * d2yaIm - rhs_aRe
    eq2 = dtaIm - u * dxaIm + Lambda * d2yaRe - rhs_aIm

    eq3 = dtbRe + u * dxbRe + Lambda * d2ybIm - rhs_bRe
    eq4 = dtbIm + u * dxbIm - Lambda * d2ybRe - rhs_bIm

    # --- norms ---
    err1 = np.max(np.abs(eq1), axis=(1, 2))
    err2 = np.max(np.abs(eq2), axis=(1, 2))
    err3 = np.max(np.abs(eq3), axis=(1, 2))
    err4 = np.max(np.abs(eq4), axis=(1, 2))

    return t, err1, err2, err3, err4

# --- main ---
def compute_residuals_from_grid(data):
    aRe = data["solution"]["aRe"]
    aIm = data["solution"]["aIm"]
    bRe = data["solution"]["bRe"]
    bIm = data["solution"]["bIm"]

    t = data["grid"]["t"]
    x = data["grid"]["x"]
    y = data["grid"]["y"]

    dx = data["step"]["dx"]
    dy = data["step"]["dy"]

    u = data["parameters"]["u"]
    Lambda = data["parameters"]["Lambda"]
    nu0 = data["parameters"]["nu0"]

    # --- производные ---
    dtaRe = dt_central(aRe, t)
    dtaIm = dt_central(aIm, t)
    dtbRe = dt_central(bRe, t)
    dtbIm = dt_central(bIm, t)

    dxaRe = dx_4th(aRe, dx)
    dxaIm = dx_4th(aIm, dx)
    dxbRe = dx_4th(bRe, dx)
    dxbIm = dx_4th(bIm, dx)

    d2yaRe = d2y_4th(aRe, dy)
    d2yaIm = d2y_4th(aIm, dy)
    d2ybRe = d2y_4th(bRe, dy)
    d2ybIm = d2y_4th(bIm, dy)
    
    print(aRe.shape, dtaRe.shape)
    
    # --- phase ---
    phi = x**2 / 2
    cos_phi = np.cos(phi)[None, :, None]
    sin_phi = np.sin(phi)[None, :, None]

    exp_y = np.exp(-y**2 / 2)[None, None, :]

    # --- RHS coupling ---
    rhs_aRe = nu0 * exp_y * (bRe * cos_phi - bIm * sin_phi)
    rhs_aIm = nu0 * exp_y * (bRe * sin_phi + bIm * cos_phi)

    rhs_bRe = nu0 * exp_y * (aRe * cos_phi + aIm * sin_phi)
    rhs_bIm = nu0 * exp_y * (-aRe * sin_phi + aIm * cos_phi)

    # --- residuals ---
    eq1 = dtaRe - u * dxaRe - Lambda * d2yaIm - rhs_aRe
    eq2 = dtaIm - u * dxaIm + Lambda * d2yaRe - rhs_aIm

    eq3 = dtbRe + u * dxbRe + Lambda * d2ybIm - rhs_bRe
    eq4 = dtbIm + u * dxbIm - Lambda * d2ybRe - rhs_bIm

    # --- norms ---
    err1 = np.max(np.abs(eq1), axis=(1, 2))
    err2 = np.max(np.abs(eq2), axis=(1, 2))
    err3 = np.max(np.abs(eq3), axis=(1, 2))
    err4 = np.max(np.abs(eq4), axis=(1, 2))

    return t, err1, err2, err3, err4
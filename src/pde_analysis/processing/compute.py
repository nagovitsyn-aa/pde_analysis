import numpy as np


def find_longest_exponential_stage(t, y, min_len=5, r2_min=0.99):

    log_y = np.log(y)
    n = len(t)

    px = np.concatenate([[0], np.cumsum(t)])
    py = np.concatenate([[0], np.cumsum(log_y)])
    pxy = np.concatenate([[0], np.cumsum(t * log_y)])
    px2 = np.concatenate([[0], np.cumsum(t * t)])
    py2 = np.concatenate([[0], np.cumsum(log_y * log_y)])

    best = None
    best_length = -1

    for i in range(n - min_len + 1):
        for j in range(i + min_len - 1, n):

            length = j - i + 1

            Sx = px[j+1] - px[i]
            Sy = py[j+1] - py[i]
            Sxy = pxy[j+1] - pxy[i]
            Sx2 = px2[j+1] - px2[i]
            Sy2 = py2[j+1] - py2[i]

            denom = (length * Sx2 - Sx * Sx)
            if denom == 0:
                continue

            slope = (length * Sxy - Sx * Sy) / denom
            if slope <= 0:
                continue

            denom_r = (length * Sx2 - Sx*Sx) * (length * Sy2 - Sy*Sy)
            if denom_r == 0:
                continue

            r2 = (length * Sxy - Sx * Sy)**2 / denom_r

            if r2 < r2_min:
                continue

            if length > best_length:
                best_length = length
                best = {
                    "increment": slope,
                    "t_start": t[i],
                    "t_end": t[j],
                    "length": length,
                    "r2": r2
                }

    return best





def find_interaction_time_full(t, y, window=5, threshold=0.1, tail=3):
    """
    Возвращает конец взаимодействия (включая насыщение)

    tail — сколько подряд точек должны быть ниже порога
    """

    log_y = np.log(y)

    slopes = []
    t_mid = []

    for i in range(len(t) - window):
        dt = t[i+window] - t[i]
        dy = log_y[i+window] - log_y[i]

        slopes.append(dy / dt)
        t_mid.append(0.5 * (t[i+window] + t[i]))

    slopes = np.array(slopes)
    t_mid = np.array(t_mid)

    if len(slopes) == 0:
        return None

    max_slope = np.max(slopes)
    mask = slopes < threshold * max_slope

    # ищем устойчивое "умирание роста"
    for i in range(len(mask) - tail):
        if np.all(mask[i:i+tail]):
            return t_mid[i]

    return t[-1]  # fallback




def compute_metrics(timeseries_dict, params):

    t, Ea = timeseries_dict["E_a"]

    # --- 1. full interaction time ---
    t_full = find_interaction_time_full(t, Ea)

    if t_full is None:
        return None

    mask_full = t <= t_full
    t_cut = t[mask_full]
    y_cut = Ea[mask_full]

    if len(t_cut) < 2:
        return None

    # --- 2. average increment ---
    log_y = np.log(y_cut)

    gr_avg = (log_y[-1] - log_y[0]) / (t_cut[-1] - t_cut[0])
    amplification_pred = np.exp(gr_avg * t_full)

    # --- 3. exponential stage ---
    exp_stage = find_longest_exponential_stage(t, Ea)

    if exp_stage is None:
        gr_stage = None
        t_stage = None
    else:
        gr_stage = exp_stage["increment"]
        t_stage = exp_stage["t_end"] - exp_stage["t_start"]

    # --- 4. real amplification ---
    amplification = np.max(Ea)

    # --- 5. Z ---
    Z = np.log(amplification) / (2 * np.pi)

    u = params["Parameters"].get("u", 1)
    ZtoZ0 = np.log(amplification) / (2 * np.pi * u**(-2))

    return {
        # времена
        "t_interaction_full": float(t_full),
        "t_exponential": None if t_stage is None else float(t_stage),

        # инкременты
        "growth_rate_avg": float(gr_avg),
        "growth_rate_stage": None if gr_stage is None else float(gr_stage),

        "amplification": float(amplification),
        "amplification_pred": float(amplification_pred),

        "Z": float(Z),
        "ZtoZ0": float(ZtoZ0),
    }
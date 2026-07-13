import numpy as np
from scipy.signal import find_peaks


# ==============================================================
# Часть 1: производные временные ряды (вызывается при загрузке)
# ==============================================================

def compute_derived_timeseries(raw_ts):
    """Создаёт нормированные версии временных рядов (если E[0] ≠ 0).

    Принимает словарь raw_ts из extract_timeseries().
    Возвращает dict {имя_ряда: (t, y)} для добавления в БД.

    Сейчас:
      - E_a_norm,     E_b_norm      — E(t) / E(0)
      - max_a_norm,   max_b_norm    — max(t) / max(0)
      - a_center_norm, b_center_norm — a_center(t) / a_center(0)
    """
    derived = {}

    norm_candidates = ["E_a", "E_b", "max_a", "max_b", "a_center", "b_center"]

    for key in norm_candidates:
        if key not in raw_ts:
            continue
        t, y = raw_ts[key]
        y = np.asarray(y, dtype=float)
        y0 = y[0]
        if y0 != 0.0 and np.isfinite(y0):
            derived[f"{key}_norm"] = (t, y / y0)
        # если y0 == 0 — просто пропускаем, нормировка не определена

    return derived


# ==============================================================
# Часть 2: скалярные метрики (НЕ вызываются при загрузке,
#          используются в ноутбуках / ad-hoc анализе)
# ==============================================================

def find_longest_exponential_stage(t, y, min_len=5, r2_min=0.99):

    log_y = np.log(np.asarray(y, dtype=float))
    t = np.asarray(t, dtype=float)
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
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)
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
            return float(t_mid[i])

    return float(t[-1])  # fallback


def compute_amplification_by_max(t, y, *, t_max=None):
    t_arr = np.asarray(t, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    if t_max is not None:
        mask = t_arr <= t_max
        t_arr = t_arr[mask]
        y_arr = y_arr[mask]

    if len(y_arr) == 0:
        return np.nan

    return float(np.nanmax(y_arr))


def compute_amplification_by_median_after_time(t, y, *, t_start):
    t_arr = np.asarray(t, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    mask = t_arr >= t_start
    if np.count_nonzero(mask) == 0:
        return np.nan

    return float(np.nanmedian(y_arr[mask]))


def find_increment_peak(
    t,
    y,
    min_len=3,
    max_len=10,
    t_min=0.0,
    t_max=None,
    peak_number=2,
    half_width=0.5,
):
    t_arr = np.asarray(t, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    mask = (y_arr > 0) & (t_arr >= t_min)
    if t_max is not None:
        mask &= t_arr <= t_max

    t_arr = t_arr[mask]
    y_arr = y_arr[mask]

    if len(t_arr) < min_len:
        return None

    unique_t, unique_indices = np.unique(t_arr, return_index=True)
    y_arr = y_arr[unique_indices]
    t_arr = unique_t

    if len(t_arr) < min_len or len(t_arr) > max_len:
        return None

    log_y = np.log(y_arr)
    dlog_y = np.gradient(log_y, t_arr)
    peaks, _ = find_peaks(dlog_y)

    if len(peaks) > 0:
        if peak_number <= len(peaks):
            peak_idx = int(peaks[peak_number - 1])
        else:
            peak_idx = int(np.nanargmax(dlog_y))
    else:
        peak_idx = int(np.nanargmax(dlog_y))

    slope = float(dlog_y[peak_idx])
    t_peak = float(t_arr[peak_idx])
    t_start = max(t_peak - half_width, t_arr[0])
    t_end = min(t_peak + half_width, t_arr[-1])

    return slope, t_start, t_end


def compute_metrics(timeseries_dict, params):
    """Вычисляет скалярные метрики. НЕ вызывается при загрузке в БД —
       используется из ноутбуков и ad-hoc анализа."""

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
    amplification_max = compute_amplification_by_max(t, Ea, t_max=60.0)
    amplification_median = compute_amplification_by_median_after_time(
        t, Ea, t_start=t_full
    )

    # --- 5. increment from the notebook-style peak finder ---
    increment_peak = find_increment_peak(
        t,
        Ea,
        min_len=3,
        max_len=10,
        t_min=0.0,
        t_max=15.0,
        peak_number=2,
        half_width=0.5,
    )
    growth_rate_peak = None if increment_peak is None else increment_peak[0]

    # --- 6. Z ---
    amplification = (
        amplification_median
        if not np.isnan(amplification_median)
        else amplification_max
    )
    Z = np.log(amplification) / (2 * np.pi)

    u = params.get("Parameters", {}).get("u", 1)
    ZtoZ0 = np.log(amplification) / (2 * np.pi * u ** (-2))

    return {
        # времена
        "t_interaction_full": float(t_full),
        "t_exponential": None if t_stage is None else float(t_stage),

        # инкременты
        "growth_rate_avg": float(gr_avg),
        "growth_rate_stage": None if gr_stage is None else float(gr_stage),
        "growth_rate_peak": (
            None if growth_rate_peak is None else float(growth_rate_peak)
        ),

        "amplification": float(amplification),
        "amplification_max": float(amplification_max),
        "amplification_median_after_t_full": float(amplification_median),
        "amplification_pred": float(amplification_pred),

        "Z": float(Z),
        "ZtoZ0": float(ZtoZ0),
    }

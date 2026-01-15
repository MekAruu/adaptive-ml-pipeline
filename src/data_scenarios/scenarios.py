from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional
import numpy as np
import pandas as pd


# -----------------------------
# Что такое "батч" в нашем фреймворке
# -----------------------------
@dataclass
class StreamBatch:
    batch_id: int
    t_start: int
    t_end: int
    X: pd.DataFrame
    y: pd.Series
    meta: Dict  # drift_type, drift_active, drift_strength, intensity, etc.


# =========================================================
# УПРАВЛЕНИЕ ИНТЕНСИВНОСТЬЮ (Ось A)
# =========================================================
def _resolve_intensity(intensity: str):
    """
    intensity: 'weak' | 'medium' | 'strong'
    Управляет:
      - noise_mul: усиление/ослабление шума
      - width_mul: насколько растягивать/сжимать gradual drift
      - drift_strength_mul: масштаб силы дрейфа в meta
    """
    intensity = (intensity or "medium").lower()
    if intensity == "weak":
        return {"noise_mul": 0.7, "width_mul": 1.5, "drift_strength_mul": 0.6}
    if intensity == "medium":
        return {"noise_mul": 1.0, "width_mul": 1.0, "drift_strength_mul": 1.0}
    if intensity == "strong":
        return {"noise_mul": 1.6, "width_mul": 0.6, "drift_strength_mul": 1.6}
    raise ValueError(f"Unknown intensity='{intensity}'. Use: weak|medium|strong")


# -----------------------------
# Базовый генератор признаков (универсальный: time + сезонность + табличные фичи)
# -----------------------------
def _make_base_features(
    t: np.ndarray,
    seed: int = 42,
    n_exog: int = 3,
    period: float = 50.0
) -> pd.DataFrame:
    """
    Универсальные признаки:
    - time index (t)
    - sin/cos сезонность
    - exogenous табличные признаки (x1..xk)
    """
    rng = np.random.default_rng(seed)

    sin = np.sin(t / period)
    cos = np.cos(t / period)

    exog = {}
    for i in range(1, n_exog + 1):
        exog[f"x{i}"] = rng.normal(loc=0.0, scale=1.0, size=len(t))

    X = pd.DataFrame({
        "t": t,
        "sin": sin,
        "cos": cos,
        **exog
    })
    return X


def _generate_target(
    X: pd.DataFrame,
    coefs: Dict[str, float],
    noise_std: float,
    seed: int
) -> pd.Series:
    """
    y = sum(coef_i * X_i) + noise
    """
    rng = np.random.default_rng(seed)
    y = np.zeros(len(X), dtype=float)
    for k, v in coefs.items():
        y += v * X[k].to_numpy()
    y += rng.normal(0.0, noise_std, size=len(X))
    return pd.Series(y)


def _streamify(
    X: pd.DataFrame,
    y: pd.Series,
    batch_size: int,
    meta_per_t: List[Dict]
) -> Iterator[StreamBatch]:
    """
    Делит поток на батчи, прикладывая meta к каждому батчу.
    meta_per_t — список метаданных для каждого t (той же длины, что X).
    """
    n = len(X)
    batch_id = 0
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        meta_slice = meta_per_t[start:end]

        drift_active = any(m.get("drift_active", False) for m in meta_slice)
        drift_strength = float(np.mean([m.get("drift_strength", 0.0) for m in meta_slice])) if meta_slice else 0.0
        drift_type = meta_slice[0].get("drift_type", "none") if meta_slice else "none"
        intensity = meta_slice[0].get("intensity", "medium") if meta_slice else "medium"

        meta = {
            "drift_type": drift_type,
            "drift_active": drift_active,
            "drift_strength": drift_strength,
            "intensity": intensity,
        }

        yield StreamBatch(
            batch_id=batch_id,
            t_start=start,
            t_end=end,
            X=X.iloc[start:end].reset_index(drop=True),
            y=y.iloc[start:end].reset_index(drop=True),
            meta=meta
        )
        batch_id += 1


# =========================================================
# СЦЕНАРИИ ДИНАМИКИ (Ось 1) + intensity
# Все сценарии принимают intensity и используют его для noise / ширины / drift_strength.
# =========================================================

def scenario_abrupt_drift(
    n: int = 2000,
    batch_size: int = 50,
    drift_point: int = 1000,
    seed: int = 42,
    intensity: str = "medium"
) -> Iterator[StreamBatch]:
    """
    Abrupt drift: в момент drift_point резко меняются коэффициенты зависимости y от X.
    """
    p = _resolve_intensity(intensity)
    noise_std = 0.05 * p["noise_mul"]
    strength_mul = p["drift_strength_mul"]

    t = np.arange(n)
    X = _make_base_features(t, seed=seed)

    coefs_a = {"sin": 0.5, "cos": 0.2, "x1": 0.15, "x2": -0.05, "x3": 0.10}
    coefs_b = {"sin": 0.1, "cos": 0.6, "x1": -0.10, "x2": 0.20, "x3": 0.05}

    y = pd.Series(np.zeros(n, dtype=float))
    y.iloc[:drift_point] = _generate_target(X.iloc[:drift_point], coefs_a, noise_std=noise_std, seed=seed + 1).to_numpy()
    y.iloc[drift_point:] = _generate_target(X.iloc[drift_point:], coefs_b, noise_std=noise_std, seed=seed + 2).to_numpy()

    meta_per_t = []
    for i in range(n):
        meta_per_t.append({
            "drift_type": "abrupt",
            "intensity": intensity,
            "drift_active": i >= drift_point,
            "drift_strength": float(strength_mul) if i >= drift_point else 0.0
        })

    return _streamify(X, y, batch_size, meta_per_t)


def scenario_gradual_drift(
    n: int = 2000,
    batch_size: int = 50,
    drift_start: int = 800,
    drift_end: int = 1300,
    seed: int = 42,
    intensity: str = "medium"
) -> Iterator[StreamBatch]:
    """
    Gradual drift: коэффициенты плавно переходят от A к B на интервале [drift_start, drift_end].
    """
    p = _resolve_intensity(intensity)
    noise_std = 0.05 * p["noise_mul"]
    strength_mul = p["drift_strength_mul"]

    t = np.arange(n)
    X = _make_base_features(t, seed=seed)

    coefs_a = {"sin": 0.55, "cos": 0.15, "x1": 0.10, "x2": 0.00, "x3": 0.12}
    coefs_b = {"sin": 0.10, "cos": 0.65, "x1": -0.08, "x2": 0.18, "x3": 0.02}

    meta_per_t = []
    y = np.zeros(n, dtype=float)

    for i in range(n):
        if i < drift_start:
            alpha = 0.0
        elif i > drift_end:
            alpha = 1.0
        else:
            alpha = (i - drift_start) / max(1, (drift_end - drift_start))

        coefs = {k: (1 - alpha) * coefs_a[k] + alpha * coefs_b[k] for k in coefs_a.keys()}
        yi = _generate_target(X.iloc[i:i+1], coefs, noise_std=noise_std, seed=seed + 100 + i).iloc[0]
        y[i] = yi

        meta_per_t.append({
            "drift_type": "gradual",
            "intensity": intensity,
            "drift_active": drift_start <= i <= drift_end,
            "drift_strength": float(alpha) * float(strength_mul)
        })

    return _streamify(X, pd.Series(y), batch_size, meta_per_t)


def scenario_recurring_drift(
    n: int = 2000,
    batch_size: int = 50,
    regime_len: int = 300,
    seed: int = 42,
    intensity: str = "medium"
) -> Iterator[StreamBatch]:
    """
    Recurring drift: режимы A/B повторяются: A -> B -> A -> B ...
    """
    p = _resolve_intensity(intensity)
    noise_std = 0.05 * p["noise_mul"]
    strength_mul = p["drift_strength_mul"]

    t = np.arange(n)
    X = _make_base_features(t, seed=seed)

    coefs_a = {"sin": 0.45, "cos": 0.25, "x1": 0.12, "x2": -0.04, "x3": 0.08}
    coefs_b = {"sin": 0.15, "cos": 0.55, "x1": -0.06, "x2": 0.22, "x3": 0.03}

    y = np.zeros(n, dtype=float)
    meta_per_t = []

    for i in range(n):
        regime_id = (i // regime_len) % 2
        coefs = coefs_a if regime_id == 0 else coefs_b
        yi = _generate_target(X.iloc[i:i+1], coefs, noise_std=noise_std, seed=seed + 200 + i).iloc[0]
        y[i] = yi

        meta_per_t.append({
            "drift_type": "recurring",
            "intensity": intensity,
            "drift_active": True,
            "drift_strength": float(strength_mul)
        })

    return _streamify(X, pd.Series(y), batch_size, meta_per_t)


def scenario_seasonality_change(
    n: int = 2000,
    batch_size: int = 50,
    change_point: int = 1000,
    seed: int = 42,
    intensity: str = "medium"
) -> Iterator[StreamBatch]:
    """
    Seasonality change: меняется период сезонности (sin/cos) после change_point.
    """
    p = _resolve_intensity(intensity)
    noise_std = 0.05 * p["noise_mul"]
    strength_mul = p["drift_strength_mul"]

    t = np.arange(n)
    rng = np.random.default_rng(seed)

    period_a = 50.0
    period_b = 20.0

    sin = np.where(t < change_point, np.sin(t / period_a), np.sin(t / period_b))
    cos = np.where(t < change_point, np.cos(t / period_a), np.cos(t / period_b))

    X = pd.DataFrame({
        "t": t,
        "sin": sin,
        "cos": cos,
        "x1": rng.normal(0, 1, size=n),
        "x2": rng.normal(0, 1, size=n),
        "x3": rng.normal(0, 1, size=n),
    })

    coefs = {"sin": 0.55, "cos": 0.25, "x1": 0.10, "x2": -0.02, "x3": 0.05}
    y = _generate_target(X, coefs, noise_std=noise_std, seed=seed + 3)

    meta_per_t = []
    for i in range(n):
        meta_per_t.append({
            "drift_type": "seasonality_change",
            "intensity": intensity,
            "drift_active": i >= change_point,
            "drift_strength": float(strength_mul) if i >= change_point else 0.0
        })

    return _streamify(X, y, batch_size, meta_per_t)


def scenario_noise_increase(
    n: int = 2000,
    batch_size: int = 50,
    change_point: int = 1000,
    noise_a: float = 0.03,
    noise_b: float = 0.15,
    seed: int = 42,
    intensity: str = "medium"
) -> Iterator[StreamBatch]:
    """
    Noise increase: модель зависимости не меняется, но шум резко увеличивается.
    intensity усиливает контраст noise_a/noise_b.
    """
    p = _resolve_intensity(intensity)
    strength_mul = p["drift_strength_mul"]

    # усиливаем разницу шума через intensity
    noise_a2 = noise_a * p["noise_mul"]
    noise_b2 = noise_b * p["noise_mul"]

    t = np.arange(n)
    X = _make_base_features(t, seed=seed)

    coefs = {"sin": 0.50, "cos": 0.20, "x1": 0.12, "x2": -0.03, "x3": 0.06}

    y = pd.Series(np.zeros(n, dtype=float))
    y.iloc[:change_point] = _generate_target(X.iloc[:change_point], coefs, noise_std=noise_a2, seed=seed + 10).to_numpy()
    y.iloc[change_point:] = _generate_target(X.iloc[change_point:], coefs, noise_std=noise_b2, seed=seed + 11).to_numpy()

    meta_per_t = []
    for i in range(n):
        meta_per_t.append({
            "drift_type": "noise_increase",
            "intensity": intensity,
            "drift_active": i >= change_point,
            "drift_strength": float(strength_mul) if i >= change_point else 0.0
        })

    return _streamify(X, y, batch_size, meta_per_t)


# =========================================================
# УНИФИКАЦИЯ API (чтобы run_matrix.py работал со всеми сценариями одинаково)
# Единый интерфейс:
# SCENARIOS[name](n=..., batch_size=..., drift_point=..., seed=..., intensity=...)
# drift_point маппится на change_point / drift_start / etc.
# =========================================================

def _wrap_change_point(fn):
    def wrapper(
        n: int = 2000,
        batch_size: int = 50,
        drift_point: int = 1000,
        seed: int = 42,
        intensity: str = "medium",
        **kwargs
    ):
        return fn(n=n, batch_size=batch_size, change_point=drift_point, seed=seed, intensity=intensity, **kwargs)
    return wrapper


def _wrap_gradual(fn, base_width: int = 500):
    # drift_point трактуем как drift_start
    def wrapper(
        n: int = 2000,
        batch_size: int = 50,
        drift_point: int = 800,
        seed: int = 42,
        intensity: str = "medium",
        **kwargs
    ):
        p = _resolve_intensity(intensity)
        width = int(max(50, base_width * p["width_mul"]))  # минимум ширины, чтобы не схлопнулось в 0
        drift_start = int(drift_point)
        drift_end = int(min(n - 1, drift_start + width))
        return fn(
            n=n,
            batch_size=batch_size,
            drift_start=drift_start,
            drift_end=drift_end,
            seed=seed,
            intensity=intensity,
            **kwargs
        )
    return wrapper


def _wrap_recurring(fn):
    # drift_point не нужен
    def wrapper(
        n: int = 2000,
        batch_size: int = 50,
        drift_point: int = 1000,
        seed: int = 42,
        intensity: str = "medium",
        **kwargs
    ):
        return fn(n=n, batch_size=batch_size, seed=seed, intensity=intensity, **kwargs)
    return wrapper


# =========================================================
# РЕЕСТР СЦЕНАРИЕВ (унифицированный)
# =========================================================
SCENARIOS = {
    "abrupt": scenario_abrupt_drift,
    "gradual": _wrap_gradual(scenario_gradual_drift, base_width=500),
    "recurring": _wrap_recurring(scenario_recurring_drift),
    "seasonality_change": _wrap_change_point(scenario_seasonality_change),
    "noise_increase": _wrap_change_point(scenario_noise_increase),
}

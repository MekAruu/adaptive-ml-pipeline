import json
from pathlib import Path
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

def load_data():
    # Демо с дрейфом: во второй половине меняется зависимость
    n = 2000
    t = np.arange(n)
    X = pd.DataFrame({
        "t": t,
        "sin": np.sin(t/50),
        "cos": np.cos(t/50)
    })
    y = np.zeros(n)
    y[:1000] = 0.5 * X["sin"][:1000] + 0.2 * X["cos"][:1000] + 0.05*np.random.randn(1000)
    y[1000:] = 0.1 * X["sin"][1000:] + 0.6 * X["cos"][1000:] + 0.05*np.random.randn(1000)
    return X, pd.Series(y)

def simple_divergence(a: np.ndarray, b: np.ndarray):
    # очень лёгкий proxy (можно заменить позже на KL/JS)
    return float(abs(a.mean() - b.mean()) / (a.std() + 1e-9))

def run_adaptive(results_dir: Path, window=200, step=50, drift_threshold=0.25):
    X, y = load_data()

    # ✅ контроль, что y нормальный (числовой)
    print("y dtype:", y.dtype, "y min/max:", float(y.min()), float(y.max()))

    # ✅ scaler обязателен для SGDRegressor
    scaler = StandardScaler()

    # initial window
    X0 = X.iloc[:window]
    y0 = y.iloc[:window]

    scaler.fit(X0)

    model = SGDRegressor(max_iter=2000, tol=1e-3, random_state=42)
    model.partial_fit(scaler.transform(X0), y0)

    maes, rmses = [], []
    drift_events = 0

    for i in range(window, len(X), step):
        X_batch = X.iloc[i:i+step]
        y_batch = y.iloc[i:i+step]
        if len(X_batch) == 0:
            break

        # drift check: compare recent window with current batch
        prev = y.iloc[max(0, i-window):i].to_numpy()
        curr = y_batch.to_numpy()
        div = simple_divergence(prev, curr)

        # predict (with scaling)
        Xb = scaler.transform(X_batch)
        pred = model.predict(Xb)

        maes.append(mean_absolute_error(y_batch, pred))
        rmses.append(mean_squared_error(y_batch, pred) ** 0.5)

        # update only if drift detected
        if div > drift_threshold:
            drift_events += 1
            recent_X = X.iloc[max(0, i-window):i]
            recent_y = y.iloc[max(0, i-window):i]

            # ✅ важно: обновление тоже через scaler
            model.partial_fit(scaler.transform(recent_X), recent_y)

    out = {
        "model": "SGDRegressor",
        "setting": f"drift-aware partial_fit + scaling (window={window}, step={step}, thr={drift_threshold})",
        "MAE": float(np.mean(maes)),
        "RMSE": float(np.mean(rmses)),
        "drift_updates": drift_events
    }
    (results_dir / "adaptive_metrics.json").write_text(json.dumps(out, indent=2))
    print("Adaptive:", out)

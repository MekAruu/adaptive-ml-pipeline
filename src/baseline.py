import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler


# ---------- Data (пока demo, позже подставим твой датасет) ----------
def load_data():
    n = 2000
    t = np.arange(n)
    X = pd.DataFrame({
        "t": t,
        "sin": np.sin(t / 50),
        "cos": np.cos(t / 50),
    })
    y = np.zeros(n)
    y[:1000] = 0.5 * X["sin"][:1000] + 0.2 * X["cos"][:1000] + 0.05 * np.random.randn(1000)
    y[1000:] = 0.1 * X["sin"][1000:] + 0.6 * X["cos"][1000:] + 0.05 * np.random.randn(1000)
    return X, pd.Series(y)


def simple_divergence(a: np.ndarray, b: np.ndarray):
    return float(abs(a.mean() - b.mean()) / (a.std() + 1e-9))


# ---------- Core evaluation protocol (walk-forward in batches) ----------
def iter_batches(X, y, start=800, step=50):
    batch_id = 0
    for i in range(start, len(X), step):
        X_train, y_train = X.iloc[:i], y.iloc[:i]
        X_test, y_test = X.iloc[i:i + step], y.iloc[i:i + step]
        if len(X_test) == 0:
            break
        yield batch_id, i, X_train, y_train, X_test, y_test
        batch_id += 1


def summarize(records, name, results_dir: Path):
    df = pd.DataFrame(records)
    mae = float(df["mae"].mean())
    rmse = float(df["rmse"].mean())

    out = {"mode": name, "MAE": mae, "RMSE": rmse}
    (results_dir / f"{name}_metrics.json").write_text(json.dumps(out, indent=2))
    return out, df


def save_plot_mae(df_dict, results_dir: Path):
    plt.figure()
    for name, df in df_dict.items():
        plt.plot(df["batch_id"], df["mae"], label=name)
    plt.xlabel("Batch id")
    plt.ylabel("MAE")
    plt.title("MAE over time (walk-forward batches)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "figures" / "mae_over_time.png", dpi=200)
    plt.close()


def save_plot_drift(df, results_dir: Path):
    plt.figure()
    plt.plot(df["batch_id"], df["divergence"], label="divergence")
    drift_points = df[df["drift_triggered"] == 1]
    plt.scatter(drift_points["batch_id"], drift_points["divergence"])
    plt.xlabel("Batch id")
    plt.ylabel("Divergence")
    plt.title("Drift events (points) over time")
    plt.tight_layout()
    plt.savefig(results_dir / "figures" / "drift_events.png", dpi=200)
    plt.close()


# ---------- Mode 1: Static (train once) ----------
def run_static(results_dir: Path, start=800, step=50):
    X, y = load_data()
    model = LinearRegression()

    # train once on initial train split
    X0, y0 = X.iloc[:start], y.iloc[:start]
    model.fit(X0, y0)

    records = []
    for batch_id, idx, X_train, y_train, X_test, y_test in iter_batches(X, y, start=start, step=step):
        pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, pred)
        rmse = mean_squared_error(y_test, pred) ** 0.5
        records.append({"batch_id": batch_id, "t_index": idx, "mae": mae, "rmse": rmse})

    out, df = summarize(records, "static", results_dir)
    (results_dir / "static_over_time.csv").write_text(df.to_csv(index=False))
    print("Static:", out)


# ---------- Mode 2: Periodic retraining (every N batches) ----------
def run_periodic(results_dir: Path, retrain_every=5, start=800, step=50):
    X, y = load_data()
    model = Ridge(alpha=1.0)

    records = []
    for batch_id, idx, X_train, y_train, X_test, y_test in iter_batches(X, y, start=start, step=step):
        if batch_id == 0 or (batch_id % retrain_every == 0):
            model.fit(X_train, y_train)

        pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, pred)
        rmse = mean_squared_error(y_test, pred) ** 0.5
        records.append({"batch_id": batch_id, "t_index": idx, "mae": mae, "rmse": rmse})

    out, df = summarize(records, "periodic", results_dir)
    (results_dir / "periodic_over_time.csv").write_text(df.to_csv(index=False))
    print("Periodic:", out)


# ---------- Mode 3: Drift-triggered retraining on sliding window ----------
def run_drift(results_dir: Path, window=200, step=50, drift_threshold=0.25, start=800):
    X, y = load_data()

    scaler = StandardScaler()
    model = Ridge(alpha=1.0)

    # initial fit on first window
    X0 = X.iloc[:window]
    y0 = y.iloc[:window]
    scaler.fit(X0)
    model.fit(scaler.transform(X0), y0)

    records = []
    drift_updates = 0

    batch_id = 0
    for i in range(start, len(X), step):
        X_test = X.iloc[i:i + step]
        y_test = y.iloc[i:i + step]
        if len(X_test) == 0:
            break

        # divergence check: prev window vs current batch
        prev = y.iloc[max(0, i - window):i].to_numpy()
        curr = y_test.to_numpy()
        div = simple_divergence(prev, curr)

        pred = model.predict(scaler.transform(X_test))
        mae = mean_absolute_error(y_test, pred)
        rmse = mean_squared_error(y_test, pred) ** 0.5

        drift_triggered = 0
        if div > drift_threshold:
            drift_triggered = 1
            drift_updates += 1

            recent_X = X.iloc[max(0, i - window):i]
            recent_y = y.iloc[max(0, i - window):i]
            # keep scaler fixed for fairness (можно отдельно исследовать)
            model.fit(scaler.transform(recent_X), recent_y)

        records.append({
            "batch_id": batch_id,
            "t_index": i,
            "mae": mae,
            "rmse": rmse,
            "divergence": div,
            "drift_triggered": drift_triggered
        })
        batch_id += 1

    df = pd.DataFrame(records)
    out = {
        "mode": "drift",
        "MAE": float(df["mae"].mean()),
        "RMSE": float(df["rmse"].mean()),
        "drift_updates": drift_updates
    }
    (results_dir / "drift_metrics.json").write_text(json.dumps(out, indent=2))
    (results_dir / "drift_over_time.csv").write_text(df.to_csv(index=False))
    save_plot_drift(df, results_dir)
    print("Drift:", out)

    # build summary + global plots (static/periodic/drift)
    return out


# ---------- Helper: run all + one combined summary file + combined plot ----------
def run_all(results_dir: Path):
    run_static(results_dir)
    run_periodic(results_dir, retrain_every=5)
    run_drift(results_dir)

    # load over-time and plot together
    df_static = pd.read_csv(results_dir / "static_over_time.csv")
    df_periodic = pd.read_csv(results_dir / "periodic_over_time.csv")
    df_drift = pd.read_csv(results_dir / "drift_over_time.csv")

    save_plot_mae({"static": df_static, "periodic": df_periodic, "drift": df_drift}, results_dir)

    summary = {
        "static": json.loads((results_dir / "static_metrics.json").read_text()),
        "periodic": json.loads((results_dir / "periodic_metrics.json").read_text()),
        "drift": json.loads((results_dir / "drift_metrics.json").read_text()),
    }
    (results_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2))

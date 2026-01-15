# src/run_matrix.py

from __future__ import annotations

from typing import Any, Dict, List
import inspect
import pandas as pd

from data_scenarios.scenarios import SCENARIOS
from models.zoo import build_model_zoo

from strategies.static import StaticStrategy
from strategies.periodic import PeriodicStrategy
from strategies.drift_aware import DriftAwareErrorStrategy


def _build_stream(
    scenario_name: str,
    n: int,
    batch_size: int,
    drift_point: int,
    intensity: str,
):
    """
    Создаёт поток батчей.
    Если сценарий принимает intensity — передаём.
    Иначе вызываем без него (чтобы не падать).
    """
    fn = SCENARIOS[scenario_name]
    sig = inspect.signature(fn)

    kwargs = dict(n=n, batch_size=batch_size, drift_point=drift_point)

    if "intensity" in sig.parameters:
        kwargs["intensity"] = intensity

    return fn(**kwargs)


def run_one(
    zoo,
    model_key: str,
    strategy,
    strategy_tag: str,
    scenario_name: str,
    intensity: str = "medium",
    n: int = 1200,
    batch_size: int = 50,
    drift_point: int = 600,
) -> Dict[str, Any]:
    """
    One experiment:
    - new stream (scenario + intensity)
    - fresh model reset
    - apply one strategy
    - return row dict for CSV
    """

    # 1) Stream
    stream = _build_stream(
        scenario_name=scenario_name,
        n=n,
        batch_size=batch_size,
        drift_point=drift_point,
        intensity=intensity,
    )

    # 2) Fresh model
    model = zoo[model_key]
    model.reset()

    # 3) Apply strategy
    summary = strategy.apply(
        model,
        stream,
        scenario_name=scenario_name,
        model_key=model_key,
    )

    # 4) Row
    # summary.strategy_name уже приходит из base.py (StrategySummary),
    # но чтобы отличать plastic vs stable — используем strategy_tag.
    return {
        "scenario": scenario_name,
        "intensity": intensity,
        "model": model_key,
        "strategy": strategy_tag,  # <- главное исправление (не трогаем property)
        "MAE": float(summary.mae_mean),
        "RMSE": float(summary.rmse_mean),
        "updates": int(summary.updates),
        "avg_update_ms": float(summary.avg_update_time_ms),
        "reaction_delay_batches": summary.reaction_delay_batches,
    }


def main():
    zoo = build_model_zoo()

    # Scenarios and drift intensity
    scenario_list = ["abrupt", "gradual", "recurring"]
    intensity_list = ["weak", "medium", "strong"]

    # Models (must exist in zoo)
    model_list = ["sgd", "ridge", "rf", "hgb"]

    # Strategies (НЕ меняем имена внутри объектов, только tags)
    strategies = [
        ("static", StaticStrategy(warmup_batches=2)),
        ("periodic_k5", PeriodicStrategy(every_k_batches=5, warmup_batches=2)),

        # Drift-aware: plastic (агрессивная)
        ("drift_error_plastic",
         DriftAwareErrorStrategy(
             err_window=3,
             spike_factor=1.5,
             update_window_batches=2,
             cooldown_batches=1,
             min_improvement=0.00,
             max_updates=20,
             warmup_batches=2,
         )),

        # Drift-aware: stable (консервативная)
        ("drift_error_stable",
         DriftAwareErrorStrategy(
             err_window=7,
             spike_factor=2.0,
             update_window_batches=4,
             cooldown_batches=4,
             min_improvement=0.02,
             max_updates=8,
             warmup_batches=2,
         )),
    ]

    rows: List[Dict[str, Any]] = []

    for sc in scenario_list:
        if sc not in SCENARIOS:
            print(f"[skip] scenario '{sc}' not in SCENARIOS")
            continue

        for intensity in intensity_list:
            for mk in model_list:
                if mk not in zoo:
                    print(f"[skip] model '{mk}' not in zoo")
                    continue

                for strat_tag, strat_obj in strategies:
                    rows.append(
                        run_one(
                            zoo=zoo,
                            model_key=mk,
                            strategy=strat_obj,
                            strategy_tag=strat_tag,
                            scenario_name=sc,
                            intensity=intensity,
                            n=1200,
                            batch_size=50,
                            drift_point=600,
                        )
                    )

    df = pd.DataFrame(rows)
    df.to_csv("results/exp_matrix.csv", index=False)

    print("\nSaved: results/exp_matrix.csv\n")
    print(df.sort_values(["scenario", "intensity", "model", "strategy"]).to_string(index=False))

    expected = len(scenario_list) * len(intensity_list) * len([m for m in model_list if m in zoo]) * len(strategies)
    print(f"\nRows: {len(df)} (expected ~ {expected})")


if __name__ == "__main__":
    main()

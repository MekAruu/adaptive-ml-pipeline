from data_scenarios.scenarios import SCENARIOS
from models.zoo import build_model_zoo

from strategies.static import StaticStrategy
from strategies.periodic import PeriodicStrategy
from strategies.drift_aware import DriftAwareErrorStrategy


def main():
    scenario_name = "abrupt"
    stream = SCENARIOS[scenario_name](n=800, batch_size=50, drift_point=400)

    zoo = build_model_zoo()
    model_key = "sgd"
    model = zoo[model_key]
    model.reset()

    strategies = [
        StaticStrategy(warmup_batches=2),
        PeriodicStrategy(every_k_batches=5, warmup_batches=2),
        DriftAwareErrorStrategy(err_window=3, spike_factor=1.8, warmup_batches=2),
    ]

    for strat in strategies:
        # важно: каждый запуск со свежим стримом и reset модели
        stream = SCENARIOS[scenario_name](n=800, batch_size=50, drift_point=400)
        model.reset()
        summary = strat.apply(model, stream, scenario_name=scenario_name, model_key=model_key)

        print(
            f"{summary.model_name} | {summary.strategy_name} | "
            f"MAE={summary.mae_mean:.4f} RMSE={summary.rmse_mean:.4f} "
            f"updates={summary.updates} avgUpdMs={summary.avg_update_time_ms:.1f} "
            f"reactionDelay={summary.reaction_delay_batches}"
        )


if __name__ == "__main__":
    main()

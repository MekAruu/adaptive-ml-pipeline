from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Iterable, List, Tuple
import time
import numpy as np

from strategies.base import StrategySummary
from data_scenarios.scenarios import StreamBatch


@dataclass
class DriftAwareErrorStrategy:
    """
    Drift-aware strategy (error-triggered) with stability/plasticity controls.

    How it works:
    - model is trained on warmup batches (initial fit)
    - then for each next batch we:
        1) predict -> compute batch error (MAE)
        2) keep recent errors in a sliding window
        3) if error spikes (relative to baseline) -> trigger update
    - update is limited by:
        - cooldown (do not update too frequently)
        - max_updates (resource constraint)
        - min_improvement (avoid useless updates / overfitting)
    """

    # --- drift trigger ---
    err_window: int = 5           # window size for recent errors (batches)
    spike_factor: float = 1.8     # trigger if curr_error > spike_factor * baseline_error

    # --- update scope / intensity ---
    update_window_batches: int = 3  # how many recent batches to use for update

    # --- stability vs plasticity ---
    cooldown_batches: int = 2       # after update, skip triggers for K batches
    min_improvement: float = 0.01   # required relative improvement after update (0.01 = 1%)

    # --- resource limits ---
    max_updates: int = 10           # hard limit for number of updates (proxy for budget)

    # --- general ---
    warmup_batches: int = 2

    @property
    def strategy_name(self) -> str:
        return "drift_error_trigger_v2"

    def apply(self, model, stream: Iterable[StreamBatch], scenario_name: str, model_key: str) -> StrategySummary:
        maes: List[float] = []
        rmses: List[float] = []

        updates = 0
        update_times_ms: List[float] = []

        # reaction delay: we estimate how many batches after drift became active we performed first update
        reaction_delay_batches: Optional[float] = None
        first_drift_batch_id: Optional[int] = None

        recent_errors: List[float] = []
        cooldown_left = 0

        # store recent data (X,y) for update window
        recent_xy: List[Tuple[np.ndarray, np.ndarray]] = []

        started = False
        batch_index = -1

        for batch_index, batch in enumerate(stream):
            Xb = batch.X
            yb = batch.y

            # remember first drift-active batch id (from scenario meta)
            if batch.meta.get("drift_active", False) and first_drift_batch_id is None:
                first_drift_batch_id = batch.batch_id

            # -------- warmup / initial fit --------
            if not started:
                # For warmup we just collect batches and do one fit at the end
                recent_xy.append((Xb.to_numpy(), yb.to_numpy()))
                if batch_index + 1 >= self.warmup_batches:
                    X0 = np.vstack([x for x, _ in recent_xy])
                    y0 = np.hstack([y for _, y in recent_xy])
                    model.fit(X0, y0)
                    started = True
                    recent_xy = []  # reset buffer after initial fit
                continue

            # -------- predict & compute errors --------
            pred = model.predict(Xb.to_numpy())

            err_mae = float(np.mean(np.abs(yb.to_numpy() - pred)))
            err_rmse = float(np.sqrt(np.mean((yb.to_numpy() - pred) ** 2)))

            maes.append(err_mae)
            rmses.append(err_rmse)

            # keep recent error history
            recent_errors.append(err_mae)
            if len(recent_errors) > self.err_window:
                recent_errors.pop(0)

            # keep recent data for potential update
            recent_xy.append((Xb.to_numpy(), yb.to_numpy()))
            if len(recent_xy) > self.update_window_batches:
                recent_xy.pop(0)

            # -------- drift trigger logic --------
            # baseline = median of recent errors (robust) excluding current if possible
            if len(recent_errors) >= max(3, self.err_window // 2):
                baseline = float(np.median(recent_errors[:-1])) if len(recent_errors) > 1 else float(np.median(recent_errors))
            else:
                baseline = float(np.mean(recent_errors)) if recent_errors else err_mae

            is_spike = (baseline > 1e-12) and (err_mae > self.spike_factor * baseline)

            # cooldown prevents too frequent updates (stability)
            if cooldown_left > 0:
                cooldown_left -= 1
                is_spike = False

            # resource budget
            if updates >= self.max_updates:
                is_spike = False

            # -------- update step --------
            if is_spike:
                # measure error BEFORE update on the same batch (err_mae is "before")
                before = err_mae

                # build update dataset from recent batches
                Xu = np.vstack([x for x, _ in recent_xy])
                yu = np.hstack([y for _, y in recent_xy])

                t0 = time.perf_counter()
                model.update(Xu, yu)  # important: model wrapper should implement update()
                t1 = time.perf_counter()

                update_times_ms.append((t1 - t0) * 1000.0)
                updates += 1
                cooldown_left = self.cooldown_batches

                # evaluate quickly AFTER update on current batch
                pred2 = model.predict(Xb.to_numpy())
                after = float(np.mean(np.abs(yb.to_numpy() - pred2)))

                # if update didn't improve enough -> rollback-like behavior:
                # we can't truly rollback for all models easily, so we just reduce aggressiveness by extending cooldown.
                rel_impr = (before - after) / max(1e-12, before)
                if rel_impr < self.min_improvement:
                    cooldown_left = max(cooldown_left, self.cooldown_batches + 2)

                # reaction delay: first update after drift started
                if reaction_delay_batches is None and first_drift_batch_id is not None:
                    reaction_delay_batches = float(batch.batch_id - first_drift_batch_id)

        mae_mean = float(np.mean(maes)) if maes else float("nan")
        rmse_mean = float(np.mean(rmses)) if rmses else float("nan")
        avg_update_ms = float(np.mean(update_times_ms)) if update_times_ms else 0.0

        return StrategySummary(
            strategy_name=self.strategy_name,
            scenario_name=scenario_name,
            model_key=model_key,
            mae_mean=mae_mean,
            rmse_mean=rmse_mean,
            updates=int(updates),
            avg_update_time_ms=avg_update_ms,
            reaction_delay_batches=reaction_delay_batches,
            batches=int(max(0, batch_index + 1))
        )

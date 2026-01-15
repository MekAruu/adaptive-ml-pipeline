from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import time
import inspect

import numpy as np
import pandas as pd


# -----------------------------
# 1) Summary (то, что нужно run_matrix.py)
# -----------------------------
@dataclass
class StrategySummary:
    # обязательные поля (их использует run_matrix.py)
    strategy_name: str
    mae_mean: float
    rmse_mean: float
    updates: int
    avg_update_time_ms: float
    reaction_delay_batches: Optional[float] = None

    # дополнительные поля (чтобы drift_aware.py мог спокойно передавать)
    scenario_name: Optional[str] = None
    model_key: Optional[str] = None   # ✅ ВАЖНО: было model_name — это ломало drift_aware.py
    batches: Optional[int] = None


# -----------------------------
# 2) Детальные метрики по батчам (если потом захочешь графики)
# -----------------------------
@dataclass
class StepMetrics:
    batch_id: int
    mae: float
    rmse: float
    update_happened: bool
    update_time_ms: float
    drift_flag: bool  # ground-truth из meta для синтетики


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2) ** 0.5)


# -----------------------------
# 3) Базовый класс стратегии
# -----------------------------
class TrainingStrategy:
    """
    База для стратегий:
    - warmup: первичное обучение
    - trigger: should_update(...)
    - scope: get_update_data(...)
    """

    name = "base"

    def __init__(self, warmup_batches: int = 1, buffer_max_batches: int = 10):
        self.warmup_batches = int(warmup_batches)
        self.buffer_max_batches = int(buffer_max_batches)

    def should_update(
        self,
        history_y: np.ndarray,
        current_y: np.ndarray,
        y_pred: np.ndarray,
        meta: Dict[str, Any],
        # IMPORTANT: делаем параметр необязательным, чтобы старые стратегии не ломались
        step_idx: Optional[int] = None,
    ) -> bool:
        """Trigger: когда обновлять."""
        return False

    def get_update_data(
        self,
        buffer_X: List[pd.DataFrame],
        buffer_y: List[pd.Series],
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Scope: на каких данных обновляться (по умолчанию — sliding window из буфера)."""
        X = pd.concat(buffer_X, axis=0, ignore_index=True)
        y = pd.concat(buffer_y, axis=0, ignore_index=True)
        return X, y

    def _call_should_update_safe(
        self,
        history_y: np.ndarray,
        current_y: np.ndarray,
        y_pred: np.ndarray,
        meta: Dict[str, Any],
        step_idx: int,
    ) -> bool:
        """
        Универсальный вызов should_update:
        - если стратегия принимает step_idx → передаём
        - если нет → вызываем без step_idx
        """
        sig = inspect.signature(self.should_update)
        params = sig.parameters

        kwargs = dict(
            history_y=history_y,
            current_y=current_y,
            y_pred=y_pred,
            meta=meta,
        )

        if "step_idx" in params:
            kwargs["step_idx"] = step_idx

        return bool(self.should_update(**kwargs))

    def apply(
        self,
        model,
        stream,
        scenario_name: str,
        model_key: str,
    ) -> StrategySummary:
        """
        Запускает стратегию на потоке.
        Требования к model:
        - fit(X, y)
        - predict(X) -> np.ndarray
        - update(X, y)
        - reset() вызывается снаружи (в run_matrix.py)
        """

        buffer_X: List[pd.DataFrame] = []
        buffer_y: List[pd.Series] = []

        steps: List[StepMetrics] = []
        updates = 0
        update_times: List[float] = []

        drift_batch_ids: List[int] = []
        update_batch_ids: List[int] = []

        # -----------------
        # A) Warmup
        # -----------------
        warm = 0
        for batch in stream:
            buffer_X.append(batch.X)
            buffer_y.append(batch.y)
            warm += 1

            if len(buffer_X) > self.buffer_max_batches:
                buffer_X.pop(0)
                buffer_y.pop(0)

            if warm >= self.warmup_batches:
                X0, y0 = self.get_update_data(buffer_X, buffer_y)
                model.fit(X0, y0)
                break

        # -----------------
        # B) Main loop
        # -----------------
        step_idx = 0
        for batch in stream:
            buffer_X.append(batch.X)
            buffer_y.append(batch.y)

            if len(buffer_X) > self.buffer_max_batches:
                buffer_X.pop(0)
                buffer_y.pop(0)

            y_true = batch.y.to_numpy()
            y_pred = model.predict(batch.X)

            mae = float(np.mean(np.abs(y_true - y_pred)))
            r = _rmse(y_true, y_pred)

            meta = getattr(batch, "meta", {}) or {}
            drift_flag = bool(meta.get("drift_active", False))
            if drift_flag:
                drift_batch_ids.append(int(batch.batch_id))

            history_y = buffer_y[-2].to_numpy() if len(buffer_y) >= 2 else y_true

            do_update = self._call_should_update_safe(
                history_y=history_y,
                current_y=y_true,
                y_pred=y_pred,
                meta=meta,
                step_idx=step_idx,
            )

            upd_ms = 0.0
            if do_update:
                t0 = time.perf_counter()
                Xu, yu = self.get_update_data(buffer_X, buffer_y)
                model.update(Xu, yu)
                upd_ms = (time.perf_counter() - t0) * 1000.0

                updates += 1
                update_times.append(upd_ms)
                update_batch_ids.append(int(batch.batch_id))

            steps.append(
                StepMetrics(
                    batch_id=int(batch.batch_id),
                    mae=mae,
                    rmse=r,
                    update_happened=bool(do_update),
                    update_time_ms=float(upd_ms),
                    drift_flag=drift_flag,
                )
            )

            step_idx += 1

        # -----------------
        # C) Summary
        # -----------------
        maes = [s.mae for s in steps]
        rmses = [s.rmse for s in steps]

        avg_upd = float(np.mean(update_times)) if update_times else 0.0

        reaction_delay = None
        if drift_batch_ids and update_batch_ids:
            first_drift = drift_batch_ids[0]
            after = [u for u in update_batch_ids if u >= first_drift]
            if after:
                reaction_delay = float(after[0] - first_drift)

        return StrategySummary(
            strategy_name=self.name,
            mae_mean=float(np.mean(maes)) if maes else float("nan"),
            rmse_mean=float(np.mean(rmses)) if rmses else float("nan"),
            updates=int(updates),
            avg_update_time_ms=float(avg_upd),
            reaction_delay_batches=reaction_delay,
            scenario_name=scenario_name,   # (не обязательно, но полезно)
            model_key=model_key,           # (не обязательно, но полезно)
            batches=len(steps),            # (не обязательно, но полезно)
        )

from __future__ import annotations
from .base import TrainingStrategy


class PeriodicStrategy(TrainingStrategy):
    name = "periodic"

    def __init__(self, every_k_batches: int = 5, warmup_batches: int = 1):
        super().__init__(warmup_batches=warmup_batches)
        self.every_k_batches = every_k_batches
        self._counter = 0

    def should_update(self, history_y, current_y, y_pred, meta):
        self._counter += 1
        return (self._counter % self.every_k_batches) == 0

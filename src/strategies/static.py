from __future__ import annotations
import numpy as np
from .base import TrainingStrategy


class StaticStrategy(TrainingStrategy):
    name = "static"

    def should_update(self, history_y, current_y, y_pred, meta):
        # Никогда не обновляемся
        return False

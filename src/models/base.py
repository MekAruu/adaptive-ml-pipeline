from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol
import numpy as np
import pandas as pd


@dataclass
class ModelInfo:
    name: str
    family: str  # "A_lightweight", "B_ensemble", "C_neural"
    online_capable: bool


class StreamModel(Protocol):
    info: ModelInfo

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        ...

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        ...

    def update(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Online/partial update if supported; otherwise fallback to fit."""
        ...

    def reset(self) -> None:
        ...

    def get_params(self) -> Dict[str, Any]:
        ...

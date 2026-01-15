from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, SGDRegressor, PassiveAggressiveRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

from .base import ModelInfo


class _BaseWrapper:
    def __init__(self) -> None:
        self._model = None
        self._template = None  # sklearn estimator or pipeline to clone
        self._is_fitted = False

    def reset(self) -> None:
        self._model = clone(self._template)
        self._is_fitted = False

    def get_params(self) -> Dict[str, Any]:
        return self._template.get_params(deep=True)


class RidgeBatch(_BaseWrapper):
    """
    Класс A (lightweight), но batch-only: update() делает fit() на новых данных.
    Хороший baseline для periodic / drift-aware retrain.
    """
    info = ModelInfo(name="Ridge", family="A_lightweight", online_capable=False)

    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__()
        self._template = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha, random_state=42))
        ])
        self.reset()

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self._model.fit(X, y)
        self._is_fitted = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._is_fitted:
            return np.zeros(len(X), dtype=float)
        return self._model.predict(X)

    def update(self, X: pd.DataFrame, y: pd.Series) -> None:
        # batch-only fallback
        self.fit(X, y)


class SGDOnline(_BaseWrapper):
    """
    Класс A (online-capable): partial_fit = дешёвые обновления.
    """
    info = ModelInfo(name="SGDRegressor", family="A_lightweight", online_capable=True)

    def __init__(self, alpha: float = 0.0001) -> None:
        super().__init__()
        self._template = Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("sgd", SGDRegressor(
                loss="squared_error",
                penalty="l2",
                alpha=alpha,
                learning_rate="invscaling",
                eta0=0.01,
                random_state=42,
                max_iter=1,   # важно для partial_fit
                tol=None
            ))
        ])
        self.reset()

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        # для онлайн моделей fit = несколько partial_fit шагов
        self._model.named_steps["scaler"].fit(X)
        Xs = self._model.named_steps["scaler"].transform(X)
        self._model.named_steps["sgd"].partial_fit(Xs, y)
        self._is_fitted = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._is_fitted:
            return np.zeros(len(X), dtype=float)
        Xs = self._model.named_steps["scaler"].transform(X)
        return self._model.named_steps["sgd"].predict(Xs)

    def update(self, X: pd.DataFrame, y: pd.Series) -> None:
        # online update на свежем батче
        if not self._is_fitted:
            self.fit(X, y)
            return
        Xs = self._model.named_steps["scaler"].transform(X)
        self._model.named_steps["sgd"].partial_fit(Xs, y)


class PassiveAggressiveOnline(_BaseWrapper):
    """
    Класс A (online-capable): PA регрессия — часто ведёт себя устойчиво при дрейфе.
    """
    info = ModelInfo(name="PassiveAggressiveRegressor", family="A_lightweight", online_capable=True)

    def __init__(self, C: float = 1.0) -> None:
        super().__init__()
        self._template = Pipeline([
            ("scaler", StandardScaler()),
            ("pa", PassiveAggressiveRegressor(
                C=C,
                random_state=42,
                max_iter=1,
                tol=None
            ))
        ])
        self.reset()

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self._model.named_steps["scaler"].fit(X)
        Xs = self._model.named_steps["scaler"].transform(X)
        self._model.named_steps["pa"].partial_fit(Xs, y)
        self._is_fitted = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._is_fitted:
            return np.zeros(len(X), dtype=float)
        Xs = self._model.named_steps["scaler"].transform(X)
        return self._model.named_steps["pa"].predict(Xs)

    def update(self, X: pd.DataFrame, y: pd.Series) -> None:
        if not self._is_fitted:
            self.fit(X, y)
            return
        Xs = self._model.named_steps["scaler"].transform(X)
        self._model.named_steps["pa"].partial_fit(Xs, y)


class RandomForestBatch(_BaseWrapper):
    """
    Класс B (ensemble): batch-only, но сильный по точности, дороже по времени.
    """
    info = ModelInfo(name="RandomForest", family="B_ensemble", online_capable=False)

    def __init__(self, n_estimators: int = 200, max_depth: Optional[int] = None) -> None:
        super().__init__()
        self._template = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )
        self.reset()

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self._model.fit(X, y)
        self._is_fitted = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._is_fitted:
            return np.zeros(len(X), dtype=float)
        return self._model.predict(X)

    def update(self, X: pd.DataFrame, y: pd.Series) -> None:
        # batch-only fallback
        self.fit(X, y)


class HistGBBatch(_BaseWrapper):
    """
    Класс B: HistGradientBoosting — быстрее обычного GB и подходит для больших данных.
    """
    info = ModelInfo(name="HistGradientBoosting", family="B_ensemble", online_capable=False)

    def __init__(self, max_depth: Optional[int] = 6, learning_rate: float = 0.05) -> None:
        super().__init__()
        self._template = HistGradientBoostingRegressor(
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42
        )
        self.reset()

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self._model.fit(X, y)
        self._is_fitted = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._is_fitted:
            return np.zeros(len(X), dtype=float)
        return self._model.predict(X)

    def update(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.fit(X, y)


def build_model_zoo() -> Dict[str, Any]:
    """
    Возвращает словарь экземпляров моделей.
    Ключ — короткое имя (используем в логах/таблицах).
    """
    return {
        "ridge": RidgeBatch(alpha=1.0),
        "sgd": SGDOnline(alpha=0.0001),
        "pa": PassiveAggressiveOnline(C=1.0),
        "rf": RandomForestBatch(n_estimators=200, max_depth=None),
        "hgb": HistGBBatch(max_depth=6, learning_rate=0.05),
    }

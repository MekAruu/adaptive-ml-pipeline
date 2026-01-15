import time
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from data_scenarios.scenarios import SCENARIOS
from models.zoo import build_model_zoo

def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5

# 1) Берём один сценарий (abrupt)
stream = SCENARIOS["abrupt"](n=800, batch_size=50, drift_point=400)

# 2) Берём зоопарк моделей
zoo = build_model_zoo()

# 3) Простая схема: train на первом батче, потом predict на каждом следующем
for name, model in zoo.items():
    model.reset()

    maes, rmses = [], []
    t0 = time.perf_counter()

    first = True
    for batch in stream:
        if first:
            model.fit(batch.X, batch.y)   # initial train
            first = False
            continue

        pred = model.predict(batch.X)
        maes.append(mean_absolute_error(batch.y, pred))
        rmses.append(rmse(batch.y, pred))

        # update для online моделей (пока просто после каждого батча — потом заменим стратегиями)
        model.update(batch.X, batch.y)

    dt = time.perf_counter() - t0
    print(f"{name:>4} | MAE={np.mean(maes):.4f} | RMSE={np.mean(rmses):.4f} | time={dt:.3f}s | online={model.info.online_capable}")

from data_scenarios.scenarios import SCENARIOS

stream = SCENARIOS["abrupt"](n=300, batch_size=50, drift_point=150)

for batch in stream:
    print(batch.batch_id, batch.meta, batch.X.shape, batch.y.shape)

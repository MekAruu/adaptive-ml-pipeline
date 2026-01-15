# Adaptive ML Pipeline for Dynamic Data Streams
This repository contains an experimental machine learning pipeline designed for
**adaptive learning in dynamically changing environments**.  
The project focuses on evaluating different retraining strategies under various
data drift scenarios.

## Project Goals
- Analyze model performance under **abrupt, gradual, and recurring drifts**
- Compare **static and adaptive retraining strategies**
- Measure trade-offs between accuracy, update frequency, and reaction delay
- Provide reproducible experiments for research and academic use

## Key Concepts
- Concept drift
- Adaptive retraining
- Error-based drift detection
- Periodic vs event-driven updates
- Online / incremental learning evaluation

## Experiments
Experiments are executed by running a full strategy–model–scenario matrix.
For each configuration, the following metrics are collected:

- MAE
- RMSE
- Number of updates
- Average update time
- Reaction delay (in batches)

Results are stored in structured CSV and JSON files for further analysis.

## Analysis
The `analysis.ipynb` notebook provides:
- Aggregated performance comparison
- Strategy-wise visualization
- Drift reaction analysis
- Error and latency trade-off exploration

## How to Run
```bash
pip install -r requirements.txt
python src/run_matrix.py



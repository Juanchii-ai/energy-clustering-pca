# Energy Consumption Segmentation with PCA and K-Means

Unsupervised-learning project for identifying operational profiles from energy-consumption indicators.

## Methodology

- Missing-value treatment and robust scaling.
- Sampling-adequacy assessment with KMO.
- Principal Component Analysis for dimensionality reduction.
- K-Means segmentation and silhouette-based model comparison.
- Cluster profiling in the original feature space.

## Recorded result

The original analysis obtained **KMO = 0.786**, retained **3 principal components** explaining approximately **79.5%** of variance, and selected **k = 3** clusters.

## Run

```bash
python src/segment.py --data data/energy.csv --output outputs/clustered_energy.csv
```

## Technologies

Python · Pandas · Scikit-learn · PCA · K-Means · Multivariate analysis


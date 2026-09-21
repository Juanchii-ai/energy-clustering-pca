from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from factor_analyzer.factor_analyzer import calculate_kmo
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--components", type=int, default=3)
    args = parser.parse_args()

    frame = pd.read_csv(args.data)
    numeric = frame.select_dtypes("number").dropna(axis=1, how="all")
    preparation = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", RobustScaler())])
    scaled = preparation.fit_transform(numeric)
    _, kmo_model = calculate_kmo(pd.DataFrame(scaled, columns=numeric.columns))
    print(f"KMO: {kmo_model:.3f}")

    pca = PCA(n_components=args.components, random_state=42)
    components = pca.fit_transform(scaled)
    print(f"Explained variance: {pca.explained_variance_ratio_.sum():.3%}")
    candidates = {}
    for k in range(2, 7):
        labels = KMeans(n_clusters=k, n_init=20, random_state=42).fit_predict(components)
        candidates[k] = silhouette_score(components, labels)
    best_k = max(candidates, key=candidates.get)
    print("Silhouette scores:", candidates)
    print("Selected k:", best_k)
    frame["cluster"] = KMeans(n_clusters=best_k, n_init=50, random_state=42).fit_predict(components)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()


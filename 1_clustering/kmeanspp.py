import pandas as pd
import random


def nearest_centroids(df: pd.DataFrame, centroids: pd.DataFrame) -> None:
    def find_nearest(row: pd.Series) -> pd.Series:
        sq_dist = (centroids - row[centroids.columns]).pow(2).sum(axis=1).pow(0.5)
        nearest_idx = int(sq_dist.idxmin())
        return pd.Series({
            "Cluster": nearest_idx,
            "Distance_to_Centroid": float(sq_dist.loc[nearest_idx])
        })

    df[["Cluster", "Distance_to_Centroid"]] = df.apply(find_nearest, axis=1)


def initialize_centroids(df: pd.DataFrame, n_clusters: int, random_state: int = 42) -> pd.DataFrame:
    random.seed(random_state)
    centroids = df.iloc[[random.randrange(len(df))]].copy()
    nearest_centroids(df, centroids)

    for _ in range(1, n_clusters):
        p_sum = df["Distance_to_Centroid"].sum() * random.random()
        for _, point in df.iterrows():
            p_sum -= point["Distance_to_Centroid"]
            if p_sum <= 0:
                centroids = pd.concat([
                    centroids,
                    point.drop(["Cluster", "Distance_to_Centroid"]).to_frame().T,
                ], ignore_index=False)
                break

        nearest_centroids(df, centroids)

    return centroids


def recalculate_centroids(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("Cluster", sort=True).mean().drop(columns=["Distance_to_Centroid"])


class KMeansPP:
    @staticmethod
    def clustering(df: pd.DataFrame, n_clusters: int, max_iter: int, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
        if n_clusters < 2:
            return (pd.DataFrame(), pd.DataFrame(), False)

        clustered_df = df.copy()
        centroids_df = initialize_centroids(clustered_df, n_clusters, random_state)

        for _ in range(max_iter):
            new_centroids_df = recalculate_centroids(clustered_df)
            if centroids_df.equals(new_centroids_df):
                return (clustered_df, centroids_df.rename_axis("Cluster"), True)

            centroids_df = new_centroids_df
            nearest_centroids(clustered_df, centroids_df)
        
        return (clustered_df, centroids_df.rename_axis("Cluster"), False)

import pandas as pd


def avg_distance(p1: pd.Series | pd.DataFrame, point2: pd.Series) -> float:
    return (p1 - point2).pow(2).sum(axis=1).pow(0.5).mean()


def internal_avg_distance(df: pd.DataFrame, point: pd.Series) -> float:
    return avg_distance(df, point)


def external_min_avg_distance(df: pd.DataFrame, point: pd.Series) -> float:
    return df.groupby("Cluster").apply(lambda x: avg_distance(x, point)).min()


def silhouette(df: pd.DataFrame, feature_columns: list[str]) -> pd.Dataframe:
    def silhouette_score(point: pd.Series) -> float:
        feature_point = point[feature_columns]
        e = external_min_avg_distance(df[df["Cluster"] != point["Cluster"]][feature_columns + ["Cluster"]], feature_point)
        i = internal_avg_distance(df[df["Cluster"] == point["Cluster"]][feature_columns], feature_point)
        return (e - i) / max(e, i)

    metrics = pd.DataFrame()
    metrics["Silhouette"] = df.apply(lambda x: silhouette_score(x), axis=1).groupby(df["Cluster"]).mean()
    return metrics

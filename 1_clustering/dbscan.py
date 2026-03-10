import numpy as np
import pandas as pd


LOUD = 3
MAIN = 2
BORDER = 1


def nearest_points_mask(df: pd.DataFrame, point: pd.Series, feature_columns: list[str], eps: float) -> pd.Series:
    return df[feature_columns].apply(lambda x: np.sqrt((x - point[feature_columns]).pow(2).sum()), axis=1).le(eps)


class DBSCAN:
    @staticmethod
    def clustering(df: pd.DataFrame, eps: float, min_pts: int, feature_columns: list[str]) -> tuple[pd.DataFrame, int]:
        clustered_df = df.copy()
        clustered_df[["Cluster", "Class"]] = [-1, -1]
        clusters_count = 0
        visited = set()

        for idx, point in df.iterrows():
            if idx in visited:
                continue
            visited.add(idx)

            nearest_points = nearest_points_mask(df, point, feature_columns, eps)
            if nearest_points.sum() < min_pts:
                clustered_df.loc[idx, "Class"] = LOUD
                continue

            clusters_count += 1
            cluster_id = idx
            clustered_df.loc[idx, ["Class", "Cluster"]] = [MAIN, cluster_id]

            queue = set([point_idx for point_idx in clustered_df.index[nearest_points].tolist() if point_idx != idx])
            while queue:
                point_idx = queue.pop()
                if point_idx not in visited:
                    visited.add(point_idx)

                    nearest_cluster_points = nearest_points_mask(df, clustered_df.loc[point_idx], feature_columns, eps)
                    if nearest_cluster_points.sum() >= min_pts:
                        clustered_df.loc[point_idx, ["Class", "Cluster"]] = [MAIN, cluster_id]
                        for neighbor_idx in clustered_df.index[nearest_cluster_points].tolist():
                            if neighbor_idx not in visited and neighbor_idx not in queue:
                                queue.add(neighbor_idx)
                    elif clustered_df.loc[point_idx, "Class"] in (-1, LOUD):
                        clustered_df.loc[point_idx, ["Class", "Cluster"]] = [BORDER, cluster_id]

                if clustered_df.loc[point_idx, "Class"] in (-1, LOUD):
                    clustered_df.loc[point_idx, ["Class", "Cluster"]] = [BORDER, cluster_id]

        return (clustered_df, clusters_count)

import pandas as pd
from itertools import combinations

from rules import association_rules


def apriori(df: pd.DataFrame, min_supp: float, bucket_id_column: str, item_column: str) -> pd.DataFrame:
    item_stats_df = (
        df[[bucket_id_column, item_column]]
        .drop_duplicates()
        .groupby(item_column)[bucket_id_column]
        .nunique()
        .rename("count")
        .to_frame()
    ).reset_index()

    baskets_df = (
        df[[bucket_id_column, item_column]]
        .drop_duplicates()
        .groupby(bucket_id_column)[item_column]
        .apply(set)
    )
    baskets_count = len(baskets_df)

    item_stats_df["support"] = item_stats_df["count"] / df[bucket_id_column].nunique()
    item_stats_df = item_stats_df[item_stats_df["support"] > min_supp]

    l_prev = {frozenset([row[item_column]]): row["support"] for _, row in item_stats_df.iterrows()}
    max_k = df.groupby(bucket_id_column)[item_column].nunique().max()

    support_df = pd.DataFrame([{"k": 1, "itemset": set(item), "support": support} for item, support in l_prev.items()])
    for k in range(2, max_k + 1):
        l_prev_list = list(l_prev.keys())

        ck = set()
        for i in range(len(l_prev_list)):
            for j in range(i + 1, len(l_prev_list)):
                new_item = l_prev_list[i] | l_prev_list[j]
                if len(new_item) == k:
                    ck.add(new_item)
        if not ck:
            break

        ck_pruned, prev_level_set = set(), set(l_prev.keys())
        for item in ck:
            if all(frozenset(subset) in prev_level_set for subset in combinations(item, k - 1)):
                ck_pruned.add(item)
        if not ck_pruned:
            break

        l_prev = {item: baskets_df.apply(lambda basket: item.issubset(basket)).sum() / baskets_count for item in ck_pruned}
        l_prev = {item: supp for item, supp in l_prev.items() if supp > min_supp}
        if not l_prev:
            break

        support_df = pd.concat([support_df, pd.DataFrame([{"k": k, "itemset": set(item), "support": support} for item, support in l_prev.items()])], ignore_index=True)

    return support_df


class Apriori:
    @staticmethod
    def create_rules(df: pd.DataFrame, min_supp: float, min_conf: float, bucket_id_column: str, item_column: str) -> pd.DataFrame:
        return association_rules(min_conf, apriori(df, min_supp, bucket_id_column, item_column))

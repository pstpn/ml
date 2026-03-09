from itertools import combinations
import pandas as pd


def find_subsets(itemset: set) -> list[list]:
    item_list = list(itemset)
    subsets = []
    for r in range(1, len(item_list)):
        subsets.extend(combinations(item_list, r))

    return subsets


def association_rules(min_confidence: float, support_df: pd.DataFrame) -> pd.DataFrame:
    normalized = support_df.copy()
    normalized["itemset"] = normalized["itemset"].apply(frozenset)

    support_map = dict(zip(normalized["itemset"], normalized["support"]))
    rules = []

    for itemset, supp_ab in support_map.items():
        if len(itemset) < 2:
            continue

        for a_items in find_subsets(itemset):
            a_set = frozenset(a_items)
            b_set = itemset - a_set
            if not b_set:
                continue

            supp_a = support_map.get(a_set)
            if supp_a is None or supp_a == 0:
                continue

            confidence = supp_ab / supp_a
            if confidence >= min_confidence:
                rules.append(
                    {
                        "A": set(a_set),
                        "B": set(b_set),
                        "support": supp_ab,
                        "confidence": confidence,
                    }
                )

    rules_df = pd.DataFrame(rules)
    if not rules_df.empty:
        rules_df = rules_df.sort_values(["confidence", "support"], ascending=False).reset_index(drop=True)

    return rules_df

import pandas as pd
from src.data_loader import load_and_clean_data, build_utility_matrix
from src.recommenders import CollaborativeFilteringRecommender
from src.association_rules import AssociationRuleMiner
from src.hybrid_recommender import HybridRecommender


def main():
    print("Step 1: Data Loading and Wrangling")
    filepath = 'data/Online Retail.xlsx'
    df = load_and_clean_data(filepath)

    print("\nStep 2: Building Utility Matrix")
    utility_matrix, user_map, item_map = build_utility_matrix(df)

    print("\nStep 3: Mining Association Rules")
    rule_miner = AssociationRuleMiner(min_support=0.01, min_confidence=0.3, min_lift=1.0)
    rule_miner.fit(df)
    rule_miner.print_top_rules(n=10)

    print("\nStep 4: Fitting Item-Item Collaborative Filtering")
    cf_rec = CollaborativeFilteringRecommender(mode='item')
    cf_rec.fit(utility_matrix, user_map, item_map)

    print("\nStep 5: Hybrid Recommender (CF -> Association Rules)")
    hybrid_rec = HybridRecommender(cf_rec, rule_miner)

    sample_user = list(user_map.keys())[0]
    print(f"\nRecommendations for User {sample_user}:")
    cf_items, rule_items = hybrid_rec.recommend(sample_user, cf_k=5, rules_per_item=3)

    print("\n  CF Recommendations:")
    for code in cf_items:
        desc = df[df['StockCode'] == code]['Description'].iloc[0]
        print(f"    {code}: {desc}")

    print("\n  Association Rule Suggestions (triggered by CF items):")
    if rule_items:
        for r in rule_items:
            trigger_desc = df[df['StockCode'] == r['triggered_by']]['Description'].iloc[0]
            print(f"    {r['stock_code']}: {r['description']}")
            print(f"      <- because you might like {r['triggered_by']} ({trigger_desc})")
            print(f"         confidence={r['confidence']:.2f}, lift={r['lift']:.2f}")
    else:
        print("    No rules triggered for this user's CF items.")


if __name__ == "__main__":
    main()

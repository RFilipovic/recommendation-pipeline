import os
import pandas as pd
from src.data_loader import load_and_clean_data, build_utility_matrix
from src.eda import plot_long_tail_distribution
from src.basket_analysis import perform_market_basket_analysis

from src.recommenders import ContentBasedRecommender, CollaborativeFilteringRecommender, LatentFactorRecommender


def main():
    print("Step 1: Data Loading and Wrangling")
    filepath = 'data/Online Retail.xlsx'
    df = load_and_clean_data(filepath)
    
    # Save the filtered data to a new CSV file in the data/ subdirectory for inspection
    filtered_data_path = 'data/filtered_online_retail.csv'
    print(f"Exporting filtered data to {filtered_data_path} for review...")
    df.to_csv(filtered_data_path, index=False)
    print("Filtered data exported successfully.\n")

    utility_matrix, user_map, item_map = build_utility_matrix(df)
    
    print("\nStep 2: EDA")
    plot_long_tail_distribution(df)

    print("\nStep 3: Market Basket Analysis")
    rules = perform_market_basket_analysis(df)
    
    # Print summary statistics
    print(f"\nTotal association rules found: {len(rules)}")
    print(f"Average Lift: {rules['lift'].mean():.2f}")
    print(f"Average Confidence: {rules['confidence'].mean():.2f}")
    
    # Print top 10 rules by lift
    print("\n" + "="*80)
    print("TOP 10 ASSOCIATION RULES (by Lift)")
    print("="*80)
    top_rules = rules.nlargest(10, 'lift')[['antecedents', 'consequents', 'support', 
                                             'confidence', 'lift', 'interest']]
    print(top_rules.to_string())
    
    # Optional: Save rules to CSV for later inspection
    os.makedirs('output', exist_ok=True)
    rules.to_csv('output/association_rules.csv', index=False)
    print(f"\nAll rules saved to: output/association_rules.csv")
    
    print("\nStep 4: Building Recommenders")
    # Updated ContentBasedRecommender initialization and fitting
    cb_rec = ContentBasedRecommender(df, n_components=100)
    cb_rec.fit()
    cb_rec.build_user_profiles(utility_matrix, user_map, item_map)
    
    # Test recommendation for a sample user
    sample_user = list(user_map.keys())[0]
    print(f"\nSample Content-Based Recommendations for User {sample_user}:")
    recommendations = cb_rec.recommend(sample_user, k=5)
    print(recommendations)

    # Add this to see what the items actually are
    for stock_code in recommendations:
        desc = df[df['StockCode'] == stock_code]['Description'].iloc[0]
        print(f"{stock_code}: {desc}")

    # The following recommenders are not yet implemented, so we return early.
    return
    
    # cf_rec = CollaborativeFilteringRecommender(mode='item')
    # cf_rec.fit(utility_matrix)
    # 
    # lf_rec = LatentFactorRecommender()
    # lf_rec.fit(utility_matrix)
    # 
    # hybrid_rec = HybridRecommender(cb_rec, cf_rec, lf_rec, lsh_candidates=jaccard_candidates)
    # 
    # print("\nStep 10: Evaluation")
    # train_df, test_df = train_test_split_chronological(df)
    # evaluate_all_recommenders(
    #     {'CB': cb_rec, 'CF': cf_rec, 'LF': lf_rec, 'Hybrid': hybrid_rec},
    #     test_df, train_df
    # )

if __name__ == "__main__":
    main()

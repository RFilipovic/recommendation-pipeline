import os
import pandas as pd
from src.data_loader import load_and_clean_data, build_utility_matrix
from src.eda import plot_long_tail_distribution
from src.basket_analysis import perform_market_basket_analysis
from src.evaluation import train_test_split_chronological, evaluate_all_recommenders

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

    # ============================================================
    # Collaborative Filtering Recommender Testing
    # ============================================================
    print("\n" + "="*80)
    print("COLLABORATIVE FILTERING RECOMMENDER (Item-Item)")
    print("="*80)
    
    cf_rec = CollaborativeFilteringRecommender(mode='item')
    cf_rec.fit(utility_matrix, user_map, item_map)
    
    print(f"\nSample Collaborative Filtering Recommendations for User {sample_user}:")
    cf_recommendations = cf_rec.predict(sample_user, top_k=5)
    print(cf_recommendations)
    
    # Print descriptions for the recommended items
    for stock_code in cf_recommendations:
        desc = df[df['StockCode'] == stock_code]['Description'].iloc[0]
        print(f"{stock_code}: {desc}")

    # ============================================================
    # Latent Factor Recommender Testing
    # ============================================================
    print("\n" + "="*80)
    print("LATENT FACTOR RECOMMENDER (SVD)")
    print("="*80)
    
    lf_rec = LatentFactorRecommender(n_factors=50)
    lf_rec.fit(utility_matrix, user_map, item_map)
    
    print(f"\nSample Latent Factor Recommendations for User {sample_user}:")
    lf_recommendations = lf_rec.predict(sample_user, top_k=5)
    print(lf_recommendations)
    
    # Print descriptions for the recommended items
    for stock_code in lf_recommendations:
        desc = df[df['StockCode'] == stock_code]['Description'].iloc[0]
        print(f"{stock_code}: {desc}")

    # ============================================================
    # Step 5: Evaluation
    # ============================================================
    print("\n" + "="*80)
    print("STEP 5: EVALUATION")
    print("="*80)
    
    # Split data chronologically
    print("Splitting data chronologically for evaluation...")
    train_df, test_df = train_test_split_chronological(df, test_days=30)
    
    # To avoid data leakage, we must rebuild the utility matrix and re-fit 
    # the recommenders using ONLY the training data.
    # We use a separate cache directory to ensure we don't load the full dataset's matrix.
    print("Rebuilding utility matrix on training data only to prevent data leakage...")
    eval_utility_matrix, eval_user_map, eval_item_map = build_utility_matrix(train_df, cache_dir='cache/eval')
    
    print("Re-fitting recommenders on training data only...")
    eval_cb_rec = ContentBasedRecommender(train_df, n_components=100)
    eval_cb_rec.fit()
    eval_cb_rec.build_user_profiles(eval_utility_matrix, eval_user_map, eval_item_map)
    
    eval_cf_rec = CollaborativeFilteringRecommender(mode='item')
    eval_cf_rec.fit(eval_utility_matrix, eval_user_map, eval_item_map)
    
    eval_lf_rec = LatentFactorRecommender(n_factors=50)
    eval_lf_rec.fit(eval_utility_matrix, eval_user_map, eval_item_map)
    
    # Evaluate all recommenders
    evaluate_all_recommenders(
        recommenders_dict={
            'Content-Based': eval_cb_rec, 
            'Collaborative Filtering': eval_cf_rec, 
            'Latent Factor': eval_lf_rec
        },
        test_df=test_df,
        train_df=train_df,
        k=10
    )

if __name__ == "__main__":
    main()

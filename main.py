"""
Main orchestrator script for The Long-Tail Merchant project.
Run this script to execute the entire pipeline.
"""
import os
import pandas as pd
from src.data_loader import load_and_clean_data, build_utility_matrix
from src.eda import plot_long_tail_distribution
from src.basket_analysis import perform_market_basket_analysis
from src.minhash_lsh import get_k_shingles, build_minhash_signatures, lsh_jaccard, lsh_cosine
from src.clustering import apply_clustering
from src.recommenders import ContentBasedRecommender, CollaborativeFilteringRecommender, LatentFactorRecommender
from src.hybrid_engine import HybridRecommender
from src.evaluation import train_test_split_chronological, evaluate_all_recommenders

def main():
    print("Step 1: Data Loading and Wrangling")
    filepath = 'data/Online Retail.xlsx'
    df = load_and_clean_data(filepath)

    utility_matrix, user_map, item_map = build_utility_matrix(df)
    
    print("\nStep 2: EDA")
    plot_long_tail_distribution(df)

    # ============================================================
    # ADD THIS: Step 3: Market Basket Analysis
    # ============================================================
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
    # ============================================================

    return
    
    print("\nStep 3: Market-Basket Analysis")
    rules = perform_market_basket_analysis(df)
    
    print("\nStep 4: Similarity & LSH")
    descriptions = df.drop_duplicates('StockCode')['Description'].tolist()
    shingles = get_k_shingles(descriptions)
    signatures = build_minhash_signatures(shingles)
    jaccard_candidates = lsh_jaccard(signatures)
    # Cosine LSH requires TF-IDF vectors - implement vectorization before calling
    # cosine_candidates = lsh_cosine(tfidf_matrix)
    
    print("\nStep 5: Clustering")
    # apply_clustering requires vectors - implement TF-IDF extraction before calling
    # apply_clustering(vectors, vector_names)
    
    print("\nStep 6-9: Building Recommenders")
    cb_rec = ContentBasedRecommender()
    cb_rec.fit(descriptions, utility_matrix)
    
    cf_rec = CollaborativeFilteringRecommender(mode='item')
    cf_rec.fit(utility_matrix)
    
    lf_rec = LatentFactorRecommender()
    lf_rec.fit(utility_matrix)
    
    hybrid_rec = HybridRecommender(cb_rec, cf_rec, lf_rec, lsh_candidates=jaccard_candidates)
    
    print("\nStep 10: Evaluation")
    train_df, test_df = train_test_split_chronological(df)
    evaluate_all_recommenders(
        {'CB': cb_rec, 'CF': cf_rec, 'LF': lf_rec, 'Hybrid': hybrid_rec},
        test_df, train_df
    )

if __name__ == "__main__":
    main()

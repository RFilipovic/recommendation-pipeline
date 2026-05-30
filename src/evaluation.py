import numpy as np
import pandas as pd
from typing import Dict, List, Set, Tuple, Any

def train_test_split_chronological(df: pd.DataFrame, test_days: int = 30) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits data chronologically into training and testing sets.
    
    Args:
        df: The cleaned pandas DataFrame containing transaction data.
        test_days: The number of days from the end of the dataset to include in the test set.
        
    Returns:
        A tuple of (train_df, test_df).
    """
    # Ensure InvoiceDate is datetime
    df = df.copy()
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    # 1. Find the max date in the DataFrame.
    max_date = df['InvoiceDate'].max()
    
    # 2. Define a cutoff date = max_date - test_days.
    cutoff_date = max_date - pd.Timedelta(days=test_days)
    
    # 3. Train set = df[df['InvoiceDate'] < cutoff].
    train_df = df[df['InvoiceDate'] < cutoff_date]
    
    # 4. Test set = df[df['InvoiceDate'] >= cutoff].
    test_df = df[df['InvoiceDate'] >= cutoff_date]
    
    # 5. Return train_df, test_df.
    return train_df, test_df

def precision_at_k(recommended_items: List[Any], relevant_items: Set[Any], k: int = 10) -> float:
    """
    Computes Precision@K.
    
    Args:
        recommended_items: A list of recommended item IDs.
        relevant_items: A set of ground truth relevant item IDs.
        k: The number of top recommendations to consider.
        
    Returns:
        The precision at k score.
    """
    if k == 0:
        return 0.0
        
    # 1. Take the top K recommended items.
    top_k_recommendations = recommended_items[:k]
    
    # 2. Count how many are in the relevant_items set.
    hits = len(set(top_k_recommendations) & relevant_items)
    
    # 3. Divide by K.
    precision = hits / k
    
    # 4. Return precision.
    return precision

def recall_at_k(recommended_items: List[Any], relevant_items: Set[Any], k: int = 10) -> float:
    """
    Computes Recall@K.
    
    Args:
        recommended_items: A list of recommended item IDs.
        relevant_items: A set of ground truth relevant item IDs.
        k: The number of top recommendations to consider.
        
    Returns:
        The recall at k score.
    """
    # Handle the case where a user has no relevant items in the test set
    if len(relevant_items) == 0:
        return 0.0
        
    # 1. Take the top K recommended items.
    top_k_recommendations = recommended_items[:k]
    
    # 2. Count how many are in the relevant_items set.
    hits = len(set(top_k_recommendations) & relevant_items)
    
    # 3. Divide by the total number of relevant items.
    recall = hits / len(relevant_items)
    
    # 4. Return recall.
    return recall

def evaluate_all_recommenders(
    recommenders_dict: Dict[str, Any], 
    test_df: pd.DataFrame, 
    train_df: pd.DataFrame, 
    k: int = 10
) -> Dict[str, Dict[str, float]]:
    """
    Evaluates all recommenders and prints a comparison.
    
    Args:
        recommenders_dict: A dictionary mapping recommender names to recommender objects.
        test_df: The test set DataFrame.
        train_df: The train set DataFrame.
        k: The number of top recommendations to evaluate.
        
    Returns:
        A dictionary containing the evaluation metrics for each recommender.
    """
    # 1. For each user in test_df, get their ground truth relevant items from test_df.
    test_users = test_df['CustomerID'].unique()
    train_users = set(train_df['CustomerID'].unique())
    
    # Build ground truth: only include users who appear in BOTH train and test sets
    user_ground_truth: Dict[Any, Set[Any]] = {}
    for user in test_users:
        if user in train_users:
            user_items = set(test_df[test_df['CustomerID'] == user]['StockCode'].unique())
            if len(user_items) > 0:
                user_ground_truth[user] = user_items
                
    results = {}
    
    # 2. For each recommender in recommenders_dict:
    for name, recommender in recommenders_dict.items():
        precisions = []
        recalls = []
        
        for user, relevant_items in user_ground_truth.items():
            # a. Get top_k predictions.
            # Handle different method names across recommender classes
            try:
                if hasattr(recommender, 'predict'):
                    recommended_items = recommender.predict(user, top_k=k)
                elif hasattr(recommender, 'recommend'):
                    recommended_items = recommender.recommend(user, k=k)
                else:
                    raise AttributeError(f"Recommender {name} has no 'predict' or 'recommend' method.")
            except Exception:
                # Fallback for users not found in mapping or other prediction errors
                recommended_items = []
                
            # b. Calculate Precision@K and Recall@K.
            precisions.append(precision_at_k(recommended_items, relevant_items, k))
            recalls.append(recall_at_k(recommended_items, relevant_items, k))
            
        # c. Average across all users.
        avg_precision = np.mean(precisions) if precisions else 0.0
        avg_recall = np.mean(recalls) if recalls else 0.0
        
        results[name] = {
            'Precision@K': avg_precision,
            'Recall@K': avg_recall
        }
        
    # 3. Print a formatted comparison table of all models.
    print(f"\n{'='*60}")
    print(f"{'MODEL EVALUATION RESULTS (K=' + str(k) + ')':^60}")
    print(f"{'='*60}")
    print(f"{'Model':<25} | {'Precision@K':<15} | {'Recall@K':<15}")
    print(f"{'-'*25}-+-{'-'*15}-+-{'-'*15}")
    for name, metrics in results.items():
        print(f"{name:<25} | {metrics['Precision@K']:<15.4f} | {metrics['Recall@K']:<15.4f}")
    print(f"{'='*60}\n")
    
    return results

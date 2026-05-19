import numpy as np

def train_test_split_chronological(df, test_days=30):
    """
    Splits data chronologically.
    
    Steps to implement:
    1. Find the max date in the DataFrame.
    2. Define a cutoff date = max_date - test_days.
    3. Train set = df[df['InvoiceDate'] < cutoff].
    4. Test set = df[df['InvoiceDate'] >= cutoff].
    5. Return train_df, test_df.
    """
    raise NotImplementedError

def precision_at_k(recommended_items, relevant_items, k=10):
    """
    Computes Precision@K.
    
    Steps to implement:
    1. Take the top K recommended items.
    2. Count how many are in the relevant_items set.
    3. Divide by K.
    4. Return precision.
    """
    raise NotImplementedError

def recall_at_k(recommended_items, relevant_items, k=10):
    """
    Computes Recall@K.
    
    Steps to implement:
    1. Take the top K recommended items.
    2. Count how many are in the relevant_items set.
    3. Divide by the total number of relevant items.
    4. Return recall.
    """
    raise NotImplementedError

def evaluate_all_recommenders(recommenders_dict, test_df, train_df, k=10):
    """
    Evaluates all recommenders and prints comparison.
    
    Steps to implement:
    1. For each user in test_df, get their ground truth relevant items from test_df.
    2. For each recommender in recommenders_dict:
       a. Get top_k predictions.
       b. Calculate Precision@K and Recall@K.
       c. Average across all users.
    3. Print a formatted comparison table of all models.
    """
    raise NotImplementedError

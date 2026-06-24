import pandas as pd
import scipy.sparse as sp
import os
import joblib

def load_and_clean_data(filepath, cache_dir='cache'):

    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, 'cleaned_df.joblib')
    
    # Check if cache exists
    if os.path.exists(cache_path):
        print("Used results from cache for cleaned data.")
        return joblib.load(cache_path)

    print(f"Processing data from {filepath}...")

    # 1. Read the Excel file
    df = pd.read_excel(filepath)
    
    # 2. Drop rows where 'CustomerID' is NaN
    df = df.dropna(subset=['CustomerID'])
    
    # 3. Drop rows where 'Quantity' <= 0
    df = df[df['Quantity'] > 0]
    
    # 4. Normalize 'Description': drop NaNs, convert to string, strip whitespace, uppercase
    df = df.dropna(subset=['Description'])
    df['Description'] = df['Description'].astype(str).str.strip().str.upper()
    
    # 5. Save to cache and return the cleaned DataFrame
    joblib.dump(df, cache_path)
    print("Saved cleaned data to cache.")
    
    return df

def build_utility_matrix(df, cache_dir='cache'):

    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, 'utility_matrix_data.joblib')
    
    # Check if cache exists
    if os.path.exists(cache_path):
        print("Used results from cache for utility matrix.")
        utility_matrix, user_mapping, item_mapping = joblib.load(cache_path)
        return utility_matrix, user_mapping, item_mapping

    print("Building utility matrix...")

    grouped = df.groupby(['CustomerID', 'StockCode'])['Quantity'].sum().reset_index()

    pivot_table = grouped.pivot_table(
        index='CustomerID',      # rows = users
        columns='StockCode',     # columns = items
        values='Quantity',       # values = purchase quantity
        aggfunc='sum',           # (already summed, but safe to keep)
        fill_value=0             # NaN -> 0 (user didn't buy this item)
    )

    user_mapping = {customer_id: idx for idx, customer_id in enumerate(pivot_table.index)}
    item_mapping = {stock_code: idx for idx, stock_code in enumerate(pivot_table.columns)}

    utility_matrix = sp.csr_matrix(pivot_table.values)
    
    data_to_cache = (utility_matrix, user_mapping, item_mapping)
    joblib.dump(data_to_cache, cache_path)
    print("Saved utility matrix data to cache.")
    
    return utility_matrix, user_mapping, item_mapping

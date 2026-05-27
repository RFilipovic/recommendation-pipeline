import pandas as pd
import scipy.sparse as sp
import os
import joblib

def load_and_clean_data(filepath, cache_dir='cache'):

    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, 'cleaned_df.joblib')
    
    # Check if cache exists
    if os.path.exists(cache_path):
        print(f"Loading cleaned data from cache: {cache_path}")
        return joblib.load(cache_path)

    print(f"Cache not found. Processing data from {filepath}...")

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
    print(f"Saved cleaned data to cache: {cache_path}")
    
    return df

def build_utility_matrix(df, cache_dir='cache'):

    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, 'utility_matrix_data.joblib')
    
    # Check if cache exists
    if os.path.exists(cache_path):
        print(f"Loading utility matrix and mappings from cache: {cache_path}")
        return joblib.load(cache_path)

    print("Cache not found. Building utility matrix...")

    # ============================================================
    # STEP 1: Group by CustomerID and StockCode, sum Quantity
    # ============================================================
    # This aggregates multiple purchases of the same item by the same user
    # Example: User 12345 bought item A twice (qty 3 + qty 2 = 5 total)
    grouped = df.groupby(['CustomerID', 'StockCode'])['Quantity'].sum().reset_index()

    print(f"After grouping: {len(grouped)} user-item pairs")

    # ============================================================
    # STEP 2: Create pivot table (dense Users × Items matrix)
    # ============================================================
    # Rows = CustomerID, Columns = StockCode, Values = summed Quantity
    # Missing values (user never bought item) become NaN, then fill with 0
    pivot_table = grouped.pivot_table(
        index='CustomerID',      # rows = users
        columns='StockCode',     # columns = items
        values='Quantity',       # values = purchase quantity
        aggfunc='sum',           # (already summed, but safe to keep)
        fill_value=0             # NaN → 0 (user didn't buy this item)
    )

    print(f"Utility matrix shape: {pivot_table.shape[0]} users x {pivot_table.shape[1]} items")

    # ============================================================
    # STEP 3: Create index mappings (for later lookup)
    # ============================================================
    # These mappings let you convert between CustomerID/StockCode and matrix indices
    user_mapping = {customer_id: idx for idx, customer_id in enumerate(pivot_table.index)}
    item_mapping = {stock_code: idx for idx, stock_code in enumerate(pivot_table.columns)}

    print(f"Unique users: {len(user_mapping)}, Unique items: {len(item_mapping)}")

    # ============================================================
    # STEP 4: Convert dense matrix to sparse CSR format
    # ============================================================
    # CSR (Compressed Sparse Row) is efficient for:
    # - Row slicing (getting all items for a user)
    # - Matrix multiplication (used in CF algorithms)
    # - Memory efficiency (stores only non-zero values)
    utility_matrix = sp.csr_matrix(pivot_table.values)

    # ============================================================
    # STEP 5: Calculate and print sparsity
    # ============================================================
    # Sparsity = proportion of ZERO entries in the matrix
    # Retail data is typically 99%+ sparse (users buy very few items)
    num_users, num_items = utility_matrix.shape
    num_non_zero = utility_matrix.nnz  # number of non-zero elements
    sparsity = 1 - (num_non_zero / (num_users * num_items))

    print(f"\n{'='*50}")
    print(f"UTILITY MATRIX STATISTICS")
    print(f"{'='*50}")
    print(f"Shape: {num_users} users x {num_items} items")
    print(f"Non-zero entries: {num_non_zero:,}")
    print(f"Total entries: {num_users * num_items:,}")
    print(f"Sparsity: {sparsity:.4%}")  # e.g., "99.87%"
    print(f"{'='*50}\n")
    
    # ============================================================
    # STEP 6: Save to cache and return the sparse matrix and mappings
    # ============================================================
    data_to_cache = (utility_matrix, user_mapping, item_mapping)
    joblib.dump(data_to_cache, cache_path)
    print(f"Saved utility matrix data to cache: {cache_path}")
    
    return utility_matrix, user_mapping, item_mapping

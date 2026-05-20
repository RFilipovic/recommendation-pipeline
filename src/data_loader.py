import pandas as pd
import scipy.sparse as sp

def load_and_clean_data(filepath):
    """
    Loads the UCI Online Retail dataset and performs data wrangling.
    
    Steps to implement:
    1. Read the Excel file using pandas.read_excel.
    2. Drop rows where 'CustomerID' is NaN.
    3. Drop rows where 'Quantity' <= 0.
    4. Normalize 'Description': drop NaNs, convert to string, strip whitespace, uppercase.
    5. Return the cleaned DataFrame.
    """
    # 1. Read the Excel file
    df = pd.read_excel(filepath)
    
    # 2. Drop rows where 'CustomerID' is NaN
    df = df.dropna(subset=['CustomerID'])
    
    # 3. Drop rows where 'Quantity' <= 0
    df = df[df['Quantity'] > 0]
    
    # 4. Normalize 'Description': drop NaNs, convert to string, strip whitespace, uppercase
    df = df.dropna(subset=['Description'])
    df['Description'] = df['Description'].astype(str).str.strip().str.upper()
    
    # 5. Return the cleaned DataFrame
    return df

def build_utility_matrix(df):
    """
    Converts the cleaned DataFrame into a sparse Users x Items utility matrix.
    
    Steps to implement:
    1. Group by 'CustomerID' and 'StockCode', summing the 'Quantity'.
    2. Use df.pivot_table or df.groupby + unstack to create a dense Users x Items matrix.
    3. Convert the dense matrix to a scipy.sparse.csr_matrix.
    4. Calculate sparsity: 1 - (number_of_non_zero_elements / (num_users * num_items))
    5. Print sparsity.
    6. Return the sparse matrix, user index mapping, and item index mapping.
    """
    # TODO: Implement utility matrix creation
    raise NotImplementedError

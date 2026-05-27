import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

def perform_market_basket_analysis(df):
    """
    Performs Market-Basket Analysis using the A-priori algorithm.
    
    Monotonicity Property Note:
    The A-priori algorithm leverages the downward closure property (monotonicity):
    If an itemset is infrequent, all of its supersets must also be infrequent.
    This allows the algorithm to prune the search space drastically without 
    counting support for large itemsets that contain known infrequent subsets.
    
    Steps to implement:
    1. Group df by 'Invoice' and aggregate 'StockCode' into lists (baskets).
    2. Use mlxtend's TransactionEncoder to one-hot encode the baskets.
    3. Run mlxtend.frequent_patterns.apriori with a min_support (e.g., 0.02).
    4. Run mlxtend.frequent_patterns.association_rules on the frequent itemsets, 
       calculating confidence and lift. Filter by lift > 1.
    5. Calculate Interest (Added Value) = Confidence(A -> B) - Support(B).
    6. Return the rules DataFrame.
    """ 
    # 1. Group df by 'Invoice' and aggregate 'StockCode' into lists (baskets).
    # Each Invoice represents a single transaction (basket) containing multiple items.
    baskets = df.groupby('Invoice')['StockCode'].apply(list).values.tolist()
    
    # 2. Use mlxtend's TransactionEncoder to one-hot encode the baskets.
    te = TransactionEncoder()
    te_ary = te.fit(baskets).transform(baskets)
    basket_sets = pd.DataFrame(te_ary, columns=te.columns_)
    
    # 3. Run mlxtend.frequent_patterns.apriori with a min_support (e.g., 0.02).
    # The A-priori algorithm leverages the monotonicity property to prune the search space.
    frequent_itemsets = apriori(basket_sets, min_support=0.02, use_colnames=True)
    
    # 4. Run mlxtend.frequent_patterns.association_rules on the frequent itemsets, 
    # calculating confidence and lift. Filter by lift > 1.
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
    rules = rules[rules['lift'] > 1]
    
    # 5. Calculate Interest (also known as Added Value).
    # Interest measures the difference between the rule's confidence and the baseline 
    # probability of the consequent. Interest(A -> B) = Confidence(A -> B) - Support(B)
    rules['interest'] = rules['confidence'] - rules['consequent support']
    
    # 6. Return the rules DataFrame.
    return rules

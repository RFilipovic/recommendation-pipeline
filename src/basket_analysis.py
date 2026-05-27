import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

def perform_market_basket_analysis(df):
    # 1. Group df by 'InvoiceNo' and aggregate 'StockCode' into lists (baskets).
    # Each InvoiceNo represents a single transaction (basket) containing multiple items.
    # Convert StockCode to string to avoid TypeError during sorting in TransactionEncoder
    # (The dataset contains a mix of integer and string stock codes like '85123A')
    baskets = df.groupby('InvoiceNo')['StockCode'].apply(lambda x: x.astype(str).tolist()).values.tolist()
    
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
    
    # 6. Drop unwanted metrics from the output
    cols_to_drop = ['representativity', 'leverage', 'conviction', 'zhangs_metric', 'certainty', 'kulczynski']
    rules = rules.drop(columns=[col for col in cols_to_drop if col in rules.columns])
    
    # 7. Return the rules DataFrame.
    return rules

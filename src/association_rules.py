import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import os
import joblib


class AssociationRuleMiner:
    def __init__(self, min_support=0.01, min_confidence=0.3, min_interest=0.0):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.min_interest = min_interest
        self.rules_df = None
        self.item_descriptions = None

    def fit(self, df, cache_dir='cache'):
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, 'association_rules.joblib')

        if os.path.exists(cache_path):
            print("Loaded association rules from cache.")
            self.rules_df, self.item_descriptions = joblib.load(cache_path)
            self._print_stats()
            return self

        print("Building basket matrix for Apriori...")
        basket_data = df.groupby(['InvoiceNo', 'StockCode'])['Quantity'].sum().reset_index()
        basket_data['Quantity'] = 1
        basket_matrix = basket_data.pivot_table(
            index='InvoiceNo',
            columns='StockCode',
            values='Quantity',
            fill_value=0
        )
        basket_matrix = basket_matrix.astype(bool)
        self.num_baskets = len(basket_matrix)

        print(f"Basket matrix shape: {basket_matrix.shape}")
        print(f"Running Apriori with min_support={self.min_support}...")

        frequent_itemsets = apriori(
            basket_matrix,
            min_support=self.min_support,
            use_colnames=True,
            low_memory=True
        )
        print(f"Found {len(frequent_itemsets)} frequent itemsets.")

        if frequent_itemsets.empty:
            print("No frequent itemsets found. Try lowering min_support.")
            self.rules_df = pd.DataFrame()
            self.item_descriptions = {}
            return self

        print("Mining association rules...")
        rules = association_rules(
            frequent_itemsets,
            metric='confidence',
            min_threshold=self.min_confidence
        )

        # Interest = confidence - consequent_support
        rules['interest'] = rules['confidence'] - rules['consequent support']

        self.rules_df = rules[rules['interest'] >= self.min_interest]
        self.rules_df = self.rules_df.sort_values('interest', ascending=False).reset_index(drop=True)

        item_info = df.drop_duplicates('StockCode')[['StockCode', 'Description']]
        self.item_descriptions = dict(zip(item_info['StockCode'], item_info['Description']))

        joblib.dump((self.rules_df, self.item_descriptions), cache_path)
        print("Saved association rules to cache.")
        self._print_stats()

        return self

    def _print_stats(self):
        print(f"\n{'='*50}")
        print("ASSOCIATION RULES STATISTICS")
        print(f"{'='*50}")
        print(f"Total rules: {len(self.rules_df)}")
        if not self.rules_df.empty:
            print(f"Avg confidence: {self.rules_df['confidence'].mean():.4f}")
            print(f"Avg interest:   {self.rules_df['interest'].mean():.4f}")
            print(f"Max interest:   {self.rules_df['interest'].max():.4f}")
        print(f"{'='*50}\n")

    def get_consequents(self, stock_code, top_n=5):
        if self.rules_df is None or self.rules_df.empty:
            return []

        matching = self.rules_df[
            self.rules_df['antecedents'].apply(lambda s: stock_code in s)
        ]

        consequents = []
        seen = set()
        for _, row in matching.iterrows():
            for item in row['consequents']:
                if item not in seen and item != stock_code:
                    seen.add(item)
                    consequents.append({
                        'stock_code': item,
                        'confidence': row['confidence'],
                        'interest': row['interest'],
                        'support': row['support'],
                        'description': self.item_descriptions.get(item, 'Unknown')
                    })
                if len(consequents) >= top_n:
                    break
            if len(consequents) >= top_n:
                break

        return consequents

    def get_rules_for_items(self, stock_codes, top_n_per_item=3):
        all_consequents = []
        seen = set()

        for code in stock_codes:
            results = self.get_consequents(code, top_n=top_n_per_item)
            for r in results:
                if r['stock_code'] not in seen:
                    seen.add(r['stock_code'])
                    all_consequents.append(r)

        all_consequents.sort(key=lambda x: x['interest'], reverse=True)
        return all_consequents

    def recommend_from_history(self, purchased_items, top_k=10):
        rule_results = self.get_rules_for_items(purchased_items, top_n_per_item=3)

        recommendations = []
        seen = set(purchased_items)
        for r in rule_results:
            if r['stock_code'] not in seen:
                seen.add(r['stock_code'])
                recommendations.append(r['stock_code'])
            if len(recommendations) >= top_k:
                break

        return recommendations

    def print_top_rules(self, n=10):
        if self.rules_df is None or self.rules_df.empty:
            print("No rules available.")
            return

        print(f"\nTop {n} Association Rules by Interest:")
        print(f"{'='*90}")
        for i, (_, row) in enumerate(self.rules_df.head(n).iterrows()):
            ant = ', '.join([self.item_descriptions.get(a, a) for a in row['antecedents']])
            cons = ', '.join([self.item_descriptions.get(c, c) for c in row['consequents']])
            print(f"{i+1}. {ant}")
            print(f"   → {cons}")
            print(f"   (support={row['support']:.4f}, confidence={row['confidence']:.4f}, interest={row['interest']:.4f})")
            print()

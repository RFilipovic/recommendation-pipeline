class HybridRecommender:
    def __init__(self, cf_recommender, rule_miner, min_confidence=0.3, min_interest=0.0):
        self.cf = cf_recommender
        self.rule_miner = rule_miner
        self.min_confidence = min_confidence
        self.min_interest = min_interest
        self.utility_matrix = cf_recommender.utility_matrix
        self.user_mapping = cf_recommender.user_mapping
        self.item_mapping = cf_recommender.item_mapping
        self.reverse_item_mapping = cf_recommender.reverse_item_mapping

    def recommend(self, user_id, cf_k=5, rules_per_item=3):
        if user_id not in self.user_mapping:
            return [], []

        user_idx = self.user_mapping[user_id]
        purchased_idx = self.utility_matrix[user_idx].nonzero()[1]
        purchased = set(self.reverse_item_mapping[idx] for idx in purchased_idx)

        cf_recs = [item for item in self.cf.predict(user_id, top_k=cf_k) if item not in purchased]

        rule_recs = []
        for cf_item in cf_recs:
            consequents = self.rule_miner.get_consequents(cf_item, top_n=rules_per_item)
            for c in consequents:
                if (c['stock_code'] not in purchased
                        and c['confidence'] >= self.min_confidence
                        and c['interest'] >= self.min_interest
                        and c['stock_code'] not in cf_recs
                        and c['stock_code'] not in [r['stock_code'] for r in rule_recs]):
                    rule_recs.append({**c, 'triggered_by': cf_item})

        return cf_recs, rule_recs

# E-Commerce Recommendation Engine: Collaborative Filtering + Association Rules

## 1. Project Overview

This project builds a **hybrid e-commerce recommendation engine** that combines two complementary techniques:

- **Item-Item Collaborative Filtering**: Recommends products based on what similar users purchased — captures **personalized taste patterns** across transactions.
- **Association Rules (Apriori)**: Recommends products frequently bought together in the same basket — captures **co-purchase intent** within a single transaction.

The **Hybrid Recommender** merges both signals into a single ranked list, producing recommendations that are both personalized and contextually relevant.

## 2. Dataset

**UCI Online Retail Dataset** — ~500K transactions from a UK-based online retailer (Dec 2010 – Dec 2011). Each row represents a purchased item within an invoice, including CustomerID, StockCode, Description, Quantity, and InvoiceDate.

## 3. Two Signals, One Engine

| | Item-Item CF | Association Rules |
|---|---|---|
| **Question answered** | "What will this user likely buy?" | "What do people buy *with* this item?" |
| **Signal source** | User purchase vectors across all transactions | Items co-occurring in the same basket (InvoiceNo) |
| **Scope** | Long-term user preferences | Impulse add-ons and complements |
| **Personalization** | High — tailored to each user | Low — same rules apply to everyone |

These signals are complementary: CF captures *who you are*, association rules capture *what goes together*. The hybrid leverages both.

## 4. Pipeline

1. **Data Cleaning**: Drop NaN CustomerIDs, remove returns (Quantity ≤ 0), normalize descriptions.
2. **Utility Matrix**: Users × Items sparse matrix of purchase quantities.
3. **Apriori**: Mine frequent itemsets from baskets (grouped by InvoiceNo), extract high-lift rules.
4. **Item-Item CF**: Compute item cosine similarity, score items for each user.
5. **Hybrid**: For each user, get CF top-K picks, then expand each pick with association rule consequents. Score = weighted blend of normalized CF rank and rule confidence × lift.
6. **Evaluation**: Chronological train/test split (last 30 days = test). Compare Precision@K and Recall@K for CF-only, Rules-only, and Hybrid.

## 5. Association Rules — Key Concepts

- **Support**: Fraction of baskets containing the itemset. Filters out rare combinations.
- **Confidence**: P(consequent | antecedent). Measures reliability of the rule.
- **Lift**: P(consequent | antecedent) / P(consequent). Values > 1 indicate the antecedent *increases* the likelihood of the consequent beyond baseline. High-lift rules are the most interesting — they reveal non-obvious co-purchase patterns.

## 6. Hybrid Scoring

For each candidate item:

```
score = cf_weight × (normalized_CF_rank) + rule_weight × (normalized_confidence × lift)
```

- Items appearing in both CF and rule outputs get both signals boosted.
- Items only from CF still appear (personalized picks).
- Items only from rules still appear ("frequently bought together" discoveries).

Default weights: 70% CF, 30% rules — personalization drives the list, but rules surface complementary items CF might miss.

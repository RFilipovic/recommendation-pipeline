# E-Commerce Recommendation Engine: Collaborative Filtering + Association Rules

## 1. Project Overview

This project builds a **hybrid e-commerce recommendation engine** that combines two complementary techniques:

- **Item-Item Collaborative Filtering**: Recommends products based on what similar users purchased — captures **personalized taste patterns** across transactions.
- **Association Rules (Apriori)**: Recommends products frequently bought together in the same basket — captures **co-purchase intent** within a single transaction.

## 2. Dataset

**UCI Online Retail Dataset** — ~500K transactions from a UK-based online retailer (Dec 2010 – Dec 2011). Each row represents a purchased item within an invoice, including CustomerID, StockCode, Description, Quantity, and InvoiceDate.

## 3. Two Signals, One Engine

| | Item-Item CF | Association Rules |
|---|---|---|
| **Question answered** | "What will this user likely buy?" | "What do people buy *with* this item?" |
| **Signal source** | User purchase vectors across all transactions | Items co-occurring in the same basket (InvoiceNo) |
| **Scope** | Long-term user preferences | Impulse add-ons and complements |
| **Personalization** | High — tailored to each user | Low — same rules apply to everyone |

## 4. Pipeline

1. **Data Cleaning**: Drop NaN CustomerIDs, remove returns (Quantity ≤ 0), normalize descriptions.
2. **Utility Matrix**: Users × Items sparse matrix of purchase quantities.
3. **Apriori**: Mine frequent itemsets from baskets (grouped by InvoiceNo), extract high-interest rules.
4. **Item-Item CF**: Compute item cosine similarity, score items for each user.
5. **Hybrid**: For each user, get CF top-K picks, then expand each pick with association rule consequents if interest/confidence thresholds are met.

## 5. Association Rules — Key Concepts

- **Support**: Fraction of baskets containing the itemset. Filters out rare combinations.
- **Confidence**: P(consequent | antecedent). Measures reliability of the rule.
- **Interest (Lift)**: P(consequent | antecedent) / P(consequent). Values > 1 indicate the antecedent *increases* the likelihood of the consequent beyond baseline. High-interest rules are the most interesting — they reveal non-obvious co-purchase patterns.

## 6. Hybrid Logic

For a given user:
1. CF generates top-K personalized item recommendations.
2. For each CF-recommended item, look up association rules where that item is the antecedent.
3. Keep rule consequents that pass `confidence ≥ threshold` and `interest ≥ threshold`.
4. Output both lists: CF recommendations + rule-triggered add-ons.

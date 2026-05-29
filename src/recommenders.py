from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds
import pandas as pd

class ContentBasedRecommender:
    def __init__(self, df, n_components=100):
        """
        Initializes the ContentBasedRecommender.
        
        Args:
            df (pd.DataFrame): The cleaned dataframe containing StockCode and Description.
            n_components (int): Number of dimensions for LSI (TruncatedSVD).
        """
        self.df = df
        self.n_components = n_components
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.svd = TruncatedSVD(n_components=n_components)
        
        self.item_profiles = None
        self.user_profiles = None
        self.utility_matrix = None
        self.user_mapping = None
        self.item_mapping = None
        self.reverse_item_mapping = None
        
    def fit(self):
        """
        Phase 1 & 2: TF-IDF Vectorization and LSI Dimensionality Reduction.
        Converts product descriptions to numerical vectors and reduces dimensions.
        """
        # Extract unique StockCode + Description pairs
        item_info = self.df.drop_duplicates('StockCode')[['StockCode', 'Description']]
        
        # Fit TF-IDF on all descriptions
        tfidf_matrix = self.tfidf.fit_transform(item_info['Description'])
        
        # Apply TruncatedSVD (LSI)
        self.item_profiles_dense = self.svd.fit_transform(tfidf_matrix)
        
        # Create a mapping from StockCode to its LSI vector index
        self.item_profile_mapping = {stock: idx for idx, stock in enumerate(item_info['StockCode'])}
        
    def build_user_profiles(self, utility_matrix, user_mapping, item_mapping):
        """
        Phase 3: User Profile Construction.
        Builds user preference vectors by computing the weighted average 
        of item LSI vectors (weighted by Quantity).
        
        Args:
            utility_matrix (scipy.sparse.csr_matrix): The user-item utility matrix.
            user_mapping (dict): Mapping from CustomerID to utility matrix row index.
            item_mapping (dict): Mapping from StockCode to utility matrix column index.
        """
        self.utility_matrix = utility_matrix
        self.user_mapping = user_mapping
        self.item_mapping = item_mapping
        self.reverse_item_mapping = {idx: stock for stock, idx in item_mapping.items()}
        
        # Align item profiles with the utility matrix columns
        # The utility matrix columns are ordered by item_mapping keys
        ordered_stockcodes = list(item_mapping.keys())
        ordered_indices = [self.item_profile_mapping[stock] for stock in ordered_stockcodes]
        
        # Reorder item profiles to match utility matrix columns
        self.item_profiles = self.item_profiles_dense[ordered_indices]
        
        # Compute weighted sum of item vectors for each user
        # utility_matrix: (n_users, n_items), item_profiles: (n_items, n_components)
        # Result: (n_users, n_components)
        weighted_sums = utility_matrix @ self.item_profiles
        
        # Compute the sum of quantities for each user to get the average
        user_sums = np.array(utility_matrix.sum(axis=1)).flatten()
        user_sums[user_sums == 0] = 1  # Avoid division by zero
        
        # Calculate weighted average
        self.user_profiles = weighted_sums / user_sums[:, np.newaxis]
        
    def recommend(self, user_id, k=10):
        """
        Phase 4: Cosine Similarity & Prediction.
        Generates top-K recommendations for a given user.
        
        Args:
            user_id (str/int): The CustomerID to generate recommendations for.
            k (int): The number of recommendations to return.
            
        Returns:
            list: Top-K recommended StockCodes.
        """
        if user_id not in self.user_mapping:
            return []
            
        user_idx = self.user_mapping[user_id]
        user_profile = self.user_profiles[user_idx]
        
        # Compute cosine similarity between user profile and all item profiles
        similarities = cosine_similarity(user_profile.reshape(1, -1), self.item_profiles).flatten()
        
        # Get items already purchased by the user
        purchased_items = self.utility_matrix[user_idx].nonzero()[1]
        
        # Filter out already purchased items by setting their similarity to -1
        similarities[purchased_items] = -1
        
        # Get top-K item indices
        top_k_indices = np.argsort(similarities)[::-1][:k]
        
        # Map indices back to StockCodes
        recommended_stockcodes = [self.reverse_item_mapping[idx] for idx in top_k_indices]
        
        return recommended_stockcodes
        
    def _cosine_similarity(self, vec1, vec2):
        """
        Computes cosine similarity between two vectors.
        
        Args:
            vec1 (np.array): First vector.
            vec2 (np.array): Second vector.
            
        Returns:
            float: Cosine similarity score.
        """
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return dot_product / (norm1 * norm2)

class CollaborativeFilteringRecommender:
    def __init__(self, mode='item'):
        self.mode = mode
        
    def fit(self, utility_matrix):
        """
        Steps to implement:
        1. If mode == 'item', compute item-item cosine similarity matrix.
        2. If mode == 'user', compute user-user cosine similarity matrix.
        3. Store the similarity matrix and utility matrix.
        """
        raise NotImplementedError

    def predict(self, user_id, top_k=10):
        """
        Steps to implement:
        1. If mode == 'item': 
           Get items purchased by user. For each purchased item, get similar items.
           Aggregate and weight similarities by user's purchase quantity.
        2. If mode == 'user':
           Get similar users. Aggregate their purchases, weighted by user similarity.
        3. Filter out known purchases.
        4. Return top_k item indices.
        """
        raise NotImplementedError

class LatentFactorRecommender:
    def __init__(self, n_factors=50):
        self.n_factors = n_factors

    def fit(self, utility_matrix):
        """
        Steps to implement:
        1. Apply SVD or NMF to the utility matrix.
           e.g., U, sigma, Vt = svds(utility_matrix, k=self.n_factors)
        2. Convert sigma to a diagonal matrix.
        3. Compute predicted matrix: predictions = np.dot(np.dot(U, sigma), Vt).
        4. Store the predictions matrix.
        """
        raise NotImplementedError

    def predict(self, user_id, top_k=10):
        """
        Steps to implement:
        1. Get the row corresponding to user_id from the predictions matrix.
        2. Filter out items the user has already purchased.
        3. Return top_k item indices based on predicted value.
        """
        raise NotImplementedError

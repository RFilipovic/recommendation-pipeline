from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds
import pandas as pd

class ContentBasedRecommender:
    def __init__(self, df, n_components=100):
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

        # Extract unique StockCode + Description pairs
        item_info = self.df.drop_duplicates('StockCode')[['StockCode', 'Description']]
        
        # Fit TF-IDF on all descriptions
        tfidf_matrix = self.tfidf.fit_transform(item_info['Description'])
        
        # Apply TruncatedSVD (LSI)
        self.item_profiles_dense = self.svd.fit_transform(tfidf_matrix)
        
        # Create a mapping from StockCode to its LSI vector index
        self.item_profile_mapping = {stock: idx for idx, stock in enumerate(item_info['StockCode'])}
        
    def build_user_profiles(self, utility_matrix, user_mapping, item_mapping):

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
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return dot_product / (norm1 * norm2)

class CollaborativeFilteringRecommender:
    def __init__(self, mode='item'):
        self.mode = mode
        self.utility_matrix = None
        self.user_mapping = None
        self.item_mapping = None
        self.reverse_user_mapping = None
        self.reverse_item_mapping = None
        self.sim_matrix = None
        
    def fit(self, utility_matrix, user_mapping, item_mapping):

        self.utility_matrix = utility_matrix
        self.user_mapping = user_mapping
        self.item_mapping = item_mapping
        self.reverse_user_mapping = {idx: user for user, idx in user_mapping.items()}
        self.reverse_item_mapping = {idx: stock for stock, idx in item_mapping.items()}
        
        if self.mode == 'item':
            # Compute item-item cosine similarity matrix
            # Transpose the utility matrix so items are rows
            self.sim_matrix = cosine_similarity(utility_matrix.T, dense_output=False)
        elif self.mode == 'user':
            # Compute user-user cosine similarity matrix
            self.sim_matrix = cosine_similarity(utility_matrix, dense_output=False)
        else:
            raise ValueError("Mode must be either 'item' or 'user'")
            
    def predict(self, user_id, top_k=10):

        if user_id not in self.user_mapping:
            return []
            
        user_idx = self.user_mapping[user_id]
        purchased_items = self.utility_matrix[user_idx].nonzero()[1]
        
        if self.mode == 'item':
            # Item-Item Collaborative Filtering
            # Get the user's purchase vector (quantities)
            user_vector = np.array(self.utility_matrix[user_idx].todense()).flatten()
            
            # Compute scores: dot product of user vector and item similarity matrix
            # user_vector (1, n_items) @ sim_matrix (n_items, n_items) -> (1, n_items)
            scores = user_vector @ self.sim_matrix
            
            # If sim_matrix is sparse, the result might be a sparse matrix or np.matrix
            if isinstance(scores, (sp.spmatrix, np.matrix)):
                scores = np.asarray(scores).flatten()
                
        else:
            # User-User Collaborative Filtering
            # Get the user similarity vector
            user_similarities = np.array(self.sim_matrix[user_idx].todense()).flatten()
            
            # Compute scores: dot product of user similarities and utility matrix
            # user_similarities (1, n_users) @ utility_matrix (n_users, n_items) -> (1, n_items)
            scores = user_similarities @ self.utility_matrix
            
            if isinstance(scores, (sp.spmatrix, np.matrix)):
                scores = np.asarray(scores).flatten()
        
        # Filter out already purchased items by setting their score to -1
        scores[purchased_items] = -1
        
        # Get top-K item indices
        top_k_indices = np.argsort(scores)[::-1][:top_k]
        
        # Map indices back to StockCodes
        recommended_stockcodes = [self.reverse_item_mapping[idx] for idx in top_k_indices]
        
        return recommended_stockcodes

class LatentFactorRecommender:
    def __init__(self, n_factors=50):
        self.n_factors = n_factors
        self.utility_matrix = None
        self.user_mapping = None
        self.item_mapping = None
        self.reverse_user_mapping = None
        self.reverse_item_mapping = None
        self.predicted_ratings = None

    def fit(self, utility_matrix, user_mapping, item_mapping):
        """
        Applies SVD to the utility matrix to find latent factors and predict missing ratings.
        
        Args:
            utility_matrix (scipy.sparse.csr_matrix): The user-item utility matrix.
            user_mapping (dict): Mapping from CustomerID to utility matrix row index.
            item_mapping (dict): Mapping from StockCode to utility matrix column index.
        """
        self.utility_matrix = utility_matrix
        self.user_mapping = user_mapping
        self.item_mapping = item_mapping
        self.reverse_user_mapping = {idx: user for user, idx in user_mapping.items()}
        self.reverse_item_mapping = {idx: stock for stock, idx in item_mapping.items()}
        
        # Ensure n_factors is less than min dimension of utility matrix
        k = min(self.n_factors, min(utility_matrix.shape) - 1)
        
        # Apply SVD
        U, sigma, Vt = svds(utility_matrix, k=k)
        
        # Convert sigma to diagonal matrix
        sigma_diag = np.diag(sigma)
        
        # Compute predicted matrix
        self.predicted_ratings = np.dot(np.dot(U, sigma_diag), Vt)

    def predict(self, user_id, top_k=10):
        """
        Generates top-K recommendations for a given user based on predicted ratings.
        
        Args:
            user_id (str/int): The CustomerID to generate recommendations for.
            top_k (int): The number of recommendations to return.
            
        Returns:
            list: Top-K recommended StockCodes.
        """
        if user_id not in self.user_mapping:
            return []
            
        user_idx = self.user_mapping[user_id]
        purchased_items = self.utility_matrix[user_idx].nonzero()[1]
        
        # Get predicted scores for the user
        scores = self.predicted_ratings[user_idx, :]
        
        # Filter out already purchased items by setting their score to -1
        scores[purchased_items] = -1
        
        # Get top-K item indices
        top_k_indices = np.argsort(scores)[::-1][:top_k]
        
        # Map indices back to StockCodes
        recommended_stockcodes = [self.reverse_item_mapping[idx] for idx in top_k_indices]
        
        return recommended_stockcodes

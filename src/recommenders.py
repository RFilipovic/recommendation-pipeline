from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import scipy.sparse as sp

class CollaborativeFilteringRecommender:
    def __init__(self, mode='item'):
        self.mode = mode
        self.utility_matrix = None
        self.user_mapping = None
        self.item_mapping = None
        self.reverse_item_mapping = None
        self.sim_matrix = None
        
    def fit(self, utility_matrix, user_mapping, item_mapping):

        self.utility_matrix = utility_matrix
        self.user_mapping = user_mapping
        self.item_mapping = item_mapping
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

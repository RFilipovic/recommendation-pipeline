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
            self.sim_matrix = cosine_similarity(utility_matrix.T, dense_output=False)
        elif self.mode == 'user':
            self.sim_matrix = cosine_similarity(utility_matrix, dense_output=False)
        else:
            raise ValueError("Mode must be either 'item' or 'user'")
            
    def predict(self, user_id, top_k=10):

        if user_id not in self.user_mapping:
            return []
            
        user_idx = self.user_mapping[user_id]
        purchased_items = self.utility_matrix[user_idx].nonzero()[1]
        
        if self.mode == 'item':

            user_vector = np.array(self.utility_matrix[user_idx].todense()).flatten()
            
            scores = user_vector @ self.sim_matrix

            if isinstance(scores, (sp.spmatrix, np.matrix)):
                scores = np.asarray(scores).flatten()
                
        else:
            user_similarities = np.array(self.sim_matrix[user_idx].todense()).flatten()
            
            scores = user_similarities @ self.utility_matrix
            
            if isinstance(scores, (sp.spmatrix, np.matrix)):
                scores = np.asarray(scores).flatten()
        
        scores[purchased_items] = -1
        
        top_k_indices = np.argsort(scores)[::-1][:top_k]
        
        recommended_stockcodes = [self.reverse_item_mapping[idx] for idx in top_k_indices]
        
        return recommended_stockcodes

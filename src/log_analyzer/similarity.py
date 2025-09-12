"""Similarity analysis using cosine similarity."""

import numpy as np
from typing import List, Dict, Tuple
import math


class SimilarityAnalyzer:
    """Analyze similarity between embeddings using cosine similarity."""
    
    @staticmethod
    def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score (0-1, where 1 is identical)
        """
        if len(vec1) != len(vec2):
            raise ValueError("Vectors must have the same length")
        
        # Convert to numpy arrays for efficient computation
        a = np.array(vec1)
        b = np.array(vec2)
        
        # Calculate cosine similarity
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        similarity = dot_product / (norm_a * norm_b)
        
        # Ensure result is between 0 and 1
        return max(0.0, min(1.0, similarity))
    
    def analyze_file_similarity(self, 
                              new_embedding: List[float], 
                              baseline_embeddings: Dict[str, List[float]],
                              threshold: float = 0.8) -> Dict[str, any]:
        """Analyze similarity of a new file against baseline embeddings.
        
        Args:
            new_embedding: Embedding of the new file
            baseline_embeddings: Dictionary of baseline file embeddings
            threshold: Similarity threshold for divergence detection
            
        Returns:
            Analysis results including similarities and divergence status
        """
        if not baseline_embeddings:
            return {
                "status": "no_baseline",
                "message": "No baseline embeddings found",
                "similarities": {},
                "is_divergent": True
            }
        
        similarities = {}
        for file_path, baseline_embedding in baseline_embeddings.items():
            try:
                similarity = self.cosine_similarity(new_embedding, baseline_embedding)
                similarities[file_path] = similarity
            except ValueError as e:
                similarities[file_path] = {"error": str(e)}
        
        # Calculate statistics
        valid_similarities = [s for s in similarities.values() if isinstance(s, float)]
        
        if not valid_similarities:
            return {
                "status": "error",
                "message": "No valid similarity calculations",
                "similarities": similarities,
                "is_divergent": True
            }
        
        max_similarity = max(valid_similarities)
        avg_similarity = sum(valid_similarities) / len(valid_similarities)
        min_similarity = min(valid_similarities)
        
        # Determine divergence based on maximum similarity
        is_divergent = max_similarity < threshold
        
        return {
            "status": "success",
            "similarities": similarities,
            "max_similarity": max_similarity,
            "avg_similarity": avg_similarity,
            "min_similarity": min_similarity,
            "threshold": threshold,
            "is_divergent": is_divergent,
            "most_similar_file": max(similarities.items(), key=lambda x: x[1] if isinstance(x[1], float) else 0)[0],
            "divergence_score": 1.0 - max_similarity
        }
    
    def get_similarity_matrix(self, embeddings: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
        """Calculate similarity matrix for all pairs of embeddings.
        
        Args:
            embeddings: Dictionary of embeddings
            
        Returns:
            Matrix of pairwise similarities
        """
        files = list(embeddings.keys())
        matrix = {}
        
        for file1 in files:
            matrix[file1] = {}
            for file2 in files:
                if file1 == file2:
                    matrix[file1][file2] = 1.0
                else:
                    try:
                        similarity = self.cosine_similarity(embeddings[file1], embeddings[file2])
                        matrix[file1][file2] = similarity
                    except ValueError:
                        matrix[file1][file2] = 0.0
        
        return matrix
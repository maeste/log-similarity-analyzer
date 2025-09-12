"""Enhanced similarity analysis using ChromaDB's native capabilities."""

import numpy as np
from typing import List, Dict, Tuple, Optional
from log_analyzer.chroma_storage import ChromaEmbeddingStorage


class ChromaSimilarityAnalyzer:
    """Enhanced similarity analyzer using ChromaDB's native similarity search."""
    
    def __init__(self, storage: ChromaEmbeddingStorage):
        """Initialize analyzer with ChromaDB storage.
        
        Args:
            storage: ChromaEmbeddingStorage instance
        """
        self.storage = storage
    
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
                              threshold: float = 0.8,
                              n_similar: int = 5,
                              log_type: Optional[str] = None,
                              source_system: Optional[str] = None) -> Dict[str, any]:
        """Analyze similarity using ChromaDB's native similarity search.
        
        Args:
            new_embedding: Embedding of the new file
            threshold: Similarity threshold for divergence detection
            n_similar: Number of most similar files to retrieve
            log_type: Filter baselines by log type (optional)
            source_system: Filter baselines by source system (optional)
            
        Returns:
            Enhanced analysis results with ChromaDB features
        """
        # Use ChromaDB's native similarity search
        similar_files = self.storage.find_similar(
            embedding=new_embedding,
            n_results=n_similar,
            log_type=log_type,
            source_system=source_system
        )
        
        if not similar_files:
            return {
                "status": "no_baseline",
                "message": "No baseline embeddings found",
                "similarities": {},
                "is_divergent": True,
                "search_filters": {
                    "log_type": log_type,
                    "source_system": source_system
                }
            }
        
        # Process results
        similarities = {}
        metadata_info = {}
        
        for file_path, similarity, metadata in similar_files:
            similarities[file_path] = similarity
            metadata_info[file_path] = {
                "log_type": metadata.get("log_type", "unknown"),
                "source_system": metadata.get("source_system", "unknown"),
                "timestamp": metadata.get("timestamp", "unknown"),
                "file_size": metadata.get("file_size", 0)
            }
        
        # Calculate statistics
        valid_similarities = list(similarities.values())
        max_similarity = max(valid_similarities)
        avg_similarity = sum(valid_similarities) / len(valid_similarities)
        min_similarity = min(valid_similarities)
        
        # Find most similar file
        most_similar_file = max(similarities.items(), key=lambda x: x[1])[0]
        
        # Determine divergence
        is_divergent = max_similarity < threshold
        
        return {
            "status": "success",
            "similarities": similarities,
            "metadata": metadata_info,
            "max_similarity": max_similarity,
            "avg_similarity": avg_similarity,
            "min_similarity": min_similarity,
            "threshold": threshold,
            "is_divergent": is_divergent,
            "most_similar_file": most_similar_file,
            "most_similar_metadata": metadata_info[most_similar_file],
            "divergence_score": 1.0 - max_similarity,
            "search_filters": {
                "log_type": log_type,
                "source_system": source_system
            },
            "total_compared": len(similar_files)
        }
    
    def get_similarity_matrix(self, 
                             log_type: Optional[str] = None,
                             source_system: Optional[str] = None) -> Dict[str, Dict[str, float]]:
        """Calculate similarity matrix for stored embeddings with filtering.
        
        Args:
            log_type: Filter by log type (optional)
            source_system: Filter by source system (optional)
            
        Returns:
            Matrix of pairwise similarities
        """
        embeddings = self.storage.load_all_embeddings(
            log_type=log_type,
            source_system=source_system
        )
        
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
    
    def find_outliers(self, 
                     threshold: float = 0.7,
                     log_type: Optional[str] = None,
                     source_system: Optional[str] = None) -> List[Dict]:
        """Find files that are outliers (low similarity to all others).
        
        Args:
            threshold: Similarity threshold below which files are considered outliers
            log_type: Filter by log type (optional)
            source_system: Filter by source system (optional)
            
        Returns:
            List of outlier file information
        """
        embeddings = self.storage.load_all_embeddings(
            log_type=log_type,
            source_system=source_system
        )
        
        if len(embeddings) < 2:
            return []
        
        outliers = []
        
        for file_path, embedding in embeddings.items():
            # Find similarity to all other files
            similarities = []
            for other_path, other_embedding in embeddings.items():
                if file_path != other_path:
                    try:
                        similarity = self.cosine_similarity(embedding, other_embedding)
                        similarities.append(similarity)
                    except ValueError:
                        continue
            
            if similarities:
                max_similarity = max(similarities)
                avg_similarity = sum(similarities) / len(similarities)
                
                if max_similarity < threshold:
                    # Get metadata
                    metadata = self.storage.list_files(log_type=log_type, source_system=source_system)
                    file_metadata = next((m for m in metadata if m.get('file_path') == file_path), {})
                    
                    outliers.append({
                        "file_path": file_path,
                        "max_similarity": max_similarity,
                        "avg_similarity": avg_similarity,
                        "divergence_score": 1.0 - max_similarity,
                        "metadata": file_metadata
                    })
        
        # Sort by divergence score (highest first)
        outliers.sort(key=lambda x: x["divergence_score"], reverse=True)
        return outliers
    
    def cluster_analysis(self, 
                        similarity_threshold: float = 0.8,
                        log_type: Optional[str] = None,
                        source_system: Optional[str] = None) -> Dict[str, List[str]]:
        """Perform simple clustering based on similarity threshold.
        
        Args:
            similarity_threshold: Threshold for grouping files together
            log_type: Filter by log type (optional)
            source_system: Filter by source system (optional)
            
        Returns:
            Dictionary mapping cluster IDs to lists of file paths
        """
        embeddings = self.storage.load_all_embeddings(
            log_type=log_type,
            source_system=source_system
        )
        
        if len(embeddings) < 2:
            return {"cluster_0": list(embeddings.keys())}
        
        files = list(embeddings.keys())
        clusters = {}
        cluster_id = 0
        assigned = set()
        
        for file_path in files:
            if file_path in assigned:
                continue
            
            # Start new cluster
            current_cluster = [file_path]
            assigned.add(file_path)
            
            # Find similar files
            for other_path in files:
                if other_path in assigned:
                    continue
                
                try:
                    similarity = self.cosine_similarity(
                        embeddings[file_path], 
                        embeddings[other_path]
                    )
                    if similarity >= similarity_threshold:
                        current_cluster.append(other_path)
                        assigned.add(other_path)
                except ValueError:
                    continue
            
            clusters[f"cluster_{cluster_id}"] = current_cluster
            cluster_id += 1
        
        return clusters
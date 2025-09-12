"""Simple storage system for embeddings."""

import json
import os
from typing import Dict, List, Optional
from datetime import datetime


class EmbeddingStorage:
    """Simple JSON-based storage for embeddings."""
    
    def __init__(self, storage_file: str = "embeddings.json"):
        """Initialize storage.
        
        Args:
            storage_file: Path to the storage file
        """
        self.storage_file = storage_file
    
    def save_embedding(self, file_path: str, embedding: List[float]) -> None:
        """Save an embedding for a file.
        
        Args:
            file_path: Path to the original file
            embedding: The embedding vector
        """
        data = self._load_data()
        
        data[file_path] = {
            "embedding": embedding,
            "timestamp": datetime.now().isoformat(),
            "file_path": file_path
        }
        
        self._save_data(data)
    
    def load_embedding(self, file_path: str) -> Optional[List[float]]:
        """Load embedding for a specific file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Embedding vector or None if not found
        """
        data = self._load_data()
        entry = data.get(file_path)
        return entry["embedding"] if entry else None
    
    def load_all_embeddings(self) -> Dict[str, List[float]]:
        """Load all stored embeddings.
        
        Returns:
            Dictionary mapping file paths to embeddings
        """
        data = self._load_data()
        return {path: entry["embedding"] for path, entry in data.items()}
    
    def list_files(self) -> List[str]:
        """List all files with stored embeddings.
        
        Returns:
            List of file paths
        """
        data = self._load_data()
        return list(data.keys())
    
    def remove_embedding(self, file_path: str) -> bool:
        """Remove embedding for a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if removed, False if not found
        """
        data = self._load_data()
        if file_path in data:
            del data[file_path]
            self._save_data(data)
            return True
        return False
    
    def clear_all(self) -> None:
        """Clear all stored embeddings."""
        self._save_data({})
    
    def _load_data(self) -> Dict:
        """Load data from storage file."""
        if not os.path.exists(self.storage_file):
            return {}
        
        try:
            with open(self.storage_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    
    def _save_data(self, data: Dict) -> None:
        """Save data to storage file."""
        try:
            with open(self.storage_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except IOError as e:
            print(f"Error saving to {self.storage_file}: {e}")
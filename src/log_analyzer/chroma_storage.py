"""ChromaDB-based storage system for embeddings with enhanced capabilities."""

import chromadb
from chromadb.config import Settings
import os
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import hashlib


class ChromaEmbeddingStorage:
    """ChromaDB-based storage for embeddings with collections and metadata."""
    
    def __init__(self, 
                 db_path: str = "./chroma_db", 
                 collection_name: str = "log_embeddings"):
        """Initialize ChromaDB storage.
        
        Args:
            db_path: Path to the ChromaDB database directory
            collection_name: Name of the collection to store embeddings
        """
        self.db_path = db_path
        self.collection_name = collection_name
        
        # Initialize ChromaDB client with persistent storage
        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
        except Exception:
            # Collection doesn't exist, create it with cosine similarity
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": "Log file embeddings for similarity analysis", "hnsw:space": "cosine"}
            )
    
    def _generate_id(self, file_path: str) -> str:
        """Generate unique ID for a file path.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Unique identifier for the file
        """
        # Use file path hash as ID to ensure uniqueness
        return hashlib.md5(file_path.encode()).hexdigest()
    
    def save_embedding(self, 
                      file_path: str, 
                      embedding: List[float],
                      log_type: str = "application",
                      source_system: Optional[str] = None) -> None:
        """Save an embedding for a file with metadata.
        
        Args:
            file_path: Path to the original file
            embedding: The embedding vector
            log_type: Type of log (application, security, system, etc.)
            source_system: Source system name (optional)
        """
        # Prepare metadata
        metadata = {
            "file_path": file_path,
            "timestamp": datetime.now().isoformat(),
            "log_type": log_type,
            "file_size": self._get_file_size(file_path),
        }
        
        if source_system:
            metadata["source_system"] = source_system
        
        # Generate unique ID
        doc_id = self._generate_id(file_path)
        
        # Read file content for document storage
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except (IOError, UnicodeDecodeError):
            content = f"[Error reading file: {file_path}]"
        
        # Add to ChromaDB collection
        self.collection.upsert(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[content],
            metadatas=[metadata]
        )
    
    def load_embedding(self, file_path: str) -> Optional[List[float]]:
        """Load embedding for a specific file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Embedding vector or None if not found
        """
        doc_id = self._generate_id(file_path)
        
        try:
            result = self.collection.get(
                ids=[doc_id],
                include=['embeddings']
            )
            
            if result['embeddings']:
                return result['embeddings'][0]
            return None
        except Exception:
            return None
    
    def load_all_embeddings(self, 
                           log_type: Optional[str] = None,
                           source_system: Optional[str] = None) -> Dict[str, List[float]]:
        """Load all stored embeddings with optional filtering.
        
        Args:
            log_type: Filter by log type (optional)
            source_system: Filter by source system (optional)
            
        Returns:
            Dictionary mapping file paths to embeddings
        """
        # Build where clause for filtering
        where = {}
        if log_type:
            where["log_type"] = log_type
        if source_system:
            where["source_system"] = source_system
        
        try:
            result = self.collection.get(
                where=where if where else None,
                include=['embeddings', 'metadatas']
            )
            
            embeddings = {}
            for i, metadata in enumerate(result['metadatas']):
                file_path = metadata['file_path']
                embeddings[file_path] = result['embeddings'][i]
            
            return embeddings
        except Exception:
            return {}
    
    def find_similar(self, 
                    embedding: List[float],
                    n_results: int = 5,
                    log_type: Optional[str] = None,
                    source_system: Optional[str] = None) -> List[Tuple[str, float, Dict]]:
        """Find most similar embeddings using ChromaDB's native similarity search.
        
        Args:
            embedding: Query embedding vector
            n_results: Number of similar results to return
            log_type: Filter by log type (optional)
            source_system: Filter by source system (optional)
            
        Returns:
            List of tuples (file_path, similarity_score, metadata)
        """
        # Build where clause for filtering
        where = {}
        if log_type:
            where["log_type"] = log_type
        if source_system:
            where["source_system"] = source_system
        
        try:
            results = self.collection.query(
                query_embeddings=[embedding],
                n_results=n_results,
                where=where if where else None,
                include=['metadatas', 'distances']
            )
            
            similar_files = []
            for i, metadata in enumerate(results['metadatas'][0]):
                file_path = metadata['file_path']
                # ChromaDB distance values - need to determine actual range
                distance = results['distances'][0][i]
                
                # For very small distances (< 0.1), treat as highly similar
                if distance < 0.1:
                    similarity = 1.0 - distance
                # For moderate distances (0.1 to 1.0), scale appropriately
                elif distance <= 1.0:
                    similarity = 1.0 - distance
                # For larger distances, use exponential decay to avoid negative similarities
                else:
                    similarity = max(0.001, 1.0 / (1.0 + distance))
                
                similar_files.append((file_path, similarity, metadata))
            
            return similar_files
        except Exception:
            return []
    
    def list_files(self, 
                  log_type: Optional[str] = None,
                  source_system: Optional[str] = None) -> List[Dict]:
        """List all files with stored embeddings and their metadata.
        
        Args:
            log_type: Filter by log type (optional)
            source_system: Filter by source system (optional)
            
        Returns:
            List of metadata dictionaries
        """
        # Build where clause for filtering
        where = {}
        if log_type:
            where["log_type"] = log_type
        if source_system:
            where["source_system"] = source_system
        
        try:
            result = self.collection.get(
                where=where if where else None,
                include=['metadatas']
            )
            return result['metadatas']
        except Exception:
            return []
    
    def remove_embedding(self, file_path: str) -> bool:
        """Remove embedding for a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if removed, False if not found
        """
        doc_id = self._generate_id(file_path)
        
        try:
            self.collection.delete(ids=[doc_id])
            return True
        except Exception:
            return False
    
    def clear_all(self, 
                 log_type: Optional[str] = None,
                 source_system: Optional[str] = None) -> None:
        """Clear embeddings with optional filtering.
        
        Args:
            log_type: Only clear embeddings of this log type (optional)
            source_system: Only clear embeddings from this source system (optional)
        """
        if log_type or source_system:
            # Selective deletion
            where = {}
            if log_type:
                where["log_type"] = log_type
            if source_system:
                where["source_system"] = source_system
            
            try:
                # Get IDs to delete
                result = self.collection.get(
                    where=where,
                    include=['ids']
                )
                if result['ids']:
                    self.collection.delete(ids=result['ids'])
            except Exception:
                pass
        else:
            # Clear entire collection
            try:
                self.client.delete_collection(name=self.collection_name)
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Log file embeddings for similarity analysis"}
                )
            except Exception:
                pass
    
    def get_stats(self) -> Dict:
        """Get collection statistics.
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            count = self.collection.count()
            
            # Get log type distribution
            all_metadata = self.collection.get(include=['metadatas'])['metadatas']
            log_types = {}
            source_systems = {}
            
            for metadata in all_metadata:
                log_type = metadata.get('log_type', 'unknown')
                log_types[log_type] = log_types.get(log_type, 0) + 1
                
                source_system = metadata.get('source_system', 'unknown')
                source_systems[source_system] = source_systems.get(source_system, 0) + 1
            
            return {
                "total_embeddings": count,
                "log_types": log_types,
                "source_systems": source_systems,
                "collection_name": self.collection_name,
                "db_path": self.db_path
            }
        except Exception:
            return {
                "total_embeddings": 0,
                "log_types": {},
                "source_systems": {},
                "collection_name": self.collection_name,
                "db_path": self.db_path
            }
    
    def _get_file_size(self, file_path: str) -> int:
        """Get file size in bytes.
        
        Args:
            file_path: Path to the file
            
        Returns:
            File size in bytes, 0 if error
        """
        try:
            return os.path.getsize(file_path)
        except OSError:
            return 0
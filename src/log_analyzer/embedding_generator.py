"""Embedding generation using Ollama embeddinggemma model."""

import requests
from typing import List, Optional
import json


class OllamaEmbeddingGenerator:
    """Generate embeddings using Ollama's embeddinggemma model."""
    
    def __init__(self, host: str = "localhost", port: int = 11434, model: str = "embeddinggemma"):
        """Initialize the embedding generator.
        
        Args:
            host: Ollama server host
            port: Ollama server port
            model: Model name to use for embeddings
        """
        self.base_url = f"http://{host}:{port}"
        self.model = model
    
    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for given text.
        
        Args:
            text: Input text to embed
            
        Returns:
            List of floats representing the embedding, None if error
        """
        try:
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={
                    "model": self.model,
                    "prompt": text
                },
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get("embedding")
            
        except requests.exceptions.RequestException as e:
            print(f"Error connecting to Ollama: {e}")
            return None
        except json.JSONDecodeError as e:
            print(f"Error parsing response: {e}")
            return None
    
    def generate_file_embedding(self, file_path: str) -> Optional[List[float]]:
        """Generate embedding for file content.
        
        Args:
            file_path: Path to the file to embed
            
        Returns:
            List of floats representing the embedding, None if error
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return self.generate_embedding(content)
        except (IOError, UnicodeDecodeError) as e:
            print(f"Error reading file {file_path}: {e}")
            return None
    
    def health_check(self) -> bool:
        """Check if Ollama server is accessible.
        
        Returns:
            True if server is accessible, False otherwise
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
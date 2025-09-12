# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build/Test/Lint Commands

- **Install dependencies**: `uv sync` - Uses uv package manager
- **Install for development**: `uv sync --dev` - Installs with optional dev dependencies  
- **Run application**: `uv run log-analyzer [command]` - Main CLI entry point
- **Check status**: `uv run log-analyzer status` - Verify Ollama connection and storage
- **Lint**: `ruff check src/` - Check code quality
- **Format**: `ruff format src/` - Auto-format code
- **Test**: `pytest` (requires pytest in dev dependencies)

## Architecture Overview

This is a log anomaly detection tool using semantic embeddings and vector similarity analysis.

### Core Components

1. **CLI Layer** (`cli.py` + `chroma_cli.py`)
   - Dual CLI system: unified interface in `cli.py` with legacy JSON support
   - Enhanced ChromaDB-specific CLI in `chroma_cli.py`
   - Uses Click framework for command-line interface

2. **Embedding Generation** (`embedding_generator.py`)
   - Integrates with Ollama API using embeddinggemma model
   - Converts log file content to high-dimensional vectors
   - Includes health checks and error handling

3. **Storage Backends**
   - **ChromaDB** (`chroma_storage.py`): Default vector database with metadata support
   - **JSON** (`storage.py`): Legacy storage for backwards compatibility
   - Storage abstraction allows switching between backends

4. **Similarity Analysis**
   - **ChromaDB analyzer** (`chroma_similarity.py`): Advanced features like clustering, outlier detection
   - **Legacy analyzer** (`similarity.py`): Basic cosine similarity calculations

### External Dependencies

- **Ollama**: Required external service for embedding generation
  - Default: localhost:11434
  - Uses embeddinggemma model
  - Must be running before using the tool

- **ChromaDB**: Default storage backend (SQLite-based)
  - Stored in `./chroma_db/` directory
  - Supports metadata filtering and advanced queries

### Key Workflows

1. **Training**: `uv run log-analyzer train files...` - Create baseline embeddings
2. **Analysis**: `uv run log-analyzer analyze file.log` - Compare against baselines  
3. **Advanced Analytics** (ChromaDB only):
   - Outlier detection: `uv run log-analyzer outliers`
   - Clustering: `uv run log-analyzer clusters`

### Configuration Options

- `--use-json`: Switch to legacy JSON storage
- `--threshold`: Similarity threshold for anomaly detection (0-1)
- `--log-type` / `--source-system`: Metadata filtering (ChromaDB only)
- `--ollama-host` / `--ollama-port`: Ollama server configuration

## Code Style

- Line length: 119 characters (configured in pyproject.toml)
- Uses ruff for linting and formatting
- Type hints required (typing module imports)
- Click framework for CLI with context passing
- Error handling with appropriate exception types
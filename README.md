# Log Similarity Analyzer

A Proof of Concept (PoC) for detecting log file anomalies and divergence using semantic embeddings. This tool helps identify when log files deviate from normal patterns by comparing their content against established baselines using machine learning embeddings.

## 🎯 What This PoC Does

This tool addresses the challenge of **automated log anomaly detection** by:

- **Learning from normal patterns**: Creates semantic embeddings from known-good log files to establish baseline behavior
- **Detecting anomalies**: Identifies when new log files significantly diverge from the learned patterns
- **Quantifying similarity**: Provides numerical similarity scores to measure how much a file deviates
- **Configurable sensitivity**: Allows tuning of detection thresholds based on your specific use case

### Use Cases
- **System monitoring**: Detect when application logs show unusual patterns indicating errors or security issues
- **DevOps automation**: Automatically flag problematic deployments based on log content changes
- **Security monitoring**: Identify potential security incidents through abnormal log patterns
- **Quality assurance**: Catch regressions in software behavior through log analysis

## 🛠️ Prerequisites

1. **Ollama** with embeddinggemma model:
   ```bash
   # Install Ollama (if not already installed)
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Pull the embedding model
   ollama pull embeddinggemma
   
   # Verify Ollama is running on localhost:11434
   ollama list
   ```

2. **Python 3.8+** and **uv** package manager:
   ```bash
   # Install uv if not already installed
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

## 🚀 Installation & Setup

1. **Clone/navigate to the project directory**:
   ```bash
   cd log-similarity-analyzer
   ```

2. **Install dependencies**:
   ```bash
   uv sync
   ```

3. **Verify installation**:
   ```bash
   uv run log-analyzer status
   ```
   
   You should see:
   ```
   🔍 Log Similarity Analyzer Status
   ========================================
   Ollama server: ✅ Connected
   Stored embeddings: 0
   ```

## 📖 Usage Guide

### Step 1: Train with Baseline Data
First, create embeddings from your known-good log files:

```bash
# Train with multiple normal log files
uv run log-analyzer train logs/normal1.log logs/normal2.log logs/normal3.log

# Example output:
# 🔄 Generating embeddings for 3 files...
# Processing logs/normal1.log... ✅
# Processing logs/normal2.log... ✅
# Processing logs/normal3.log... ✅
# 📊 Successfully processed 3/3 files
# 💾 Embeddings saved to embeddings.json
```

### Step 2: Analyze New Files
Check if new log files show anomalous patterns:

```bash
# Basic analysis
uv run log-analyzer analyze logs/new_file.log

# With custom threshold and detailed output
uv run log-analyzer analyze logs/suspicious.log --threshold 0.9 --verbose
```

### Step 3: Monitor and Manage
```bash
# Check current status and stored baselines
uv run log-analyzer status

# Clear all stored embeddings (start fresh)
uv run log-analyzer clear
```

## 📊 Understanding the Output

### Normal File Detection
```
🔄 Analyzing normal_log.txt...
✅ NORMAL: File is similar to baseline (max similarity: 0.912)
📈 Most similar to: baseline1.log
📊 Divergence score: 0.088
```

### Anomaly Detection
```
🔄 Analyzing suspicious.log...
🚨 DIVERGENT: File shows significant differences (max similarity: 0.346)
📈 Most similar to: baseline2.log
📊 Divergence score: 0.654
```

### Detailed Analysis (with --verbose)
```
📋 Detailed similarity scores:
  baseline1.log: 0.909
  baseline2.log: 0.911
  baseline3.log: 0.905

📈 Statistics:
  Max similarity: 0.911
  Avg similarity: 0.908
  Min similarity: 0.905
```

## ⚙️ Configuration Options

### Threshold Tuning
The similarity threshold determines sensitivity:
- **0.9-0.95**: High sensitivity (catches subtle changes)
- **0.8** (default): Balanced sensitivity
- **0.7-0.8**: Lower sensitivity (only major differences)

### Command Line Options
```bash
# Custom Ollama server
uv run log-analyzer --ollama-host 192.168.1.100 --ollama-port 11434 train logs/*.log

# Custom storage location
uv run log-analyzer --storage-file /path/to/custom_embeddings.json train logs/*.log

# Analysis with custom threshold
uv run log-analyzer analyze file.log --threshold 0.95 --verbose
```

## 🔬 How It Works Technically

1. **Embedding Generation**: Uses Ollama's embeddinggemma model to convert log file content into high-dimensional vectors that capture semantic meaning

2. **Baseline Storage**: Stores embeddings of normal log files as JSON with metadata (timestamps, file paths)

3. **Similarity Calculation**: Computes cosine similarity between new file embeddings and stored baselines

4. **Anomaly Detection**: Flags files as divergent when maximum similarity falls below the configured threshold

5. **Result Analysis**: Provides detailed metrics including similarity scores, divergence measures, and statistical summaries

## 📁 Project Structure

```
log-similarity-analyzer/
├── src/log_analyzer/
│   ├── __init__.py
│   ├── embedding_generator.py    # Ollama API integration
│   ├── storage.py               # Embedding persistence
│   ├── similarity.py            # Cosine similarity analysis
│   └── cli.py                   # Command-line interface
├── test_logs/                   # Example log files
├── pyproject.toml              # Project configuration
├── embeddings.json             # Generated embeddings storage
└── README.md                   # This file
```

## 🧪 Example: Complete Workflow

```bash
# 1. Train with normal application logs
uv run log-analyzer train test_logs/normal1.log test_logs/normal2.log

# 2. Analyze a suspicious log file
uv run log-analyzer analyze test_logs/suspicious.log --verbose

# 3. Test with completely different content
uv run log-analyzer analyze test_logs/script.sh --threshold 0.8

# 4. Check what's stored
uv run log-analyzer status
```

## 🔍 Troubleshooting

### Common Issues

**"Cannot connect to Ollama server"**
```bash
# Check if Ollama is running
ollama list

# Start Ollama if needed
ollama serve

# Verify embeddinggemma model is available
ollama pull embeddinggemma
```

**"No baseline embeddings found"**
```bash
# Train with some baseline files first
uv run log-analyzer train path/to/normal/logs/*.log
```

**Low similarity scores for similar files**
- This is normal for the embeddinggemma model, which focuses on semantic content
- Adjust threshold based on your specific log patterns
- Consider training with more diverse baseline examples

## 🚀 ChromaDB Integration (Enhanced Features)

This PoC now includes **ChromaDB integration** as the default storage backend, providing significant advantages over simple JSON storage:

### Why ChromaDB?

**ChromaDB is a vector database specifically designed for embeddings:**
- 🚀 **Native similarity search** - Efficient vector operations with built-in cosine similarity
- 📊 **Metadata support** - Rich filtering by log type, source system, timestamps
- 🗂️ **Collections** - Organize embeddings by categories (app logs, security logs, etc.)
- 💾 **Persistent storage** - Automatic disk persistence with SQLite backend
- 🔍 **Hybrid search** - Combine similarity with metadata filters
- ⚡ **Performance** - Optimized for large-scale vector operations

### Enhanced Commands with ChromaDB

#### Advanced Training with Metadata
```bash
# Train with log type classification
uv run log-analyzer train app1.log app2.log --log-type application --source-system web-server

# Train security logs separately
uv run log-analyzer train security1.log security2.log --log-type security --source-system firewall
```

#### Filtered Analysis
```bash
# Analyze against only application logs
uv run log-analyzer analyze new.log --log-type application

# Analyze against specific source system
uv run log-analyzer analyze suspicious.log --source-system web-server --threshold 0.9

# Compare against top 10 most similar files
uv run log-analyzer analyze test.log --n-similar 10 --verbose
```

#### Advanced Analytics (ChromaDB Only)
```bash
# Find outlier files that don't fit normal patterns
uv run log-analyzer outliers --threshold 0.7 --log-type application

# Cluster similar files together
uv run log-analyzer clusters --threshold 0.8

# Enhanced status with metadata breakdown
uv run log-analyzer status
```

## 📋 Detailed Command Reference

### `analyze` - File Similarity Analysis

The core command for detecting anomalies in log files by comparing against stored baselines.

#### Basic Usage
```bash
uv run log-analyzer analyze suspicious.log
```

#### Advanced Options
```bash
uv run log-analyzer analyze new_file.log \
    --threshold 0.9 \           # Similarity threshold (0-1, higher = more strict)
    --log-type application \    # Only compare against application logs
    --source-system nginx \     # Only compare against nginx logs
    --n-similar 10 \           # Compare against top 10 most similar files
    --verbose                  # Show detailed similarity scores
```

#### Output Interpretation
```
🔄 Analyzing suspicious.log...
🚨 DIVERGENT: File shows significant differences (max similarity: 0.346)
📈 Most similar to: baseline2.log
   └─ Log type: application
   └─ Source: web-server
📊 Divergence score: 0.654
🔢 Compared against 5 baseline files

📋 Detailed similarity scores (with --verbose):
  baseline1.log: 0.346
    └─ application | web-server
  baseline2.log: 0.298
    └─ application | database
```

**Key Metrics:**
- **Similarity Score**: 0-1 range (1 = identical, 0 = completely different)
- **Divergence Score**: 1 - max_similarity (higher = more anomalous)
- **Threshold**: Files below threshold are flagged as divergent

#### Use Cases
- **Production monitoring**: Detect when new logs show unusual patterns
- **Security analysis**: Identify potentially malicious activity
- **System health**: Spot degraded performance or errors
- **Change detection**: Verify deployments haven't introduced issues

### `outliers` - Anomaly Detection (ChromaDB Only)

Identifies files in your baseline that are significantly different from all other baselines. Useful for finding problematic training data or genuinely unusual patterns.

#### Basic Usage
```bash
uv run log-analyzer outliers
```

#### Advanced Options
```bash
uv run log-analyzer outliers \
    --threshold 0.7 \          # Files with max similarity < 0.7 are outliers
    --log-type security \      # Only analyze security logs
    --source-system firewall   # Only analyze firewall logs
```

#### Example Output
```
🔄 Finding outliers with threshold 0.7...
🚨 Found 2 outlier files:

📄 logs/corrupted_log.txt
   Max similarity: 0.423
   Avg similarity: 0.401
   Divergence score: 0.577
   Log type: application
   Source: web-server

📄 logs/emergency_shutdown.log
   Max similarity: 0.651
   Avg similarity: 0.612
   Divergence score: 0.349
   Log type: system
   Source: database
```

#### Use Cases
- **Training data cleanup**: Remove bad examples from your baselines
- **Anomaly discovery**: Find rare but legitimate patterns in historical data
- **Data quality assessment**: Identify corrupted or incomplete log files
- **Pattern validation**: Ensure your baselines represent normal behavior

### `clusters` - Similarity Grouping (ChromaDB Only)

Groups baseline files into clusters based on similarity. Files in the same cluster have similar patterns, while different clusters represent distinct log patterns.

#### Basic Usage
```bash
uv run log-analyzer clusters
```

#### Advanced Options
```bash
uv run log-analyzer clusters \
    --threshold 0.8 \          # Files with similarity ≥ 0.8 are grouped together
    --log-type application \   # Only cluster application logs
    --source-system nginx     # Only cluster nginx logs
```

#### Example Output
```
🔄 Clustering files with threshold 0.8...
📊 Found 3 clusters:

🗂️  CLUSTER_0 (4 files):
   • logs/normal_startup_1.log
   • logs/normal_startup_2.log
   • logs/normal_startup_3.log
   • logs/normal_startup_4.log

🗂️  CLUSTER_1 (2 files):
   • logs/high_load_1.log
   • logs/high_load_2.log

🗂️  CLUSTER_2 (1 files):
   • logs/maintenance_mode.log
```

#### Interpretation
- **Large clusters**: Common patterns (normal operations, typical errors)
- **Small clusters**: Rare but legitimate patterns (maintenance, edge cases)
- **Single-file clusters**: Unique patterns (potential outliers or special events)

#### Use Cases
- **Pattern discovery**: Understand different types of normal behavior
- **Baseline organization**: Group similar logs for more targeted analysis
- **Monitoring strategy**: Set different thresholds for different pattern types
- **System understanding**: Map log patterns to operational states

## 🔄 Legacy JSON Storage

For backwards compatibility or simpler deployments, you can use the original JSON-based storage system.

### When to Use JSON Storage
- **Simple deployments**: No need for vector database setup
- **Small datasets**: < 1000 log files
- **Quick prototyping**: Minimal dependencies
- **Backwards compatibility**: Existing JSON embeddings files

### JSON Mode Commands
```bash
# Enable JSON mode for all commands
uv run log-analyzer --use-json train logs/*.log
uv run log-analyzer --use-json analyze new.log --threshold 0.8 --verbose
uv run log-analyzer --use-json status
uv run log-analyzer --use-json clear

# Custom JSON storage location
uv run log-analyzer --use-json --storage-file /path/to/custom.json train logs/*.log
```

### JSON vs ChromaDB Comparison

| Feature | JSON Storage | ChromaDB Storage |
|---------|-------------|------------------|
| **Setup Complexity** | ✅ Simple | ⚠️ Requires ChromaDB |
| **Performance** | ⚠️ Loads all into memory | ✅ Efficient vector search |
| **Metadata Support** | ❌ File paths only | ✅ Rich metadata |
| **Filtering** | ❌ No filtering | ✅ Log type, source, etc. |
| **Advanced Analytics** | ❌ Basic similarity only | ✅ Outliers, clustering |
| **Scalability** | ⚠️ Limited by memory | ✅ Scales to millions |
| **Production Ready** | ⚠️ Basic use cases | ✅ Enterprise features |

### Migration from JSON to ChromaDB

To migrate existing JSON embeddings to ChromaDB:

```bash
# 1. Backup existing JSON file
cp embeddings.json embeddings.json.backup

# 2. Train with ChromaDB using same files
uv run log-analyzer train original_files/*.log --log-type application

# 3. Verify migration
uv run log-analyzer status

# 4. Test analysis works the same
uv run log-analyzer analyze test_file.log
```

### JSON Storage Format
```json
{
  "path/to/file.log": {
    "embedding": [0.123, -0.456, 0.789, ...],
    "timestamp": "2024-01-15T10:30:00",
    "file_path": "path/to/file.log"
  }
}
```

ChromaDB provides significantly more capabilities, but JSON storage remains available for simpler use cases or compatibility requirements.

#### Selective Data Management
```bash
# Clear only security logs
uv run log-analyzer clear --log-type security

# Clear logs from specific system
uv run log-analyzer clear --source-system old-server
```

### Usage Comparison

#### Basic JSON Storage (Legacy)
```bash
# Use simple JSON storage (backwards compatible)
uv run log-analyzer --use-json train logs/*.log
uv run log-analyzer --use-json analyze new.log
```

#### ChromaDB Storage (Default & Recommended)
```bash
# ChromaDB with enhanced features (default)
uv run log-analyzer train logs/*.log --log-type application
uv run log-analyzer analyze new.log --log-type application --verbose
uv run log-analyzer outliers --threshold 0.8
```

### Example: Multi-System Log Monitoring

```bash
# 1. Train different log types with metadata
uv run log-analyzer train web-access.log --log-type application --source-system nginx
uv run log-analyzer train firewall.log --log-type security --source-system iptables
uv run log-analyzer train db.log --log-type database --source-system postgresql

# 2. Analyze new logs with specific filtering
uv run log-analyzer analyze suspicious-web.log --log-type application --source-system nginx

# 3. Find outliers in security logs only
uv run log-analyzer outliers --log-type security --threshold 0.7

# 4. Check system status with metadata breakdown
uv run log-analyzer status
```

## 🚀 Next Steps for Production Use

1. **Scale storage**: ChromaDB already provides production-ready vector storage
2. **Add monitoring**: Integrate with metrics systems (Prometheus, Grafana)
3. **Batch processing**: Process multiple files simultaneously
4. **Model optimization**: Fine-tune embedding models for your specific log formats
5. **Integration**: Build APIs for integration with existing monitoring tools
6. **Alerting**: Add notification systems for anomaly detection
7. **Multi-collection support**: Separate collections for different environments (dev, staging, prod)

## 📝 License

This PoC is provided as-is for demonstration and learning purposes.
"""Enhanced CLI with ChromaDB features for log similarity analyzer."""

import click
import os
from log_analyzer.embedding_generator import OllamaEmbeddingGenerator
from log_analyzer.chroma_storage import ChromaEmbeddingStorage
from log_analyzer.chroma_similarity import ChromaSimilarityAnalyzer


@click.group()
@click.option('--db-path', default='./chroma_db', help='ChromaDB database path')
@click.option('--collection', default='log_embeddings', help='ChromaDB collection name')
@click.option('--ollama-host', default='localhost', help='Ollama server host')
@click.option('--ollama-port', default=11434, help='Ollama server port')
@click.pass_context
def cli(ctx, db_path, collection, ollama_host, ollama_port):
    """Enhanced log similarity analyzer using ChromaDB."""
    ctx.ensure_object(dict)
    ctx.obj['storage'] = ChromaEmbeddingStorage(db_path, collection)
    ctx.obj['generator'] = OllamaEmbeddingGenerator(ollama_host, ollama_port)
    ctx.obj['analyzer'] = ChromaSimilarityAnalyzer(ctx.obj['storage'])


@cli.command()
@click.argument('files', nargs=-1, type=click.Path(exists=True), required=True)
@click.option('--log-type', default='application', help='Type of log files (application, security, system, etc.)')
@click.option('--source-system', help='Source system name (optional)')
@click.pass_context
def train(ctx, files, log_type, source_system):
    """Create baseline embeddings from training files with metadata."""
    storage = ctx.obj['storage']
    generator = ctx.obj['generator']
    
    # Check Ollama connection
    if not generator.health_check():
        click.echo("❌ Cannot connect to Ollama server. Please ensure it's running.", err=True)
        return
    
    click.echo(f"🔄 Generating embeddings for {len(files)} files...")
    if log_type != 'application':
        click.echo(f"📂 Log type: {log_type}")
    if source_system:
        click.echo(f"🖥️  Source system: {source_system}")
    
    success_count = 0
    for file_path in files:
        click.echo(f"Processing {file_path}...", nl=False)
        
        embedding = generator.generate_file_embedding(file_path)
        if embedding:
            storage.save_embedding(
                file_path=file_path,
                embedding=embedding,
                log_type=log_type,
                source_system=source_system
            )
            click.echo(" ✅")
            success_count += 1
        else:
            click.echo(" ❌")
    
    click.echo(f"\n📊 Successfully processed {success_count}/{len(files)} files")
    if success_count > 0:
        click.echo(f"💾 Embeddings saved to ChromaDB at {storage.db_path}")


@cli.command()
@click.argument('file', type=click.Path(exists=True), required=True)
@click.option('--threshold', default=0.8, help='Similarity threshold for divergence detection')
@click.option('--log-type', help='Filter baselines by log type')
@click.option('--source-system', help='Filter baselines by source system')
@click.option('--n-similar', default=5, help='Number of most similar files to compare against')
@click.option('--verbose', '-v', is_flag=True, help='Show detailed similarity scores')
@click.pass_context
def analyze(ctx, file, threshold, log_type, source_system, n_similar, verbose):
    """Analyze a file against stored baseline embeddings with filtering."""
    storage = ctx.obj['storage']
    generator = ctx.obj['generator']
    analyzer = ctx.obj['analyzer']
    
    # Check Ollama connection
    if not generator.health_check():
        click.echo("❌ Cannot connect to Ollama server. Please ensure it's running.", err=True)
        return
    
    # Check if we have any baselines
    stats = storage.get_stats()
    if stats['total_embeddings'] == 0:
        click.echo("❌ No baseline embeddings found. Run 'train' command first.", err=True)
        return
    
    # Show filter information
    filters_applied = []
    if log_type:
        filters_applied.append(f"log_type={log_type}")
    if source_system:
        filters_applied.append(f"source_system={source_system}")
    
    if filters_applied:
        click.echo(f"🔍 Applying filters: {', '.join(filters_applied)}")
    
    click.echo(f"🔄 Analyzing {file}...")
    
    # Generate embedding for new file
    embedding = generator.generate_file_embedding(file)
    if not embedding:
        click.echo("❌ Failed to generate embedding for the file.", err=True)
        return
    
    # Analyze similarity with ChromaDB
    results = analyzer.analyze_file_similarity(
        new_embedding=embedding,
        threshold=threshold,
        n_similar=n_similar,
        log_type=log_type,
        source_system=source_system
    )
    
    if results['status'] != 'success':
        click.echo(f"❌ Analysis failed: {results['message']}", err=True)
        return
    
    # Display results
    is_divergent = results['is_divergent']
    max_similarity = results['max_similarity']
    most_similar = results['most_similar_file']
    most_similar_meta = results['most_similar_metadata']
    
    if is_divergent:
        click.echo(f"🚨 DIVERGENT: File shows significant differences (max similarity: {max_similarity:.3f})")
    else:
        click.echo(f"✅ NORMAL: File is similar to baseline (max similarity: {max_similarity:.3f})")
    
    click.echo(f"📈 Most similar to: {most_similar}")
    click.echo(f"   └─ Log type: {most_similar_meta['log_type']}")
    if most_similar_meta['source_system'] != 'unknown':
        click.echo(f"   └─ Source: {most_similar_meta['source_system']}")
    click.echo(f"📊 Divergence score: {results['divergence_score']:.3f}")
    click.echo(f"🔢 Compared against {results['total_compared']} baseline files")
    
    if verbose:
        click.echo("\n📋 Detailed similarity scores:")
        for baseline_file, similarity in results['similarities'].items():
            metadata = results['metadata'][baseline_file]
            click.echo(f"  {baseline_file}: {similarity:.3f}")
            click.echo(f"    └─ {metadata['log_type']} | {metadata['source_system']}")
        
        click.echo(f"\n📈 Statistics:")
        click.echo(f"  Max similarity: {results['max_similarity']:.3f}")
        click.echo(f"  Avg similarity: {results['avg_similarity']:.3f}")
        click.echo(f"  Min similarity: {results['min_similarity']:.3f}")


@cli.command()
@click.option('--log-type', help='Filter by log type')
@click.option('--source-system', help='Filter by source system')
@click.pass_context
def status(ctx):
    """Show current status and stored embeddings with enhanced ChromaDB info."""
    storage = ctx.obj['storage']
    generator = ctx.obj['generator']
    
    click.echo("🔍 Enhanced Log Similarity Analyzer Status")
    click.echo("=" * 50)
    
    # Check Ollama connection
    ollama_status = "✅ Connected" if generator.health_check() else "❌ Disconnected"
    click.echo(f"Ollama server: {ollama_status}")
    
    # Show ChromaDB statistics
    stats = storage.get_stats()
    click.echo(f"ChromaDB path: {stats['db_path']}")
    click.echo(f"Collection: {stats['collection_name']}")
    click.echo(f"Total embeddings: {stats['total_embeddings']}")
    
    if stats['total_embeddings'] > 0:
        # Show log type distribution
        if stats['log_types']:
            click.echo("\n📂 Log types:")
            for log_type, count in stats['log_types'].items():
                click.echo(f"  • {log_type}: {count} files")
        
        # Show source system distribution
        if stats['source_systems'] and any(s != 'unknown' for s in stats['source_systems']):
            click.echo("\n🖥️  Source systems:")
            for system, count in stats['source_systems'].items():
                if system != 'unknown':
                    click.echo(f"  • {system}: {count} files")
        
        # Show recent files
        files = storage.list_files()
        if files:
            click.echo(f"\n📁 Recent baseline files (showing up to 10):")
            for file_info in files[:10]:
                click.echo(f"  • {file_info['file_path']}")
                click.echo(f"    └─ {file_info['log_type']} | {file_info.get('timestamp', 'unknown')[:19]}")
    else:
        click.echo("\n💡 No baseline embeddings found. Use 'train' command to create them.")


@cli.command()
@click.option('--threshold', default=0.7, help='Similarity threshold for outlier detection')
@click.option('--log-type', help='Filter by log type')
@click.option('--source-system', help='Filter by source system')
@click.pass_context
def outliers(ctx, threshold, log_type, source_system):
    """Find outlier files that are dissimilar to others."""
    storage = ctx.obj['storage']
    analyzer = ctx.obj['analyzer']
    
    stats = storage.get_stats()
    if stats['total_embeddings'] < 2:
        click.echo("❌ Need at least 2 baseline files to detect outliers.", err=True)
        return
    
    # Show filter information
    filters_applied = []
    if log_type:
        filters_applied.append(f"log_type={log_type}")
    if source_system:
        filters_applied.append(f"source_system={source_system}")
    
    if filters_applied:
        click.echo(f"🔍 Applying filters: {', '.join(filters_applied)}")
    
    click.echo(f"🔄 Finding outliers with threshold {threshold}...")
    
    outliers = analyzer.find_outliers(
        threshold=threshold,
        log_type=log_type,
        source_system=source_system
    )
    
    if not outliers:
        click.echo("✅ No outliers found - all files are within normal similarity ranges.")
        return
    
    click.echo(f"🚨 Found {len(outliers)} outlier files:")
    
    for outlier in outliers:
        click.echo(f"\n📄 {outlier['file_path']}")
        click.echo(f"   Max similarity: {outlier['max_similarity']:.3f}")
        click.echo(f"   Avg similarity: {outlier['avg_similarity']:.3f}")
        click.echo(f"   Divergence score: {outlier['divergence_score']:.3f}")
        
        metadata = outlier['metadata']
        if metadata:
            click.echo(f"   Log type: {metadata.get('log_type', 'unknown')}")
            if metadata.get('source_system', 'unknown') != 'unknown':
                click.echo(f"   Source: {metadata['source_system']}")


@cli.command()
@click.option('--threshold', default=0.8, help='Similarity threshold for clustering')
@click.option('--log-type', help='Filter by log type')
@click.option('--source-system', help='Filter by source system')
@click.pass_context
def clusters(ctx, threshold, log_type, source_system):
    """Show file clusters based on similarity."""
    storage = ctx.obj['storage']
    analyzer = ctx.obj['analyzer']
    
    stats = storage.get_stats()
    if stats['total_embeddings'] < 2:
        click.echo("❌ Need at least 2 baseline files for clustering.", err=True)
        return
    
    # Show filter information
    filters_applied = []
    if log_type:
        filters_applied.append(f"log_type={log_type}")
    if source_system:
        filters_applied.append(f"source_system={source_system}")
    
    if filters_applied:
        click.echo(f"🔍 Applying filters: {', '.join(filters_applied)}")
    
    click.echo(f"🔄 Clustering files with threshold {threshold}...")
    
    clusters = analyzer.cluster_analysis(
        similarity_threshold=threshold,
        log_type=log_type,
        source_system=source_system
    )
    
    click.echo(f"📊 Found {len(clusters)} clusters:")
    
    for cluster_id, files in clusters.items():
        click.echo(f"\n🗂️  {cluster_id.upper()} ({len(files)} files):")
        for file_path in files:
            click.echo(f"   • {file_path}")


@cli.command()
@click.option('--log-type', help='Only clear embeddings of this log type')
@click.option('--source-system', help='Only clear embeddings from this source system')
@click.confirmation_option(prompt='Are you sure you want to clear embeddings?')
@click.pass_context
def clear(ctx, log_type, source_system):
    """Clear stored embeddings with optional filtering."""
    storage = ctx.obj['storage']
    
    if log_type or source_system:
        filters = []
        if log_type:
            filters.append(f"log_type={log_type}")
        if source_system:
            filters.append(f"source_system={source_system}")
        click.echo(f"🗑️  Clearing embeddings with filters: {', '.join(filters)}")
    else:
        click.echo("🗑️  Clearing all embeddings...")
    
    storage.clear_all(log_type=log_type, source_system=source_system)
    click.echo("✅ Embeddings cleared.")


def main():
    """Entry point for the enhanced CLI."""
    cli()
"""Command-line interface for file organizer."""

from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.progress import Progress
from rich.table import Table
from src.clustering import cluster_embeddings
from src.config import Config
from src.features.audio_features import AudioFeatureExtractor
from src.features.cache import EmbeddingCache
from src.features.document_features import DocumentFeatureExtractor
from src.features.image_features import ImageFeatureExtractor
from src.features.video_features import VideoFeatureExtractor
from src.filetype import detect_file_type
from src.hashing import (
    compute_image_hash,
    compute_sha256,
    compute_video_hash,
    find_duplicates,
    find_perceptual_duplicates,
)
from src.models import FileRecord, FileType
from src.organizer import FileOrganizer
from src.scanner import count_files_by_type, scan_directory
from src.undo_manager import UndoManager
from src.utils.logging import get_logger, setup_logging

app = typer.Typer(help="AI-powered file organizer with semantic classification")
console = Console()


@app.command()
def organize(
    input_dir: Path = typer.Option(..., "--input", "-i", help="Input directory to scan"),
    output_dir: Path = typer.Option(..., "--output", "-o", help="Output directory for organized files"),
    mode: str = typer.Option("cluster", "--mode", "-m", help="Mode: classify or cluster"),
    labels: Optional[str] = typer.Option(None, "--labels", "-l", help="Comma-separated labels for classify mode"),
    move: bool = typer.Option(False, "--move", help="Move files instead of copying (default: copy)"),
    workers: int = typer.Option(4, "--workers", "-w", help="Number of parallel workers"),
    models_cache: Optional[Path] = typer.Option(None, "--models-cache", help="Model cache directory"),
    log_file: Optional[Path] = typer.Option(None, "--log-file", help="Transaction log file path"),
    min_similarity: int = typer.Option(5, "--min-similarity", help="Perceptual hash threshold (0-64)"),
    min_file_size: int = typer.Option(100, "--min-file-size", help="Minimum file size in bytes (skip smaller)"),
    rotation_invariant: bool = typer.Option(True, "--rotation-invariant/--no-rotation-invariant", help="Detect rotated duplicates"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Simulate without actual operations"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Minimal output (errors only)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose debug output"),
) -> None:
    """Organize files with AI-powered semantic classification."""
    
    # Setup logging
    if quiet:
        log_level = "ERROR"  # Only errors in quiet mode
    elif verbose:
        log_level = "DEBUG"
    else:
        log_level = "INFO"
    
    setup_logging(log_level=log_level, log_file=log_file, verbose=verbose)
    logger = get_logger(__name__)

    # Parse labels
    label_list = [l.strip() for l in labels.split(",")] if labels else []

    console.print(f"Input: {input_dir}")
    console.print(f"Output: {output_dir}")
    console.print(f"Mode: {mode}")
    console.print(f"Operation: {'Move' if move else 'Copy'}")
    if dry_run:
        console.print("[yellow]DRY RUN MODE - No files will be modified[/yellow]")
    console.print()

    # Initialize components
    logger.info("Initializing components")
    undo_manager = UndoManager(config.log_file)
    cache = EmbeddingCache(config.models_cache / "cache")
    organizer = FileOrganizer(config, undo_manager)

    # Scan directory
    console.print("[cyan]Scanning directory...[/cyan]")
    records = list(scan_directory(input_dir, config, progress=True))

    if not records:
        console.print("[yellow]No files found to organize[/yellow]")
        return

    # Show statistics
    counts = count_files_by_type(records)
    table = Table(title="File Type Distribution")
    table.add_column("Type", style="cyan")
    table.add_column("Count", style="magenta", justify="right")
    for file_type, count in sorted(counts.items()):
        table.add_row(file_type, str(count))
    console.print(table)

    # Compute hashes
    console.print("\n[cyan]Computing hashes...[/cyan]")
    with Progress() as progress:
        task = progress.add_task("Hashing files...", total=len(records))
        for record in records:
            try:
                record.sha256 = compute_sha256(record.path)
                progress.update(task, advance=1)
            except Exception as e:
                logger.error(f"Failed to hash {record.path}: {e}")
                progress.update(task, advance=1)

    # Find exact duplicates
    console.print("\n[cyan]Finding exact duplicates...[/cyan]")
    exact_dupes = find_duplicates(records, hash_field="sha256")
    console.print(f"Found {len(exact_dupes)} duplicate groups (exact match)")

    # Mark duplicates
    for hash_value, paths in exact_dupes.items():
        original = paths[0]
        for dup_path in paths[1:]:
            for record in records:
                if record.path == dup_path:
                    record.is_duplicate = True
                    record.duplicate_of = original

    # Process by file type
    console.print("\n[cyan]Processing files by type...[/cyan]")

    # Images
    image_records = [r for r in records if r.file_type == FileType.IMAGE and not r.is_duplicate]
    if image_records:
        console.print(f"\n[green]Processing {len(image_records)} images...[/green]")
        _process_images(image_records, config, cache)

    # Audio
    audio_records = [r for r in records if r.file_type == FileType.AUDIO and not r.is_duplicate]
    if audio_records:
        console.print(f"\n[green]Processing {len(audio_records)} audio files...[/green]")
        _process_audio(audio_records, config)

    # Video
    video_records = [r for r in records if r.file_type == FileType.VIDEO and not r.is_duplicate]
    if video_records:
        console.print(f"\n[green]Processing {len(video_records)} videos...[/green]")
        _process_videos(video_records, config, cache)

    # Documents
    document_records = [r for r in records if r.file_type == FileType.DOCUMENT and not r.is_duplicate]
    if document_records:
        console.print(f"\n[green]Processing {len(document_records)} documents...[/green]")
        _process_documents(document_records, config, cache)

    # Organize files
    console.print("\n[cyan]Organizing files...[/cyan]")
    successful, failed = organizer.organize_batch(records, dry_run=dry_run)

    # Summary
    console.print(f"\n[bold green]Organization complete![/bold green]")
    console.print(f"Successful: {successful}")
    console.print(f"Failed: {failed}")
    if dry_run:
        console.print("[yellow]DRY RUN - No files were actually moved/copied[/yellow]")
    else:
        console.print(f"\nTransaction log: {config.log_file}")


def _process_images(records: list[FileRecord], config: Config, cache: EmbeddingCache) -> None:
    """Process image files."""
    logger = get_logger(__name__)
    extractor = ImageFeatureExtractor(config, cache)

    if config.mode == "classify" and config.labels:
        # Zero-shot classification
        console.print(f"Classifying with labels: {', '.join(config.labels)}")
        for record in records:
            label = extractor.classify_zero_shot(record.path, config.labels)
            record.label = label if label else "uncategorized"
    else:
        # Clustering
        console.print("Extracting embeddings for clustering...")
        embeddings = []
        valid_records = []

        for record in records:
            emb = extractor.extract_image_embedding(record.path, sha256=record.sha256)
            if emb is not None:
                embeddings.append(emb)
                valid_records.append(record)

        if embeddings:
            import numpy as np
            embeddings_array = np.array(embeddings)
            
            console.print(f"Clustering {len(embeddings)} images...")
            labels = cluster_embeddings(
                embeddings_array,
                algorithm=config.clustering_algorithm,
                n_clusters=config.n_clusters,
            )

            for record, label in zip(valid_records, labels):
                record.cluster_id = int(label)


def _process_audio(records: list[FileRecord], config: Config) -> None:
    """Process audio files."""
    logger = get_logger(__name__)
    extractor = AudioFeatureExtractor()

    if config.mode == "classify":
        # Simple rule-based classification
        console.print("Classifying audio files...")
        for record in records:
            label = extractor.classify_simple(record.path)
            record.label = label if label else "uncategorized"
    else:
        # Clustering
        console.print("Extracting audio features for clustering...")
        embeddings = []
        valid_records = []

        for record in records:
            emb = extractor.extract_embedding(record.path)
            if emb is not None:
                embeddings.append(emb)
                valid_records.append(record)

        if embeddings:
            import numpy as np
            embeddings_array = np.array(embeddings)
            
            console.print(f"Clustering {len(embeddings)} audio files...")
            labels = cluster_embeddings(
                embeddings_array,
                algorithm=config.clustering_algorithm,
                n_clusters=config.n_clusters,
            )

            for record, label in zip(valid_records, labels):
                record.cluster_id = int(label)


def _process_videos(records: list[FileRecord], config: Config, cache: EmbeddingCache) -> None:
    """Process video files."""
    logger = get_logger(__name__)
    image_extractor = ImageFeatureExtractor(config, cache)
    video_extractor = VideoFeatureExtractor(
        image_extractor,
        frame_interval=config.video_frame_interval,
        max_frames=config.max_video_frames,
    )

    console.print("Extracting video embeddings...")
    embeddings = []
    valid_records = []

    for record in records:
        emb = video_extractor.extract_embedding(record.path)
        if emb is not None:
            embeddings.append(emb)
            valid_records.append(record)

    if embeddings:
        import numpy as np
        embeddings_array = np.array(embeddings)
        
        console.print(f"Clustering {len(embeddings)} videos...")
        labels = cluster_embeddings(
            embeddings_array,
            algorithm=config.clustering_algorithm,
            n_clusters=config.n_clusters,
        )

        for record, label in zip(valid_records, labels):
            record.cluster_id = int(label)


def _process_documents(records: list[FileRecord], config: Config, cache: EmbeddingCache) -> None:
    """Process document files."""
    logger = get_logger(__name__)
    extractor = DocumentFeatureExtractor(cache=cache, n_components=128)

    console.print("Extracting document embeddings using TF-IDF + LSA...")
    
    # Batch process for efficiency
    file_paths = [record.path for record in records]
    sha256s = [record.sha256 for record in records]
    
    embeddings = extractor.extract_embeddings_batch(file_paths, sha256s)
    
    # Filter valid embeddings
    valid_embeddings = []
    valid_records = []
    
    for record, emb in zip(records, embeddings):
        if emb is not None:
            valid_embeddings.append(emb)
            valid_records.append(record)
    
    if valid_embeddings:
        import numpy as np
        embeddings_array = np.array(valid_embeddings)
        
        console.print(f"Clustering {len(valid_embeddings)} documents...")
        labels = cluster_embeddings(
            embeddings_array,
            algorithm=config.clustering_algorithm,
            n_clusters=config.n_clusters,
        )

        for record, label in zip(valid_records, labels):
            record.cluster_id = int(label)


@app.command()
def undo(
    log_file: Path = typer.Option(..., "--log-file", help="Transaction log file path"),
    count: int = typer.Option(1, "--count", "-n", help="Number of transactions to undo"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Simulate without actual operations"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
) -> None:
    """Undo last N transactions."""
    
    setup_logging(log_level="DEBUG" if verbose else "INFO", verbose=verbose)
    logger = get_logger(__name__)

    console.print(f"\n[bold yellow]Undo Manager[/bold yellow]")
    console.print(f"Log file: {log_file}")
    console.print(f"Undoing last {count} transaction(s)")
    if dry_run:
        console.print("[yellow]DRY RUN MODE[/yellow]")
    console.print()

    undo_manager = UndoManager(log_file)
    successful, failed = undo_manager.undo(count=count, dry_run=dry_run)

    console.print(f"\n[bold green]Undo complete![/bold green]")
    console.print(f"Successful: {successful}")
    console.print(f"Failed: {failed}")


@app.command()
def version() -> None:
    """Show version information."""
    from src import __version__
    console.print(f"File Organizer v{__version__}")


if __name__ == "__main__":
    app()

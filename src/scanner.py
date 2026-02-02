"""Recursive directory scanner."""

import os
from datetime import datetime
from pathlib import Path
from typing import Generator, Optional

from tqdm import tqdm

from src.config import Config
from src.filetype import detect_file_type
from src.models import FileRecord
from src.utils.logging import get_logger

logger = get_logger(__name__)


def scan_directory(
    input_dir: Path,
    config: Config,
    progress: bool = True,
) -> Generator[FileRecord, None, None]:
    """Recursively scan directory and yield FileRecord objects.

    Args:
        input_dir: Root directory to scan
        config: Application configuration
        progress: Show progress bar

    Yields:
        FileRecord objects for each file found
    """
    if not input_dir.exists():
        logger.error("Input directory does not exist", path=str(input_dir))
        raise FileNotFoundError(f"Directory not found: {input_dir}")

    if not input_dir.is_dir():
        logger.error("Input path is not a directory", path=str(input_dir))
        raise NotADirectoryError(f"Not a directory: {input_dir}")

    logger.info("Starting directory scan", path=str(input_dir))

    # First pass: count files for progress bar
    file_count = 0
    if progress:
        for root, _, files in os.walk(input_dir):
            file_count += len(files)
        logger.debug("File count for progress", count=file_count)

    # Second pass: process files
    pbar = tqdm(total=file_count, desc="Scanning files", disable=not progress, unit="file")

    scanned = 0
    skipped = 0

    for root, _, files in os.walk(input_dir):
        root_path = Path(root)

        for filename in files:
            file_path = root_path / filename

            try:
                # Skip symlinks
                if file_path.is_symlink():
                    logger.debug("Skipping symlink", path=str(file_path))
                    skipped += 1
                    pbar.update(1)
                    continue

                # Get file stats
                stat = file_path.stat()
                file_size = stat.st_size

                # Skip files below minimum size
                if file_size < config.min_file_size:
                    logger.debug(
                        "Skipping file below minimum size",
                        path=str(file_path),
                        size=file_size,
                        min_size=config.min_file_size,
                    )
                    skipped += 1
                    pbar.update(1)
                    continue

                # Detect file type
                file_type, mime_type = detect_file_type(file_path)

                # Create FileRecord
                record = FileRecord(
                    path=file_path,
                    size=file_size,
                    mtime=datetime.fromtimestamp(stat.st_mtime),
                    mime_type=mime_type,
                    file_type=file_type,
                )

                scanned += 1
                yield record

            except PermissionError:
                logger.warning("Permission denied", path=str(file_path))
                skipped += 1
            except Exception as e:
                logger.error("Error processing file", path=str(file_path), error=str(e))
                skipped += 1
            finally:
                pbar.update(1)

    pbar.close()

    logger.info(
        "Directory scan complete",
        scanned=scanned,
        skipped=skipped,
        total=file_count,
    )


def count_files_by_type(records: list[FileRecord]) -> dict[str, int]:
    """Count files by type.

    Args:
        records: List of FileRecord objects

    Returns:
        Dictionary mapping file type to count
    """
    counts: dict[str, int] = {}
    for record in records:
        file_type = record.file_type.value if record.file_type else "unknown"
        counts[file_type] = counts.get(file_type, 0) + 1
    return counts

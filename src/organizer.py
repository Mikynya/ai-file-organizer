"""File organization logic with copy/move operations."""

import shutil
from pathlib import Path
from typing import Optional

from src.config import Config
from src.models import FileRecord, FileType, OperationType, Transaction
from src.undo_manager import UndoManager
from src.utils.logging import get_logger

logger = get_logger(__name__)


class FileOrganizer:
    """Organize files into structured output directory."""

    def __init__(self, config: Config, undo_manager: UndoManager):
        """Initialize file organizer.

        Args:
            config: Application configuration
            undo_manager: Undo manager for transaction logging
        """
        self.config = config
        self.undo_manager = undo_manager
        self.output_dir = config.output_dir

        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Initialized file organizer",
            output_dir=str(self.output_dir) if self.output_dir else None,
            mode="copy" if config.copy_mode else "move",
        )

    def get_output_path(
        self,
        record: FileRecord,
        is_duplicate: bool = False,
    ) -> Path:
        """Determine output path for a file.

        Args:
            record: File record
            is_duplicate: Whether file is a duplicate

        Returns:
            Output path
        """
        if not self.output_dir:
            raise ValueError("Output directory not configured")

        # Handle duplicates
        if is_duplicate:
            if record.duplicate_of:
                # Perceptual duplicate
                return (
                    self.output_dir
                    / "duplicates"
                    / "perceptual"
                    / record.duplicate_of.stem
                    / record.path.name
                )
            else:
                # Exact duplicate (use SHA256 prefix)
                sha_prefix = record.sha256[:8] if record.sha256 else "unknown"
                return (
                    self.output_dir
                    / "duplicates"
                    / "exact"
                    / sha_prefix
                    / record.path.name
                )

        # Normal organization
        file_type = record.file_type or FileType.OTHER
        type_dir = file_type.value + "s"  # e.g., "images", "videos"

        # Semantic label or cluster
        if record.label:
            subdir = self._sanitize_label(record.label)
        elif record.cluster_id is not None:
            # HDBSCAN uses -1 for noise/outliers
            if record.cluster_id == -1:
                subdir = "outliers"
            else:
                subdir = f"cluster_{record.cluster_id}"
        else:
            subdir = "uncategorized"

        return self.output_dir / type_dir / subdir / record.path.name

    def _sanitize_label(self, label: str) -> str:
        """Sanitize label for use as directory name.

        Args:
            label: Label string

        Returns:
            Sanitized label
        """
        # Replace invalid characters
        sanitized = label.replace("/", "_").replace("\\", "_").replace(":", "_")
        sanitized = sanitized.strip().lower()
        return sanitized if sanitized else "unknown"

    def organize_file(
        self,
        record: FileRecord,
        dry_run: bool = False,
    ) -> Optional[Path]:
        """Organize a single file.

        Args:
            record: File record
            dry_run: If True, simulate without actual operations

        Returns:
            Destination path or None if failed
        """
        dest_path = self.get_output_path(record, is_duplicate=record.is_duplicate)

        # Ensure parent directory exists
        if not dry_run:
            dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Handle conflicts
        if dest_path.exists():
            dest_path = self._resolve_conflict(record, dest_path)

        # Perform operation
        if dry_run:
            logger.info(
                "[DRY RUN] Would organize",
                src=str(record.path),
                dst=str(dest_path),
                mode="copy" if self.config.copy_mode else "move",
            )
            return dest_path

        try:
            if self.config.copy_mode:
                shutil.copy2(record.path, dest_path)
                operation = OperationType.COPY
                logger.debug("Copied file", src=str(record.path), dst=str(dest_path))
            else:
                shutil.move(str(record.path), str(dest_path))
                operation = OperationType.MOVE
                logger.debug("Moved file", src=str(record.path), dst=str(dest_path))

            # Log transaction
            transaction = Transaction(
                operation=operation,
                source=record.path,
                destination=dest_path,
                sha256=record.sha256 or "",
                file_type=record.file_type or FileType.OTHER,
                label=record.label,
                cluster_id=record.cluster_id,
                metadata={
                    "is_duplicate": record.is_duplicate,
                    "mime_type": record.mime_type,
                },
            )
            self.undo_manager.log_transaction(transaction)

            return dest_path

        except Exception as e:
            logger.error("Failed to organize file", src=str(record.path), error=str(e))
            return None

    def _resolve_conflict(self, record: FileRecord, dest_path: Path) -> Path:
        """Resolve filename conflict.

        Args:
            record: File record
            dest_path: Conflicting destination path

        Returns:
            New destination path
        """
        # Strategy: append SHA256 prefix or timestamp
        counter = 1
        stem = dest_path.stem
        suffix = dest_path.suffix

        # Try SHA256 suffix first
        if record.sha256:
            sha_suffix = record.sha256[:8]
            new_path = dest_path.parent / f"{stem}_{sha_suffix}{suffix}"
            if not new_path.exists():
                logger.debug("Resolved conflict with SHA256", new_path=str(new_path))
                return new_path

        # Fallback: numeric counter
        while True:
            new_path = dest_path.parent / f"{stem}_{counter}{suffix}"
            if not new_path.exists():
                logger.debug("Resolved conflict with counter", new_path=str(new_path))
                return new_path
            counter += 1
            if counter > 1000:
                logger.error("Too many conflicts")
                raise ValueError("Cannot resolve filename conflict")

    def organize_batch(
        self,
        records: list[FileRecord],
        dry_run: bool = False,
    ) -> tuple[int, int]:
        """Organize multiple files.

        Args:
            records: List of file records
            dry_run: If True, simulate without actual operations

        Returns:
            Tuple of (successful, failed) counts
        """
        logger.info("Starting batch organization", count=len(records), dry_run=dry_run)

        successful = 0
        failed = 0

        for record in records:
            try:
                result = self.organize_file(record, dry_run=dry_run)
                if result:
                    successful += 1
                else:
                    failed += 1
            except Exception as e:
                logger.error("Failed to organize file", path=str(record.path), error=str(e))
                failed += 1

        logger.info(
            "Batch organization complete",
            successful=successful,
            failed=failed,
            dry_run=dry_run,
        )

        return successful, failed

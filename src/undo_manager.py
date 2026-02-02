"""Transaction-based undo manager."""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import UUID

from src.hashing import compute_sha256
from src.models import OperationType, Transaction
from src.utils.logging import get_logger

logger = get_logger(__name__)


class UndoManager:
    """Manage transactions and provide undo functionality."""

    def __init__(self, log_file: Path):
        """Initialize undo manager.

        Args:
            log_file: Path to the transaction log file (JSONL format)
        """
        self.log_file = log_file
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        # Create log file if it doesn't exist
        if not self.log_file.exists():
            self.log_file.touch()

        logger.info("Initialized undo manager", log_file=str(log_file))

    def log_transaction(self, transaction: Transaction) -> None:
        """Log a transaction to the JSONL file.

        Args:
            transaction: Transaction to log
        """
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                # Convert transaction to JSON
                tx_dict = transaction.model_dump()
                # Convert Path objects to strings
                tx_dict["source"] = str(tx_dict["source"])
                if tx_dict["destination"]:
                    tx_dict["destination"] = str(tx_dict["destination"])
                # Convert UUID to string
                tx_dict["id"] = str(tx_dict["id"])
                # Convert datetime to ISO format
                tx_dict["timestamp"] = tx_dict["timestamp"].isoformat()

                json.dump(tx_dict, f, ensure_ascii=False)
                f.write("\n")

            logger.debug("Logged transaction", transaction_id=str(transaction.id))

        except Exception as e:
            logger.error("Failed to log transaction", transaction_id=str(transaction.id), error=str(e))

    def read_transactions(self, limit: Optional[int] = None) -> list[Transaction]:
        """Read transactions from log file.

        Args:
            limit: Optional limit on number of transactions to read (most recent)

        Returns:
            List of Transaction objects
        """
        transactions = []

        try:
            with open(self.log_file, "r", encoding="utf-8") as f:
                lines = f.readlines()

                # Reverse for most recent first
                if limit:
                    lines = lines[-limit:]

                for line in lines:
                    try:
                        tx_dict = json.loads(line)
                        # Convert strings back to proper types
                        tx_dict["source"] = Path(tx_dict["source"])
                        if tx_dict["destination"]:
                            tx_dict["destination"] = Path(tx_dict["destination"])
                        tx_dict["id"] = UUID(tx_dict["id"])
                        tx_dict["timestamp"] = datetime.fromisoformat(tx_dict["timestamp"])

                        transaction = Transaction(**tx_dict)
                        transactions.append(transaction)

                    except Exception as e:
                        logger.warning("Failed to parse transaction line", error=str(e))

            logger.debug("Read transactions", count=len(transactions))
            return transactions

        except FileNotFoundError:
            logger.warning("Log file not found", path=str(self.log_file))
            return []
        except Exception as e:
            logger.error("Failed to read transactions", error=str(e))
            return []

    def undo(
        self,
        count: int = 1,
        dry_run: bool = False,
    ) -> tuple[int, int]:
        """Undo last N transactions.

        Args:
            count: Number of transactions to undo
            dry_run: If True, only simulate without actual operations

        Returns:
            Tuple of (successful_undos, failed_undos)
        """
        logger.info("Starting undo operation", count=count, dry_run=dry_run)

        # Read all transactions
        all_transactions = self.read_transactions()

        # Filter out already undone transactions
        active_transactions = [tx for tx in all_transactions if not tx.is_undone]

        # Get last N transactions
        transactions_to_undo = list(reversed(active_transactions[-count:]))

        successful = 0
        failed = 0

        for transaction in transactions_to_undo:
            try:
                success = self._undo_single(transaction, dry_run=dry_run)
                if success:
                    successful += 1
                else:
                    failed += 1
            except Exception as e:
                logger.error(
                    "Failed to undo transaction",
                    transaction_id=str(transaction.id),
                    error=str(e),
                )
                failed += 1

        logger.info(
            "Undo operation complete",
            successful=successful,
            failed=failed,
            dry_run=dry_run,
        )

        return successful, failed

    def _undo_single(self, transaction: Transaction, dry_run: bool = False) -> bool:
        """Undo a single transaction.

        Args:
            transaction: Transaction to undo
            dry_run: If True, only simulate

        Returns:
            True if successful
        """
        logger.debug(
            "Undoing transaction",
            transaction_id=str(transaction.id),
            operation=transaction.operation.value,
            source=str(transaction.source),
            destination=str(transaction.destination) if transaction.destination else None,
        )

        if transaction.operation == OperationType.COPY:
            return self._undo_copy(transaction, dry_run=dry_run)
        elif transaction.operation == OperationType.MOVE:
            return self._undo_move(transaction, dry_run=dry_run)
        elif transaction.operation == OperationType.DELETE:
            logger.error("Cannot undo delete operation", transaction_id=str(transaction.id))
            return False
        else:
            logger.error("Unknown operation type", operation=transaction.operation.value)
            return False

    def _undo_copy(self, transaction: Transaction, dry_run: bool = False) -> bool:
        """Undo a copy operation (delete the destination).

        Args:
            transaction: Transaction to undo
            dry_run: If True, only simulate

        Returns:
            True if successful
        """
        if not transaction.destination:
            logger.error("No destination in transaction")
            return False

        dest_path = transaction.destination

        # Check if destination exists
        if not dest_path.exists():
            logger.warning("Destination file not found", path=str(dest_path))
            return False

        # Verify hash for safety
        try:
            current_hash = compute_sha256(dest_path)
            if current_hash != transaction.sha256:
                logger.warning(
                    "Destination file hash mismatch, refusing to delete",
                    path=str(dest_path),
                    expected=transaction.sha256[:16] + "...",
                    actual=current_hash[:16] + "...",
                )
                return False
        except Exception as e:
            logger.error("Failed to verify hash", path=str(dest_path), error=str(e))
            return False

        # Delete destination
        if dry_run:
            logger.info("[DRY RUN] Would delete", path=str(dest_path))
        else:
            try:
                dest_path.unlink()
                logger.info("Deleted copied file", path=str(dest_path))

                # Mark as undone in log
                undo_tx = Transaction(
                    operation=OperationType.DELETE,
                    source=dest_path,
                    destination=None,
                    sha256=transaction.sha256,
                    file_type=transaction.file_type,
                    metadata={"undo_of": str(transaction.id)},
                )
                self.log_transaction(undo_tx)

            except Exception as e:
                logger.error("Failed to delete file", path=str(dest_path), error=str(e))
                return False

        return True

    def _undo_move(self, transaction: Transaction, dry_run: bool = False) -> bool:
        """Undo a move operation (move file back to source).

        Args:
            transaction: Transaction to undo
            dry_run: If True, only simulate

        Returns:
            True if successful
        """
        if not transaction.destination:
            logger.error("No destination in transaction")
            return False

        src_path = transaction.source
        dest_path = transaction.destination

        # Check if destination exists
        if not dest_path.exists():
            logger.warning("Destination file not found", path=str(dest_path))
            return False

        # Check if source already exists (conflict)
        if src_path.exists():
            logger.error("Source path already exists, cannot undo move", path=str(src_path))
            return False

        # Verify hash
        try:
            current_hash = compute_sha256(dest_path)
            if current_hash != transaction.sha256:
                logger.warning(
                    "File hash mismatch",
                    expected=transaction.sha256[:16] + "...",
                    actual=current_hash[:16] + "...",
                )
                return False
        except Exception as e:
            logger.error("Failed to verify hash", error=str(e))
            return False

        # Move back
        if dry_run:
            logger.info("[DRY RUN] Would move", src=str(dest_path), dst=str(src_path))
        else:
            try:
                # Ensure parent directory exists
                src_path.parent.mkdir(parents=True, exist_ok=True)

                # Move file
                dest_path.rename(src_path)
                logger.info("Moved file back", src=str(dest_path), dst=str(src_path))

                # Log undo transaction
                undo_tx = Transaction(
                    operation=OperationType.MOVE,
                    source=dest_path,
                    destination=src_path,
                    sha256=transaction.sha256,
                    file_type=transaction.file_type,
                    metadata={"undo_of": str(transaction.id)},
                )
                self.log_transaction(undo_tx)

            except Exception as e:
                logger.error("Failed to move file", error=str(e))
                return False

        return True

"""Test hashing functions."""

import tempfile
from pathlib import Path

import pytest

from src.hashing import (
    compute_image_hash,
    compute_sha256,
    find_duplicates,
    hamming_distance,
)
from src.models import FileRecord, FileType


def test_compute_sha256():
    """Test SHA256 hash computation."""
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(b"test content")
        tmp_path = Path(tmp.name)

    try:
        hash1 = compute_sha256(tmp_path)
        assert len(hash1) == 64  # SHA256 produces 64 hex characters
        assert hash1 == compute_sha256(tmp_path)  # Consistent

    finally:
        tmp_path.unlink()


def test_compute_sha256_different_content():
    """Test that different content produces different hashes."""
    with tempfile.NamedTemporaryFile(delete=False) as tmp1:
        tmp1.write(b"content1")
        tmp1_path = Path(tmp1.name)

    with tempfile.NamedTemporaryFile(delete=False) as tmp2:
        tmp2.write(b"content2")
        tmp2_path = Path(tmp2.name)

    try:
        hash1 = compute_sha256(tmp1_path)
        hash2 = compute_sha256(tmp2_path)
        assert hash1 != hash2

    finally:
        tmp1_path.unlink()
        tmp2_path.unlink()


def test_find_duplicates():
    """Test finding duplicate files by hash."""
    records = [
        FileRecord(
            path=Path("file1.txt"),
            size=100,
            mtime="2025-01-01T00:00:00",
            sha256="abc123",
        ),
        FileRecord(
            path=Path("file2.txt"),
            size=100,
            mtime="2025-01-01T00:00:00",
            sha256="abc123",
        ),
        FileRecord(
            path=Path("file3.txt"),
            size=100,
            mtime="2025-01-01T00:00:00",
            sha256="def456",
        ),
    ]

    duplicates = find_duplicates(records, hash_field="sha256")

    assert len(duplicates) == 1
    assert "abc123" in duplicates
    assert len(duplicates["abc123"]) == 2


def test_hamming_distance():
    """Test Hamming distance calculation."""
    # Identical hashes
    dist = hamming_distance("a" * 16, "a" * 16)
    assert dist == 0

    # Different hashes
    hash1 = "0" * 16
    hash2 = "f" * 16
    dist = hamming_distance(hash1, hash2)
    assert dist > 0

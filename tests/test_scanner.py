"""Test file scanner."""

import tempfile
from pathlib import Path

import pytest

from src.config import Config
from src.models import FileType
from src.scanner import count_files_by_type, scan_directory


@pytest.fixture
def temp_dir():
    """Create a temporary directory with test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Create test files
        (tmp_path / "test.txt").write_text("test content")
        (tmp_path / "image.jpg").write_bytes(b"\xFF\xD8\xFF")  # JPEG header
        (tmp_path / "audio.mp3").write_bytes(b"ID3")  # MP3 header

        # Create subdirectory
        sub_dir = tmp_path / "subdir"
        sub_dir.mkdir()
        (sub_dir / "nested.txt").write_text("nested content")

        yield tmp_path


def test_scan_directory_basic(temp_dir):
    """Test basic directory scanning."""
    config = Config(input_dir=temp_dir, output_dir=temp_dir /  "output")
    records = list(scan_directory(temp_dir, config, progress=False))

    assert len(records) == 4
    assert all(r.path.exists() for r in records)
    assert all(r.size > 0 for r in records)


def test_scan_directory_file_types(temp_dir):
    """Test file type detection during scan."""
    config = Config(input_dir=temp_dir, output_dir=temp_dir / "output")
    records = list(scan_directory(temp_dir, config, progress=False))

    # Check that file types are detected
    txt_files = [r for r in records if r.path.suffix == ".txt"]
    assert all(r.file_type == FileType.DOCUMENT for r in txt_files)


def test_count_files_by_type(temp_dir):
    """Test file type counting."""
    config = Config(input_dir=temp_dir, output_dir=temp_dir / "output")
    records = list(scan_directory(temp_dir, config, progress=False))
    counts = count_files_by_type(records)

    assert "document" in counts
    assert counts["document"] >= 2  # At least 2 txt files


def test_scan_empty_directory():
    """Test scanning an empty directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        config = Config(input_dir=tmp_path, output_dir=tmp_path / "output")
        records = list(scan_directory(tmp_path, config, progress=False))

        assert len(records) == 0


def test_scan_nonexistent_directory():
    """Test scanning a nonexistent directory raises error."""
    config = Config(
        input_dir=Path("/nonexistent"),
        output_dir=Path("/output"),
    )

    with pytest.raises(FileNotFoundError):
        list(scan_directory(Path("/nonexistent"), config, progress=False))

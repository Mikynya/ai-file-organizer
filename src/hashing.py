"""File hashing for exact and perceptual duplicate detection."""

import hashlib
from pathlib import Path
from typing import Optional

import imagehash
import numpy as np
from PIL import Image

from src.models import FileType
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Buffer size for reading files (64KB)
BUFFER_SIZE = 65536


def compute_sha256(file_path: Path) -> str:
    """Compute SHA256 hash of a file.

    Args:
        file_path: Path to the file

    Returns:
        Hexadecimal SHA256 hash string

    Raises:
        IOError: If file cannot be read
    """
    sha256_hash = hashlib.sha256()

    try:
        with open(file_path, "rb") as f:
            while True:
                data = f.read(BUFFER_SIZE)
                if not data:
                    break
                sha256_hash.update(data)

        hash_value = sha256_hash.hexdigest()
        logger.debug("Computed SHA256", path=str(file_path), hash=hash_value[:16] + "...")
        return hash_value

    except Exception as e:
        logger.error("Failed to compute SHA256", path=str(file_path), error=str(e))
        raise


def compute_image_hash(
    file_path: Path,
    hash_type: str = "phash",
    hash_size: int = 8,
    rotation_invariant: bool = True,
) -> Optional[str]:
    """Compute perceptual hash of an image.

    Args:
        file_path: Path to the image file
        hash_type: Type of hash ('dhash', 'phash', 'ahash', 'whash')
        hash_size: Hash size (default: 8 for 64-bit hash)
        rotation_invariant: Compute hash invariant to 90° rotations

    Returns:
        Hexadecimal hash string or None if failed

    Raises:
        ValueError: If hash_type is invalid
    """
    hash_functions = {
        "dhash": imagehash.dhash,
        "phash": imagehash.phash,
        "ahash": imagehash.average_hash,
        "whash": imagehash.whash,
    }

    if hash_type not in hash_functions:
        raise ValueError(f"Invalid hash type: {hash_type}. Choose from {list(hash_functions.keys())}")

    try:
        with Image.open(file_path) as img:
            # Convert to RGB if necessary
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")

            hash_func = hash_functions[hash_type]
            
            if rotation_invariant:
                # Compute hash for all 4 rotations and take minimum
                # This makes the hash invariant to 90° rotations
                hashes = []
                for angle in [0, 90, 180, 270]:
                    if angle == 0:
                        rotated = img
                    else:
                        rotated = img.rotate(angle, expand=True)
                    img_hash = hash_func(rotated, hash_size=hash_size)
                    hashes.append(str(img_hash))
                
                # Use lexicographically smallest hash (consistent across rotations)
                hash_value = min(hashes)
                
                logger.debug(
                    "Computed rotation-invariant image hash",
                    path=str(file_path),
                    hash_type=hash_type,
                    hash=hash_value,
                )
            else:
                img_hash = hash_func(img, hash_size=hash_size)
                hash_value = str(img_hash)
                
                logger.debug(
                    "Computed image hash",
                    path=str(file_path),
                    hash_type=hash_type,
                    hash=hash_value,
                )
            
            return hash_value

    except Exception as e:
        logger.warning("Failed to compute image hash", path=str(file_path), error=str(e))
        return None


def compute_video_hash(
    frame_hashes: list[str],
    method: str = "average",
) -> Optional[str]:
    """Compute aggregate hash from video key frames.

    Args:
        frame_hashes: List of perceptual hashes from key frames
        method: Aggregation method ('average', 'median')

    Returns:
        Aggregated hash string or None if no frames
    """
    if not frame_hashes:
        return None

    try:
        # Convert hex strings to imagehash objects
        hash_objects = [imagehash.hex_to_hash(h) for h in frame_hashes]

        # Convert to numpy arrays
        hash_arrays = [np.array(h.hash).flatten() for h in hash_objects]
        stacked = np.vstack(hash_arrays)

        # Aggregate
        if method == "average":
            aggregated = np.mean(stacked, axis=0) > 0.5
        elif method == "median":
            aggregated = np.median(stacked, axis=0) > 0.5
        else:
            raise ValueError(f"Invalid aggregation method: {method}")

        # Reconstruct hash
        hash_size = int(np.sqrt(len(aggregated)))
        aggregated_2d = aggregated.reshape((hash_size, hash_size))
        result_hash = imagehash.ImageHash(aggregated_2d)

        hash_value = str(result_hash)
        logger.debug(
            "Computed video hash",
            num_frames=len(frame_hashes),
            method=method,
            hash=hash_value,
        )
        return hash_value

    except Exception as e:
        logger.error("Failed to compute video hash", error=str(e))
        return None


def hamming_distance(hash1: str, hash2: str) -> int:
    """Calculate Hamming distance between two hashes.

    Args:
        hash1: First hash string
        hash2: Second hash string

    Returns:
        Hamming distance (number of differing bits)
    """
    try:
        h1 = imagehash.hex_to_hash(hash1)
        h2 = imagehash.hex_to_hash(hash2)
        return h1 - h2  # ImageHash overloads subtraction for Hamming distance
    except Exception as e:
        logger.warning("Failed to compute Hamming distance", error=str(e))
        # Fallback: character-wise comparison
        return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))


def are_similar(
    hash1: str,
    hash2: str,
    threshold: int = 5,
) -> bool:
    """Check if two perceptual hashes are similar.

    Args:
        hash1: First hash string
        hash2: Second hash string
        threshold: Maximum Hamming distance for similarity (0-64)

    Returns:
        True if hashes are similar (distance <= threshold)
    """
    distance = hamming_distance(hash1, hash2)
    return distance <= threshold


def find_duplicates(
    records: list,
    hash_field: str = "sha256",
) -> dict[str, list]:
    """Find duplicate files based on hash field.

    Args:
        records: List of FileRecord objects
        hash_field: Field name to use for duplicate detection

    Returns:
        Dictionary mapping hash to list of file paths
    """
    hash_to_paths: dict[str, list] = {}

    for record in records:
        hash_value = getattr(record, hash_field, None)
        if hash_value:
            if hash_value not in hash_to_paths:
                hash_to_paths[hash_value] = []
            hash_to_paths[hash_value].append(record.path)

    # Filter to only include actual duplicates (more than one file)
    duplicates = {h: paths for h, paths in hash_to_paths.items() if len(paths) > 1}

    logger.info(
        "Duplicate detection complete",
        hash_field=hash_field,
        total_files=len(records),
        unique_hashes=len(hash_to_paths),
        duplicate_groups=len(duplicates),
    )

    return duplicates


def find_perceptual_duplicates(
    records: list,
    threshold: int = 5,
) -> list[tuple[Path, Path, int]]:
    """Find perceptually similar files.

    Args:
        records: List of FileRecord objects with perceptual_hash field
        threshold: Maximum Hamming distance for similarity

    Returns:
        List of tuples: (path1, path2, distance)
    """
    similar_pairs: list[tuple[Path, Path, int]] = []

    # Filter records with perceptual hashes
    hashed_records = [r for r in records if r.perceptual_hash]

    # Compare all pairs (O(n²) - could be optimized with LSH for large datasets)
    for i, record1 in enumerate(hashed_records):
        for record2 in hashed_records[i + 1 :]:
            if record1.perceptual_hash and record2.perceptual_hash:
                distance = hamming_distance(record1.perceptual_hash, record2.perceptual_hash)
                if distance <= threshold:
                    similar_pairs.append((record1.path, record2.path, distance))

    logger.info(
        "Perceptual duplicate detection complete",
        total_files=len(hashed_records),
        similar_pairs=len(similar_pairs),
        threshold=threshold,
    )

    return similar_pairs

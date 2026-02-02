"""Data models and type definitions for file organizer."""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator


class FileType(str, Enum):
    """File type classification."""

    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    DOCUMENT = "document"
    OTHER = "other"


class HashType(str, Enum):
    """Hash type for duplicate detection."""

    SHA256 = "sha256"
    PERCEPTUAL_IMAGE = "perceptual_image"
    PERCEPTUAL_VIDEO = "perceptual_video"


class OperationType(str, Enum):
    """File operation types for transaction log."""

    COPY = "copy"
    MOVE = "move"
    DELETE = "delete"


class FileRecord(BaseModel):
    """Metadata for a single file."""

    path: Path
    size: int = Field(gt=0, description="File size in bytes")
    mtime: datetime = Field(description="Last modification time")
    mime_type: Optional[str] = None
    file_type: Optional[FileType] = None
    sha256: Optional[str] = None
    perceptual_hash: Optional[str] = None
    embedding: Optional[list[float]] = None
    label: Optional[str] = None
    cluster_id: Optional[int] = None
    is_duplicate: bool = False
    duplicate_of: Optional[Path] = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("path", mode="before")
    @classmethod
    def validate_path(cls, v: Any) -> Path:
        """Convert string to Path."""
        if isinstance(v, str):
            return Path(v)
        return v

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True


class DuplicateInfo(BaseModel):
    """Information about duplicate files."""

    original_path: Path
    duplicate_paths: list[Path] = Field(default_factory=list)
    hash_type: HashType
    hash_value: str
    similarity_score: Optional[float] = None

    @field_validator("original_path", mode="before")
    @classmethod
    def validate_original_path(cls, v: Any) -> Path:
        """Convert string to Path."""
        if isinstance(v, str):
            return Path(v)
        return v

    @field_validator("duplicate_paths", mode="before")
    @classmethod
    def validate_duplicate_paths(cls, v: Any) -> list[Path]:
        """Convert strings to Paths."""
        if isinstance(v, list):
            return [Path(p) if isinstance(p, str) else p for p in v]
        return v


class Transaction(BaseModel):
    """Transaction record for undo functionality."""

    id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    operation: OperationType
    source: Path
    destination: Optional[Path] = None
    sha256: str
    file_type: FileType
    label: Optional[str] = None
    cluster_id: Optional[int] = None
    is_undone: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("source", "destination", mode="before")
    @classmethod
    def validate_paths(cls, v: Any) -> Optional[Path]:
        """Convert string to Path."""
        if v is None:
            return None
        if isinstance(v, str):
            return Path(v)
        return v

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True
        json_encoders = {
            Path: str,
            UUID: str,
            datetime: lambda dt: dt.isoformat(),
        }


class ClusterResult(BaseModel):
    """Result of clustering operation."""

    cluster_id: int
    label: Optional[str] = None
    file_paths: list[Path] = Field(default_factory=list)
    centroid: Optional[list[float]] = None
    size: int = 0

    @field_validator("file_paths", mode="before")
    @classmethod
    def validate_file_paths(cls, v: Any) -> list[Path]:
        """Convert strings to Paths."""
        if isinstance(v, list):
            return [Path(p) if isinstance(p, str) else p for p in v]
        return v

"""Configuration management using Pydantic settings."""

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    """Application configuration."""

    model_config = SettingsConfigDict(
        env_prefix="FO_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Directories
    input_dir: Optional[Path] = None
    output_dir: Optional[Path] = None
    models_cache: Path = Field(default=Path.home() / ".cache" / "file-organizer")
    log_file: Optional[Path] = None

    # Processing options
    mode: str = Field(default="cluster", pattern="^(classify|cluster)$")
    labels: list[str] = Field(default_factory=list)
    workers: int = Field(default=4, ge=1, le=32)
    batch_size: int = Field(default=16, ge=1, le=128)

    # Duplicate detection
    min_similarity: int = Field(default=5, ge=0, le=64, description="Hamming distance threshold")
    enable_perceptual: bool = True

    # File operations
    copy_mode: bool = True  # True = copy, False = move
    dry_run: bool = False
    min_file_size: int = Field(default=0, ge=0, description="Minimum file size in bytes")

    # Clustering
    n_clusters: Optional[int] = Field(default=None, ge=2, le=50)
    auto_clusters: bool = True
    clustering_algorithm: str = Field(default="kmeans", pattern="^(kmeans|hdbscan|agglomerative)$")

    # Video processing
    video_frame_interval: int = Field(default=5, ge=1, description="Extract frame every N seconds")
    max_video_frames: int = Field(default=10, ge=1, le=100)

    # Logging
    log_level: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    verbose: bool = False

    # Model settings
    clip_model: str = "openai/clip-vit-base-patch32"
    device: str = "cuda"  # Will auto-detect and fallback to CPU

    def __init__(self, **kwargs):  # type: ignore
        """Initialize config."""
        super().__init__(**kwargs)
        # Create cache directory if it doesn't exist
        self.models_cache.mkdir(parents=True, exist_ok=True)

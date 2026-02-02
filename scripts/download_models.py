"""Download and cache models for offline use."""

import sys
from pathlib import Path
from typing import Optional

from transformers import CLIPModel, CLIPProcessor


def check_model_exists(model_name: str, cache_dir: Path) -> bool:
    """Check if model is already cached.
    
    Args:
        model_name: HuggingFace model name
        cache_dir: Cache directory path
        
    Returns:
        True if model exists in cache, False otherwise
    """
    # Check for model files in cache directory
    # Transformers uses a hash-based directory structure
    if not cache_dir.exists():
        return False
    
    # Look for model files - if cache has any .bin or .safetensors files, model likely exists
    model_files = list(cache_dir.rglob("*.bin")) + list(cache_dir.rglob("*.safetensors"))
    config_files = list(cache_dir.rglob("config.json"))
    
    # Model is considered cached if we have both model weights and config
    return len(model_files) > 0 and len(config_files) > 0


def download_models(cache_dir: Path, force: bool = False) -> None:
    """Download all required models.

    Args:
        cache_dir: Directory to cache models
        force: Force re-download even if models exist
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Model Download Manager")
    print("=" * 60)
    print(f"Cache directory: {cache_dir}")
    print(f"Force re-download: {force}")
    print()

    # CLIP model
    model_name = "openai/clip-vit-base-patch32"
    print(f"[1/1] CLIP Model: {model_name}")
    
    # Check if model exists
    if not force and check_model_exists(model_name, cache_dir):
        print("  ✓ Model already cached, skipping download")
        print("  → Use --force to re-download")
        print()
        print("=" * 60)
        print("✓ All models ready!")
        print("=" * 60)
        return
    
    print("  ⬇ Downloading model... (this may take a few minutes)")
    
    try:
        # Download model with progress
        CLIPModel.from_pretrained(
            model_name, 
            cache_dir=str(cache_dir),
            resume_download=True  # Resume interrupted downloads
        )
        CLIPProcessor.from_pretrained(
            model_name, 
            cache_dir=str(cache_dir),
            resume_download=True
        )
        print("  ✓ Download complete")
    except Exception as e:
        print(f"  ✗ Download failed: {e}")
        sys.exit(1)

    print()
    print("=" * 60)
    print("✓ All models downloaded successfully!")
    print("=" * 60)
    print(f"Models cached in: {cache_dir}")
    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Download and cache models for offline use",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download models to default cache (/models)
  python download_models.py
  
  # Download to custom directory
  python download_models.py --cache /path/to/cache
  
  # Force re-download even if models exist
  python download_models.py --force
        """
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("/models"),
        help="Cache directory (default: /models)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if models already exist",
    )
    args = parser.parse_args()

    download_models(args.cache, force=args.force)

"""Image feature extraction using CLIP and perceptual hashing."""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from src.config import Config
from src.features.cache import EmbeddingCache
from src.utils.logging import get_logger

logger = get_logger(__name__)


class ImageFeatureExtractor:
    """Extract semantic and perceptual features from images using CLIP."""

    def __init__(self, config: Config, cache: Optional[EmbeddingCache] = None):
        """Initialize the image feature extractor.

        Args:
            config: Configuration object
            cache: Optional embedding cache
        """
        self.config = config
        self.cache = cache
        
        # Detect GPU availability
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(
            "Initializing ImageFeatureExtractor",
            device=self.device,
            model_name=config.clip_model,
        )
        
        # Log GPU information if available
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(
                "GPU detected",
                gpu_name=gpu_name,
                gpu_memory_gb=f"{gpu_memory:.2f}",
            )
        else:
            logger.info("No GPU detected, using CPU for processing")

        # Load CLIP model
        try:
            self.model = CLIPModel.from_pretrained(
                config.clip_model,
                cache_dir=str(config.models_cache),
            ).to(self.device)  # Move model to GPU if available
            
            self.processor = CLIPProcessor.from_pretrained(
                config.clip_model,
                cache_dir=str(config.models_cache),
            )
            
            self.model.eval()
            logger.info("CLIP model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            raise

    def extract_image_embedding(
        self,
        image_path: Path,
        sha256: Optional[str] = None,
    ) -> Optional[np.ndarray]:
        """Extract CLIP embedding from a single image.

        Args:
            image_path: Path to the image
            sha256: Optional SHA256 hash for caching

        Returns:
            Normalized embedding vector (512-dim) or None if failed
        """
        # Check cache first
        if sha256 and self.cache:
            cached = self.cache.get(f"clip_{sha256}")
            if cached is not None:
                return cached

        try:
            # Load and preprocess image
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode not in ("RGB", "L"):
                    img = img.convert("RGB")

                # Process image
                inputs = self.processor(images=img, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Extract features
                with torch.no_grad():
                    image_features = self.model.get_image_features(**inputs)
                    # Normalize
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                # Convert to numpy
                embedding = image_features.cpu().numpy()[0]

                # Cache if possible
                if sha256 and self.cache:
                    self.cache.set(f"clip_{sha256}", embedding)

                logger.debug("Extracted image embedding", path=str(image_path), shape=embedding.shape)
                return embedding

        except Exception as e:
            logger.error("Failed to extract image embedding", path=str(image_path), error=str(e))
            return None

    def extract_batch_embeddings(
        self,
        image_paths: list[Path],
        sha256s: Optional[list[str]] = None,
    ) -> list[Optional[np.ndarray]]:
        """Extract embeddings for multiple images in batch.

        Args:
            image_paths: List of image paths
            sha256s: Optional list of SHA256 hashes for caching

        Returns:
            List of embeddings (same length as input)
        """
        embeddings: list[Optional[np.ndarray]] = []
        batch_images = []
        batch_indices = []

        # Check cache and prepare batch
        for i, image_path in enumerate(image_paths):
            sha256 = sha256s[i] if sha256s else None

            # Check cache
            if sha256 and self.cache:
                cached = self.cache.get(f"clip_{sha256}")
                if cached is not None:
                    embeddings.append(cached)
                    continue

            # Load image for batch processing
            try:
                with Image.open(image_path) as img:
                    if img.mode not in ("RGB", "L"):
                        img = img.convert("RGB")
                    batch_images.append(img.copy())
                    batch_indices.append(i)
                    embeddings.append(None)  # Placeholder
            except Exception as e:
                logger.error("Failed to load image", path=str(image_path), error=str(e))
                embeddings.append(None)

        # Process batch
        if batch_images:
            try:
                inputs = self.processor(images=batch_images, return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    image_features = self.model.get_image_features(**inputs)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                batch_embeddings = image_features.cpu().numpy()

                # Assign to results and cache
                for batch_idx, result_idx in enumerate(batch_indices):
                    embedding = batch_embeddings[batch_idx]
                    embeddings[result_idx] = embedding

                    # Cache
                    sha256 = sha256s[result_idx] if sha256s else None
                    if sha256 and self.cache:
                        self.cache.set(f"clip_{sha256}", embedding)

                logger.debug("Extracted batch embeddings", count=len(batch_images))

            except Exception as e:
                logger.error("Failed to process batch", error=str(e))

        return embeddings

    def classify_zero_shot(
        self,
        image_path: Path,
        labels: list[str],
        threshold: float = 0.2,
    ) -> Optional[str]:
        """Classify image using zero-shot classification.

        Args:
            image_path: Path to the image
            labels: List of candidate labels
            threshold: Minimum similarity threshold (0-1)

        Returns:
            Best matching label or None if below threshold
        """
        try:
            with Image.open(image_path) as img:
                if img.mode not in ("RGB", "L"):
                    img = img.convert("RGB")

                # Prepare text prompts
                text_prompts = [f"a photo of {label}" for label in labels]

                # Process inputs
                inputs = self.processor(
                    text=text_prompts,
                    images=img,
                    return_tensors="pt",
                    padding=True,
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Get similarities
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits_per_image = outputs.logits_per_image
                    probs = logits_per_image.softmax(dim=1)

                # Get best label
                max_prob, max_idx = probs[0].max(dim=0)
                max_prob = max_prob.item()
                best_label = labels[max_idx.item()]

                if max_prob >= threshold:
                    logger.debug(
                        "Zero-shot classification",
                        path=str(image_path),
                        label=best_label,
                        confidence=max_prob,
                    )
                    return best_label
                else:
                    logger.debug(
                        "No confident label found",
                        path=str(image_path),
                        max_prob=max_prob,
                        threshold=threshold,
                    )
                    return None

        except Exception as e:
            logger.error("Failed zero-shot classification", path=str(image_path), error=str(e))
            return None

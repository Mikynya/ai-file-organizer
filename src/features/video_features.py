"""Video feature extraction using OpenCV and CLIP."""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.features.image_features import ImageFeatureExtractor
from src.utils.logging import get_logger

logger = get_logger(__name__)


class VideoFeatureExtractor:
    """Extract features from video files using key frame sampling."""

    def __init__(
        self,
        image_extractor: ImageFeatureExtractor,
        frame_interval: int = 5,
        max_frames: int = 10,
    ):
        """Initialize the feature extractor.

        Args:
            image_extractor: CLIP extractor for processing frames
            frame_interval: Extract frame every N seconds
            max_frames: Maximum number of frames to extract
        """
        self.image_extractor = image_extractor
        self.frame_interval = frame_interval
        self.max_frames = max_frames
        logger.info(
            "Initialized video feature extractor",
            frame_interval=frame_interval,
            max_frames=max_frames,
        )

    def extract_key_frames(self, video_path: Path) -> list[np.ndarray]:
        """Extract key frames from video.

        Args:
            video_path: Path to the video file

        Returns:
            List of frames as numpy arrays (RGB)
        """
        frames = []

        try:
            # Open video
            cap = cv2.VideoCapture(str(video_path))

            if not cap.isOpened():
                logger.error("Failed to open video", path=str(video_path))
                return frames

            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0

            logger.debug(
                "Video properties",
                path=str(video_path),
                fps=fps,
                total_frames=total_frames,
                duration=duration,
            )

            # Calculate frame indices to extract
            frame_step = int(fps * self.frame_interval)
            if frame_step == 0:
                frame_step = 1

            frame_indices = list(range(0, total_frames, frame_step))
            # Limit to max_frames
            if len(frame_indices) > self.max_frames:
                # Sample evenly
                step = len(frame_indices) // self.max_frames
                frame_indices = frame_indices[::step][: self.max_frames]

            # Extract frames
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()

                if ret:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)
                else:
                    logger.warning(
                        "Failed to read frame",
                        path=str(video_path),
                        frame_idx=frame_idx,
                    )

            cap.release()

            logger.debug("Extracted key frames", path=str(video_path), count=len(frames))
            return frames

        except Exception as e:
            logger.error("Failed to extract key frames", path=str(video_path), error=str(e))
            return frames

    def extract_embedding(self, video_path: Path) -> Optional[np.ndarray]:
        """Extract embedding by averaging frame embeddings.

        Args:
            video_path: Path to the video file

        Returns:
            Averaged embedding vector or None if failed
        """
        try:
            # Extract key frames
            frames = self.extract_key_frames(video_path)

            if not frames:
                logger.warning("No frames extracted", path=str(video_path))
                return None

            # Save frames to temp files and extract embeddings
            # (This is a simplified approach; could be optimized)
            embeddings = []

            from PIL import Image
            import tempfile

            for i, frame in enumerate(frames):
                try:
                    # Convert numpy array to PIL Image
                    pil_image = Image.fromarray(frame)

                    # Save to temporary file
                    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                        tmp_path = Path(tmp.name)
                        pil_image.save(tmp_path, format="JPEG")

                    # Extract embedding
                    embedding = self.image_extractor.extract_image_embedding(tmp_path)

                    # Clean up
                    tmp_path.unlink()

                    if embedding is not None:
                        embeddings.append(embedding)

                except Exception as e:
                    logger.warning("Failed to process frame", frame_idx=i, error=str(e))

            if not embeddings:
                logger.warning("No frame embeddings extracted", path=str(video_path))
                return None

            # Average embeddings
            avg_embedding = np.mean(embeddings, axis=0)
            # Normalize
            avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)

            logger.debug(
                "Extracted video embedding",
                path=str(video_path),
                num_frames=len(embeddings),
                shape=avg_embedding.shape,
            )
            return avg_embedding

        except Exception as e:
            logger.error("Failed to extract video embedding", path=str(video_path), error=str(e))
            return None

    def extract_frame_hashes(self, video_path: Path) -> list[str]:
        """Extract perceptual hashes from key frames.

        Args:
            video_path: Path to the video file

        Returns:
            List of perceptual hash strings
        """
        hashes = []

        try:
            frames = self.extract_key_frames(video_path)

            from PIL import Image
            import imagehash
            import tempfile

            for i, frame in enumerate(frames):
                try:
                    pil_image = Image.fromarray(frame)

                    # Save to temp file
                    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                        tmp_path = Path(tmp.name)
                        pil_image.save(tmp_path, format="JPEG")

                    # Compute hash
                    img_hash = imagehash.phash(Image.open(tmp_path))
                    hashes.append(str(img_hash))

                    # Clean up
                    tmp_path.unlink()

                except Exception as e:
                    logger.warning("Failed to hash frame", frame_idx=i, error=str(e))

            logger.debug("Extracted frame hashes", path=str(video_path), count=len(hashes))
            return hashes

        except Exception as e:
            logger.error("Failed to extract frame hashes", path=str(video_path), error=str(e))
            return hashes

"""Audio feature extraction using librosa."""

from pathlib import Path
from typing import Optional

import librosa
import numpy as np

from src.utils.logging import get_logger

logger = get_logger(__name__)


class AudioFeatureExtractor:
    """Extract features from audio files using librosa."""

    def __init__(self, sample_rate: int = 22050, duration: Optional[float] = None):
        """Initialize the feature extractor.

        Args:
            sample_rate: Target sample rate for audio
            duration: Optional duration limit in seconds (None = full file)
        """
        self.sample_rate = sample_rate
        self.duration = duration
        logger.info(
            "Initialized audio feature extractor",
            sample_rate=sample_rate,
            duration=duration,
        )

    def extract_features(self, audio_path: Path) -> Optional[dict]:
        """Extract comprehensive audio features.

        Args:
            audio_path: Path to the audio file

        Returns:
            Dictionary of features or None if failed
        """
        try:
            # Load audio
            y, sr = librosa.load(
                str(audio_path),
                sr=self.sample_rate,
                duration=self.duration,
                mono=True,
            )

            if len(y) == 0:
                logger.warning("Empty audio file", path=str(audio_path))
                return None

            features = {}

            # MFCCs (Mel Frequency Cepstral Coefficients)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            features["mfcc_mean"] = np.mean(mfccs, axis=1).tolist()
            features["mfcc_std"] = np.std(mfccs, axis=1).tolist()

            # Mel spectrogram
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
            # log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
            features["mel_mean"] = np.mean(mel_spec).item()
            features["mel_std"] = np.std(mel_spec).item()

            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            features["spectral_centroid_mean"] = np.mean(spectral_centroids).item()
            features["spectral_centroid_std"] = np.std(spectral_centroids).item()

            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            features["spectral_rolloff_mean"] = np.mean(spectral_rolloff).item()
            features["spectral_rolloff_std"] = np.std(spectral_rolloff).item()

            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            features["zcr_mean"] = np.mean(zcr).item()
            features["zcr_std"] = np.std(zcr).item()

            # Chroma features
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            features["chroma_mean"] = np.mean(chroma, axis=1).tolist()
            features["chroma_std"] = np.std(chroma, axis=1).tolist()

            # Tempo
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            features["tempo"] = float(tempo)

            # RMS energy
            rms = librosa.feature.rms(y=y)[0]
            features["rms_mean"] = np.mean(rms).item()
            features["rms_std"] = np.std(rms).item()

            # Duration
            features["duration"] = float(len(y) / sr)

            logger.debug("Extracted audio features", path=str(audio_path))
            return features

        except Exception as e:
            logger.error("Failed to extract audio features", path=str(audio_path), error=str(e))
            return None

    def extract_embedding(self, audio_path: Path) -> Optional[np.ndarray]:
        """Extract a feature vector for clustering.

        Args:
            audio_path: Path to the audio file

        Returns:
            Feature vector or None if failed
        """
        features = self.extract_features(audio_path)
        if not features:
            return None

        try:
            # Flatten all features into a single vector
            feature_vector = []

            # MFCCs (13 mean + 13 std = 26)
            feature_vector.extend(features["mfcc_mean"])
            feature_vector.extend(features["mfcc_std"])

            # Spectral features (6)
            feature_vector.append(features["spectral_centroid_mean"])
            feature_vector.append(features["spectral_centroid_std"])
            feature_vector.append(features["spectral_rolloff_mean"])
            feature_vector.append(features["spectral_rolloff_std"])
            feature_vector.append(features["zcr_mean"])
            feature_vector.append(features["zcr_std"])

            # Chroma (12 mean + 12 std = 24)
            feature_vector.extend(features["chroma_mean"])
            feature_vector.extend(features["chroma_std"])

            # Tempo + RMS (3)
            feature_vector.append(features["tempo"])
            feature_vector.append(features["rms_mean"])
            feature_vector.append(features["rms_std"])

            # Total: 26 + 6 + 24 + 3 = 59 features
            return np.array(feature_vector, dtype=np.float32)

        except Exception as e:
            logger.error("Failed to create embedding", path=str(audio_path), error=str(e))
            return None

    def classify_simple(self, audio_path: Path) -> Optional[str]:
        """Simple rule-based audio classification.

        Args:
            audio_path: Path to the audio file

        Returns:
            Classification label or None
        """
        features = self.extract_features(audio_path)
        if not features:
            return None

        try:
            # Simple heuristics
            rms_mean = features["rms_mean"]
            zcr_mean = features["zcr_mean"]
            spectral_centroid_mean = features["spectral_centroid_mean"]

            # Silence detection
            if rms_mean < 0.01:
                return "silence"

            # Speech vs music heuristic (very rough!)
            # Speech typically has higher ZCR and varied spectral centroid
            if zcr_mean > 0.1 and spectral_centroid_mean > 2000:
                return "speech"
            elif features["tempo"] > 60:
                return "music"
            else:
                return "ambient"

        except Exception as e:
            logger.error("Failed classification", path=str(audio_path), error=str(e))
            return None

"""File type detection using MIME types and magic numbers."""

import mimetypes
from pathlib import Path
from typing import Optional

try:
    import magic
    HAS_MAGIC = True
except ImportError:
    HAS_MAGIC = False

from src.models import FileType
from src.utils.logging import get_logger

logger = get_logger(__name__)


# File extension to FileType mapping
EXTENSION_MAP = {
    # Images
    ".jpg": FileType.IMAGE,
    ".jpeg": FileType.IMAGE,
    ".png": FileType.IMAGE,
    ".gif": FileType.IMAGE,
    ".bmp": FileType.IMAGE,
    ".webp": FileType.IMAGE,
    ".heic": FileType.IMAGE,
    ".heif": FileType.IMAGE,
    ".tiff": FileType.IMAGE,
    ".tif": FileType.IMAGE,
    ".svg": FileType.IMAGE,
    ".ico": FileType.IMAGE,
    ".raw": FileType.IMAGE,
    ".cr2": FileType.IMAGE,
    ".nef": FileType.IMAGE,
    # Audio
    ".mp3": FileType.AUDIO,
    ".wav": FileType.AUDIO,
    ".flac": FileType.AUDIO,
    ".m4a": FileType.AUDIO,
    ".aac": FileType.AUDIO,
    ".ogg": FileType.AUDIO,
    ".oga": FileType.AUDIO,
    ".wma": FileType.AUDIO,
    ".opus": FileType.AUDIO,
    ".aiff": FileType.AUDIO,
    # Video
    ".mp4": FileType.VIDEO,
    ".mkv": FileType.VIDEO,
    ".avi": FileType.VIDEO,
    ".mov": FileType.VIDEO,
    ".wmv": FileType.VIDEO,
    ".flv": FileType.VIDEO,
    ".webm": FileType.VIDEO,
    ".m4v": FileType.VIDEO,
    ".mpg": FileType.VIDEO,
    ".mpeg": FileType.VIDEO,
    ".3gp": FileType.VIDEO,
    # Documents
    ".pdf": FileType.DOCUMENT,
    ".doc": FileType.DOCUMENT,
    ".docx": FileType.DOCUMENT,
    ".txt": FileType.DOCUMENT,
    ".md": FileType.DOCUMENT,
    ".odt": FileType.DOCUMENT,
    ".rtf": FileType.DOCUMENT,
    ".tex": FileType.DOCUMENT,
    ".xls": FileType.DOCUMENT,
    ".xlsx": FileType.DOCUMENT,
    ".ppt": FileType.DOCUMENT,
    ".pptx": FileType.DOCUMENT,
    ".csv": FileType.DOCUMENT,
    ".json": FileType.DOCUMENT,
    ".xml": FileType.DOCUMENT,
    ".html": FileType.DOCUMENT,
    ".htm": FileType.DOCUMENT,
}


def detect_file_type(file_path: Path) -> tuple[FileType, Optional[str]]:
    """Detect file type using multiple methods.

    Args:
        file_path: Path to the file

    Returns:
        Tuple of (FileType, mime_type)
    """
    mime_type: Optional[str] = None

    # Method 1: python-magic (most reliable, content-based)
    if HAS_MAGIC:
        try:
            mime_type = magic.from_file(str(file_path), mime=True)
            file_type = _mime_to_filetype(mime_type)
            if file_type:
                logger.debug(
                    "Detected file type via magic",
                    path=str(file_path),
                    mime_type=mime_type,
                    file_type=file_type.value,
                )
                return file_type, mime_type
        except Exception as e:
            logger.warning("Magic detection failed", path=str(file_path), error=str(e))

    # Method 2: mimetypes module (extension-based)
    mime_type, _ = mimetypes.guess_type(str(file_path))
    if mime_type:
        file_type = _mime_to_filetype(mime_type)
        if file_type:
            logger.debug(
                "Detected file type via mimetypes",
                path=str(file_path),
                mime_type=mime_type,
                file_type=file_type.value,
            )
            return file_type, mime_type

    # Method 3: Extension mapping (fallback)
    extension = file_path.suffix.lower()
    if extension in EXTENSION_MAP:
        file_type = EXTENSION_MAP[extension]
        logger.debug(
            "Detected file type via extension",
            path=str(file_path),
            extension=extension,
            file_type=file_type.value,
        )
        return file_type, mime_type

    # Default to OTHER
    logger.debug("File type unknown, defaulting to OTHER", path=str(file_path))
    return FileType.OTHER, mime_type


def _mime_to_filetype(mime_type: str) -> Optional[FileType]:
    """Convert MIME type to FileType.

    Args:
        mime_type: MIME type string (e.g., 'image/jpeg')

    Returns:
        FileType or None if not recognized
    """
    if not mime_type:
        return None

    mime_type = mime_type.lower()

    # Image types
    if mime_type.startswith("image/"):
        return FileType.IMAGE

    # Audio types
    if mime_type.startswith("audio/"):
        return FileType.AUDIO

    # Video types
    if mime_type.startswith("video/"):
        return FileType.VIDEO

    # Document types
    document_mimes = {
        "application/pdf",
        "application/msword",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.oasis.opendocument.text",
        "text/plain",
        "text/markdown",
        "text/html",
        "text/csv",
        "application/json",
        "application/xml",
        "application/vnd.ms-excel",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.ms-powerpoint",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    }

    if mime_type in document_mimes or mime_type.startswith("text/"):
        return FileType.DOCUMENT

    return None

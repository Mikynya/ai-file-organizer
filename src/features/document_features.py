"""Document feature extraction using TF-IDF and LSA.

This module provides semantic analysis for text documents (PDF, DOC, DOCX, TXT, EPUB)
using TF-IDF vectorization and Latent Semantic Analysis (LSA) via Truncated SVD.
"""

import io
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from src.features.cache import EmbeddingCache
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Lazy imports for document parsing libraries
try:
    import PyPDF2
    HAS_PDF = True
except ImportError:
    HAS_PDF = False

try:
    import docx
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False

try:
    import ebooklib
    from ebooklib import epub
    from bs4 import BeautifulSoup
    HAS_EPUB = True
except ImportError:
    HAS_EPUB = False


class DocumentFeatureExtractor:
    """Extract semantic features from text documents using TF-IDF and LSA."""
    
    def __init__(
        self,
        cache: Optional[EmbeddingCache] = None,
        n_components: int = 128,
        max_features: int = 5000,
        min_df: int = 1,
        max_df: float = 0.95,
    ):
        """Initialize document feature extractor.
        
        Args:
            cache: Optional embedding cache for persistence
            n_components: Number of SVD components (embedding dimensions)
            max_features: Maximum number of TF-IDF features
            min_df: Minimum document frequency for terms
            max_df: Maximum document frequency for terms (filter common words)
        """
        self.cache = cache
        self.n_components = n_components
        
        # Initialize TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            stop_words='english',
            ngram_range=(1, 2),  # Include unigrams and bigrams
            strip_accents='unicode',
            lowercase=True,
        )
        
        # LSA/SVD for dimensionality reduction
        self.svd = TruncatedSVD(
            n_components=n_components,
            random_state=42,
        )
        
        # Storage for fitted models (needed for consistent embeddings)
        self._is_fitted = False
        self._corpus_texts = []
        
    def extract_text(self, file_path: Path) -> Optional[str]:
        """Extract text content from document.
        
        Args:
            file_path: Path to document
            
        Returns:
            Extracted text or None if extraction failed
        """
        suffix = file_path.suffix.lower()
        
        try:
            if suffix == '.pdf':
                return self._extract_pdf(file_path)
            elif suffix in {'.docx', '.doc'}:
                return self._extract_docx(file_path)
            elif suffix == '.txt':
                return self._extract_txt(file_path)
            elif suffix in {'.epub', '.mobi'}:
                return self._extract_epub(file_path)
            else:
                logger.warning(f"Unsupported document type: {suffix}", path=str(file_path))
                return None
        except Exception as e:
            logger.error(f"Failed to extract text from {file_path}: {e}")
            return None
    
    def _extract_pdf(self, file_path: Path) -> Optional[str]:
        """Extract text from PDF."""
        if not HAS_PDF:
            logger.warning("PyPDF2 not installed, cannot process PDF files")
            return None
        
        try:
            text_parts = []
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        text_parts.append(text)
            
            full_text = '\n'.join(text_parts)
            logger.debug(f"Extracted {len(full_text)} chars from PDF", path=str(file_path))
            return full_text if full_text.strip() else None
        except Exception as e:
            logger.error(f"PDF extraction failed: {e}", path=str(file_path))
            return None
    
    def _extract_docx(self, file_path: Path) -> Optional[str]:
        """Extract text from DOCX."""
        if not HAS_DOCX:
            logger.warning("python-docx not installed, cannot process DOCX files")
            return None
        
        try:
            doc = docx.Document(str(file_path))
            text_parts = [para.text for para in doc.paragraphs if para.text.strip()]
            full_text = '\n'.join(text_parts)
            logger.debug(f"Extracted {len(full_text)} chars from DOCX", path=str(file_path))
            return full_text if full_text.strip() else None
        except Exception as e:
            logger.error(f"DOCX extraction failed: {e}", path=str(file_path))
            return None
    
    def _extract_txt(self, file_path: Path) -> Optional[str]:
        """Extract text from plain text file."""
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        text = f.read()
                    logger.debug(f"Read {len(text)} chars from TXT ({encoding})", path=str(file_path))
                    return text if text.strip() else None
                except UnicodeDecodeError:
                    continue
            
            logger.warning(f"Could not decode text file with any encoding", path=str(file_path))
            return None
        except Exception as e:
            logger.error(f"TXT extraction failed: {e}", path=str(file_path))
            return None
    
    def _extract_epub(self, file_path: Path) -> Optional[str]:
        """Extract text from EPUB."""
        if not HAS_EPUB:
            logger.warning("ebooklib not installed, cannot process EPUB files")
            return None
        
        try:
            book = epub.read_epub(str(file_path))
            text_parts = []
            
            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    # Parse HTML content
                    soup = BeautifulSoup(item.get_content(), 'html.parser')
                    text = soup.get_text()
                    if text.strip():
                        text_parts.append(text)
            
            full_text = '\n'.join(text_parts)
            logger.debug(f"Extracted {len(full_text)} chars from EPUB", path=str(file_path))
            return full_text if full_text.strip() else None
        except Exception as e:
            logger.error(f"EPUB extraction failed: {e}", path=str(file_path))
            return None
    
    def fit_corpus(self, texts: list[str]) -> None:
        """Fit TF-IDF and SVD models on a corpus of documents.
        
        This should be called once with all documents before extracting embeddings.
        
        Args:
            texts: List of document texts
        """
        if not texts:
            logger.warning("Empty corpus provided for fitting")
            return
        
        logger.info(f"Fitting TF-IDF on corpus of {len(texts)} documents")
        
        # Fit TF-IDF
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        
        # Fit SVD for dimensionality reduction
        logger.info(f"Applying LSA/SVD (reducing to {self.n_components} dimensions)")
        self.svd.fit(tfidf_matrix)
        
        self._is_fitted = True
        self._corpus_texts = texts
        
        explained_variance = self.svd.explained_variance_ratio_.sum()
        logger.info(f"SVD explained variance: {explained_variance:.2%}")
    
    def extract_embedding(
        self,
        file_path: Path,
        sha256: Optional[str] = None,
    ) -> Optional[np.ndarray]:
        """Extract semantic embedding from document using TF-IDF + LSA.
        
        Args:
            file_path: Path to document
            sha256: Optional file hash for caching
            
        Returns:
            Embedding vector (n_components dimensions) or None if extraction failed
        """
        # Check cache
        if self.cache and sha256:
            cached = self.cache.get(sha256)
            if cached is not None:
                logger.debug(f"Using cached embedding", path=str(file_path))
                return cached
        
        # Extract text
        text = self.extract_text(file_path)
        if not text or len(text.strip()) < 10:  # Minimum text length
            logger.warning(f"Insufficient text extracted", path=str(file_path))
            return None
        
        # If not fitted, fit on this single document
        if not self._is_fitted:
            logger.warning("Extractor not fitted on corpus, fitting on single document")
            self.fit_corpus([text])
        
        # Transform to TF-IDF
        try:
            tfidf_vector = self.vectorizer.transform([text])
            
            # Apply SVD
            embedding = self.svd.transform(tfidf_vector)
            embedding = embedding[0]  # Get first (and only) row
            
            # Normalize to unit length
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            # Cache result
            if self.cache and sha256:
                self.cache.set(sha256, embedding)
                logger.debug(f"Cached document embedding", path=str(file_path))
            
            logger.debug(f"Extracted embedding (dim={len(embedding)})", path=str(file_path))
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to extract embedding: {e}", path=str(file_path))
            return None
    
    def extract_embeddings_batch(
        self,
        file_paths: list[Path],
        sha256s: Optional[list[str]] = None,
    ) -> list[Optional[np.ndarray]]:
        """Extract embeddings for multiple documents efficiently.
        
        Args:
            file_paths: List of document paths
            sha256s: Optional list of file hashes for caching
            
        Returns:
            List of embeddings (or None for failed extractions)
        """
        if sha256s is None:
            sha256s = [None] * len(file_paths)
        
        # Extract all texts
        texts = []
        valid_indices = []
        
        for i, file_path in enumerate(file_paths):
            text = self.extract_text(file_path)
            if text and len(text.strip()) >= 10:
                texts.append(text)
                valid_indices.append(i)
        
        if not texts:
            logger.warning("No valid documents to process")
            return [None] * len(file_paths)
        
        # Fit on corpus if not already fitted
        if not self._is_fitted:
            self.fit_corpus(texts)
        
        # Extract embeddings
        embeddings = [None] * len(file_paths)
        
        for idx, text in zip(valid_indices, texts):
            file_path = file_paths[idx]
            sha256 = sha256s[idx]
            
            # Try cache first
            if self.cache and sha256:
                cached = self.cache.get(sha256)
                if cached is not None:
                    embeddings[idx] = cached
                    continue
            
            # Compute embedding
            try:
                tfidf_vector = self.vectorizer.transform([text])
                embedding = self.svd.transform(tfidf_vector)[0]
                
                # Normalize
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
                
                embeddings[idx] = embedding
                
                # Cache
                if self.cache and sha256:
                    self.cache.set(sha256, embedding)
                    
            except Exception as e:
                logger.error(f"Failed to extract embedding: {e}", path=str(file_path))
                embeddings[idx] = None
        
        logger.info(f"Extracted {sum(1 for e in embeddings if e is not None)}/{len(file_paths)} document embeddings")
        return embeddings

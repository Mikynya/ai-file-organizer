"""Clustering algorithms for semantic grouping."""

from typing import Optional, Union

import numpy as np
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans, HDBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize



from src.models import ClusterResult
from src.utils.logging import get_logger

logger = get_logger(__name__)


def reduce_dimensions(
    embeddings: np.ndarray,
    n_components: int = 50,
    explained_variance: float = 0.95,
) -> np.ndarray:
    """Reduce dimensionality using PCA.

    Args:
        embeddings: Input embeddings (n_samples, n_features)
        n_components: Target number of components
        explained_variance: Minimum explained variance to retain

    Returns:
        Reduced embeddings
    """
    if embeddings.shape[1] <= n_components:
        logger.debug("Skipping PCA, already low dimensional")
        return embeddings

    try:
        pca = PCA(n_components=min(n_components, embeddings.shape[0], embeddings.shape[1]))
        reduced = pca.fit_transform(embeddings)

        actual_variance = np.sum(pca.explained_variance_ratio_)
        logger.info(
            "PCA dimensionality reduction",
            original_dims=embeddings.shape[1],
            reduced_dims=reduced.shape[1],
            explained_variance=actual_variance,
        )

        return reduced

    except Exception as e:
        logger.error("PCA failed", error=str(e))
        return embeddings


def find_optimal_clusters(
    embeddings: np.ndarray,
    min_clusters: int = 2,
    max_clusters: int = 20,
    method: str = "silhouette",
) -> int:
    """Find optimal number of clusters.

    Args:
        embeddings: Input embeddings
        min_clusters: Minimum number of clusters to try
        max_clusters: Maximum number of clusters to try
        method: Method to use ('silhouette' or 'elbow')

    Returns:
        Optimal number of clusters
    """
    if len(embeddings) < min_clusters:
        return max(2, len(embeddings))

    scores = []
    k_range = range(min_clusters, min(max_clusters + 1, len(embeddings)))

    for k in k_range:
        try:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings)

            if method == "silhouette":
                score = silhouette_score(embeddings, labels)
                scores.append(score)
            else:  # elbow method uses inertia
                scores.append(-kmeans.inertia_)  # Negative for maximization

        except Exception as e:
            logger.warning("Failed to evaluate k", k=k, error=str(e))
            scores.append(-np.inf)

    # Find best k
    best_idx = np.argmax(scores)
    best_k = list(k_range)[best_idx]

    logger.info(
        "Optimal cluster search",
        method=method,
        best_k=best_k,
        best_score=scores[best_idx],
    )

    return best_k


def cluster_kmeans(
    embeddings: np.ndarray,
    n_clusters: Optional[int] = None,
    auto_tune: bool = True,
) -> np.ndarray:
    """Cluster embeddings using K-means.

    Args:
        embeddings: Input embeddings (n_samples, n_features)
        n_clusters: Number of clusters (None for auto-detection)
        auto_tune: Automatically find optimal k

    Returns:
        Cluster labels array
    """
    if n_clusters is None and auto_tune:
        n_clusters = find_optimal_clusters(embeddings)
    elif n_clusters is None:
        n_clusters = min(5, max(2, len(embeddings) // 10))

    logger.info("Running K-means clustering", n_clusters=n_clusters, n_samples=len(embeddings))

    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)

        unique_labels = np.unique(labels)
        logger.info("K-means complete", n_clusters=len(unique_labels))

        return labels

    except Exception as e:
        logger.error("K-means clustering failed", error=str(e))
        # Fallback: all in one cluster
        return np.zeros(len(embeddings), dtype=int)


def cluster_hdbscan(
    embeddings: np.ndarray,
    min_cluster_size: int = 5,
    min_samples: int = 3,
) -> np.ndarray:
    """Cluster embeddings using HDBSCAN.

    Args:
        embeddings: Input embeddings
        min_cluster_size: Minimum cluster size
        min_samples: Minimum samples for core points

    Returns:
        Cluster labels array (-1 for noise)
    """


    logger.info(
        "Running HDBSCAN clustering",
        n_samples=len(embeddings),
        min_cluster_size=min_cluster_size,
    )

    try:
        clusterer = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean",
        )
        labels = clusterer.fit_predict(embeddings)

        unique_labels = np.unique(labels)
        n_noise = np.sum(labels == -1)

        logger.info(
            "HDBSCAN complete",
            n_clusters=len(unique_labels) - (1 if -1 in unique_labels else 0),
            n_noise=n_noise,
        )

        return labels

    except Exception as e:
        logger.error("HDBSCAN clustering failed", error=str(e))
        return cluster_kmeans(embeddings, auto_tune=True)


def cluster_agglomerative(
    embeddings: np.ndarray,
    n_clusters: Optional[int] = None,
) -> np.ndarray:
    """Cluster embeddings using Agglomerative (hierarchical) clustering.

    Args:
        embeddings: Input embeddings
        n_clusters: Number of clusters (None for auto-detection)

    Returns:
        Cluster labels array
    """
    if n_clusters is None:
        n_clusters = find_optimal_clusters(embeddings)

    logger.info(
        "Running Agglomerative clustering",
        n_clusters=n_clusters,
        n_samples=len(embeddings),
    )

    try:
        clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
        labels = clusterer.fit_predict(embeddings)

        logger.info("Agglomerative clustering complete")
        return labels

    except Exception as e:
        logger.error("Agglomerative clustering failed", error=str(e))
        return np.zeros(len(embeddings), dtype=int)


def cluster_embeddings(
    embeddings: np.ndarray,
    algorithm: str = "kmeans",
    n_clusters: Optional[int] = None,
    reduce_dims: bool = True,
) -> np.ndarray:
    """Main clustering function with algorithm selection.

    Args:
        embeddings: Input embeddings
        algorithm: Clustering algorithm ('kmeans', 'hdbscan', 'agglomerative')
        n_clusters: Number of clusters (for algorithms that need it)
        reduce_dims: Apply PCA before clustering

    Returns:
        Cluster labels array
    """
    # Normalize embeddings
    embeddings_normalized = normalize(embeddings)

    # Dimensionality reduction
    if reduce_dims and embeddings_normalized.shape[1] > 50:
        embeddings_normalized = reduce_dimensions(embeddings_normalized, n_components=50)

    # Cluster
    if algorithm == "kmeans":
        return cluster_kmeans(embeddings_normalized, n_clusters=n_clusters, auto_tune=(n_clusters is None))
    elif algorithm == "hdbscan":
        return cluster_hdbscan(embeddings_normalized)
    elif algorithm == "agglomerative":
        return cluster_agglomerative(embeddings_normalized, n_clusters=n_clusters)
    else:
        logger.warning(f"Unknown algorithm: {algorithm}, using kmeans")
        return cluster_kmeans(embeddings_normalized, n_clusters=n_clusters)

"""Test clustering algorithms."""

import numpy as np
import pytest

from src.clustering import (
    cluster_agglomerative,
    cluster_embeddings,
    cluster_hdbscan,
    cluster_kmeans,
    find_optimal_clusters,
    reduce_dimensions,
)


def test_reduce_dimensions():
    """Test PCA dimensionality reduction."""
    embeddings = np.random.rand(100, 512)
    reduced = reduce_dimensions(embeddings, n_components=50)

    assert reduced.shape == (100, 50)


def test_reduce_dimensions_low_dim():
    """Test that low-dimensional data is not reduced."""
    embeddings = np.random.rand(100, 10)
    reduced = reduce_dimensions(embeddings, n_components=50)

    assert reduced.shape == embeddings.shape


def test_cluster_kmeans():
    """Test K-means clustering."""
    embeddings = np.random.rand(100, 10)
    labels = cluster_kmeans(embeddings, n_clusters=3, auto_tune=False)

    assert len(labels) == 100
    assert len(np.unique(labels)) <= 3


def test_cluster_kmeans_auto():
    """Test K-means with auto cluster detection."""
    embeddings = np.random.rand(50, 10)
    labels = cluster_kmeans(embeddings, n_clusters=None, auto_tune=True)

    assert len(labels) == 50
    assert len(np.unique(labels)) >= 2


def test_cluster_agglomerative():
    """Test agglomerative clustering."""
    embeddings = np.random.rand(50, 10)
    labels = cluster_agglomerative(embeddings, n_clusters=5)

    assert len(labels) == 50
    assert len(np.unique(labels)) == 5


def test_cluster_embeddings():
    """Test main clustering function."""
    embeddings = np.random.rand(100, 512)

    # Test with different algorithms
    labels_kmeans = cluster_embeddings(embeddings, algorithm="kmeans", n_clusters=5)
    assert len(labels_kmeans) == 100

    labels_agglom = cluster_embeddings(embeddings, algorithm="agglomerative", n_clusters=5)
    assert len(labels_agglom) == 100


def test_find_optimal_clusters():
    """Test optimal cluster finding."""
    embeddings = np.random.rand(50, 10)
    k = find_optimal_clusters(embeddings, min_clusters=2, max_clusters=10)

    assert 2 <= k <= 10


def test_cluster_small_dataset():
    """Test clustering with very small dataset."""
    embeddings = np.random.rand(5, 10)
    labels = cluster_kmeans(embeddings, n_clusters=2, auto_tune=False)

    assert len(labels) == 5

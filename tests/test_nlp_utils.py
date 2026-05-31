"""
test_nlp_utils.py
-----------------
Unit tests for the NLP utilities module.

Run with: pytest tests/test_nlp_utils.py -v

NOTE: These tests load the NLP model, so the first run may take ~30 seconds.
      Subsequent runs are fast due to model caching.
"""

import pytest
import numpy as np
from app.nlp_utils import generate_embedding, compute_similarity, run_full_analysis


class TestGenerateEmbedding:
    """Tests for the embedding generation function."""

    def test_returns_numpy_array(self):
        """Output should be a numpy array."""
        emb = generate_embedding("Python developer with Django skills")
        assert isinstance(emb, np.ndarray)

    def test_correct_dimensions(self):
        """all-MiniLM-L6-v2 always produces 384-dim vectors."""
        emb = generate_embedding("Software engineer resume")
        assert emb.shape == (384,)

    def test_different_texts_different_embeddings(self):
        """Two different texts should produce different embeddings."""
        emb1 = generate_embedding("Python developer")
        emb2 = generate_embedding("Graphic designer")
        assert not np.allclose(emb1, emb2)

    def test_similar_texts_close_embeddings(self):
        """Semantically similar texts should have high cosine similarity."""
        emb1 = generate_embedding("I build web applications with Python")
        emb2 = generate_embedding("I develop websites using Python")
        similarity = float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))
        assert similarity > 0.85, f"Expected similarity > 0.85, got {similarity:.3f}"

    def test_empty_text_raises(self):
        """Empty string should raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            generate_embedding("")

    def test_long_text_truncated_without_error(self):
        """Very long text should be truncated and processed without error."""
        long_text = "Python developer " * 1000   # ~17,000 characters
        emb = generate_embedding(long_text)
        assert emb.shape == (384,)


class TestComputeSimilarity:
    """Tests for the cosine similarity function."""

    def test_identical_vectors_score_100(self):
        """Two identical vectors should give ~100% similarity."""
        emb = generate_embedding("Python machine learning engineer")
        score = compute_similarity(emb, emb)
        assert score > 99.0, f"Expected ~100, got {score}"

    def test_score_in_valid_range(self):
        """Score must always be between 0 and 100."""
        emb1 = generate_embedding("frontend react developer")
        emb2 = generate_embedding("backend database administrator")
        score = compute_similarity(emb1, emb2)
        assert 0.0 <= score <= 100.0

    def test_similar_texts_high_score(self):
        """Semantically similar texts should score above 60%."""
        emb1 = generate_embedding("Experienced Python engineer with Flask and REST API skills")
        emb2 = generate_embedding("Looking for a Python developer with web framework experience")
        score = compute_similarity(emb1, emb2)
        assert score > 60.0, f"Expected > 60%, got {score:.2f}%"


class TestRunFullAnalysis:
    """Integration tests for the full NLP pipeline."""

    def test_returns_required_keys(self):
        """Result dict must contain all expected keys."""
        result = run_full_analysis(
            "Python developer with 3 years Django experience",
            "We need a Python engineer with web development skills"
        )
        assert "similarity_score" in result
        assert "processing_time" in result
        assert "resume_embedding" in result
        assert "jd_embedding" in result

    def test_processing_time_reasonable(self):
        """Should complete in under 5 seconds on CPU."""
        result = run_full_analysis(
            "Software engineer with Java Spring Boot Kubernetes experience",
            "Java backend developer needed with Kubernetes and AWS"
        )
        assert result["processing_time"] < 5.0, (
            f"Took {result['processing_time']:.2f}s — exceeds 5s budget"
        )

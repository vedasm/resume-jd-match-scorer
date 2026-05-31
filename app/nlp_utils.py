"""
nlp_utils.py
------------
All NLP logic lives here.

WHAT WE USE:
  Model: 'all-MiniLM-L6-v2' from Sentence-Transformers
  - "all"       → trained on many diverse datasets
  - "MiniLM"    → Mini Language Model (fast, lightweight)
  - "L6"        → 6 transformer layers
  - "v2"        → version 2 (improved accuracy)
  - Output size → 384-dimensional vector

WHY THIS MODEL:
  ✓ Fast on CPU (no GPU needed)
  ✓ ~80MB size (reasonable for Docker)
  ✓ Excellent semantic understanding
  ✓ Perfect for semantic similarity tasks

COSINE SIMILARITY EXPLAINED:
  Imagine two arrows pointing in 3D space.
  Cosine similarity measures the angle between them.
  - Same direction (0° angle) → similarity = 1.0 → identical meaning
  - Right angle (90°)         → similarity = 0.0 → unrelated
  - Opposite direction (180°) → similarity = -1.0 → opposite meaning
  
  Text embeddings are always positive, so we get 0.0–1.0.
  We multiply by 100 to get a percentage.
"""

import time
import logging
import os
from functools import lru_cache

import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load variables from .env file into the environment
load_dotenv()

logger = logging.getLogger(__name__)

# Read model name from environment (default to the recommended model)
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Maximum text length to process
# all-MiniLM-L6-v2 works best with ≤256 tokens (~1000 words)
# We allow a generous character limit and let the model handle truncation
MAX_TEXT_LENGTH = 8000  # characters


@lru_cache(maxsize=1)
def load_model() -> SentenceTransformer:
    """
    Load the sentence transformer model into memory.

    KEY CONCEPT — @lru_cache:
      Without caching: model loads fresh on EVERY request → 5-10 seconds each time
      With @lru_cache:  model loads ONCE, then reused  → < 1 second per request

    lru_cache(maxsize=1) stores exactly 1 result (the loaded model).
    The function args are used as the cache key; since load_model() has
    no args, it's always the same key → always returns the cached model.

    Returns:
        The loaded SentenceTransformer model
    """
    logger.info(f"Loading embedding model: '{MODEL_NAME}'")
    logger.info("This takes 10–30 seconds on the first run (downloading weights)...")

    t0 = time.time()
    model = SentenceTransformer(MODEL_NAME)
    elapsed = time.time() - t0

    logger.info(f"Model ready in {elapsed:.1f}s")
    return model


def generate_embedding(text: str) -> np.ndarray:
    """
    Convert text into a 384-dimensional embedding vector.

    HOW IT WORKS:
      1. Text is tokenized into sub-word pieces
      2. Passed through 6 transformer layers (MiniLM)
      3. The final hidden states are pooled into one vector
      4. The vector is L2-normalized (so cosine similarity = dot product)

    Args:
        text: Input text (resume or job description content)

    Returns:
        NumPy array of shape (384,)

    Raises:
        ValueError: If text is empty
    """
    if not text or not text.strip():
        raise ValueError("Cannot generate embedding: text is empty.")

    # Truncate if text exceeds our limit
    if len(text) > MAX_TEXT_LENGTH:
        logger.warning(
            f"Text truncated from {len(text)} to {MAX_TEXT_LENGTH} characters"
        )
        text = text[:MAX_TEXT_LENGTH]

    # Get the cached model
    model = load_model()

    # model.encode() → numpy array of shape (384,)
    # show_progress_bar=False keeps logs clean
    embedding: np.ndarray = model.encode(
        text,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True
    )

    logger.debug(f"Embedding generated: shape={embedding.shape}, dtype={embedding.dtype}")
    # Ensure numeric stability: L2-normalize the vector
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm

    return embedding


def compute_similarity(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
    """
    Compute cosine similarity between two embedding vectors.

    sklearn's cosine_similarity expects 2D arrays: (n_samples, n_features)
    Our embeddings are 1D: (384,) → we reshape to (1, 384) before calling.

    Args:
        emb_a: Embedding of text A (e.g., resume)
        emb_b: Embedding of text B (e.g., job description)

    Returns:
        Similarity as a percentage float, e.g. 78.5
    """
    # Reshape: (384,) → (1, 384)
    a = emb_a.reshape(1, -1)
    b = emb_b.reshape(1, -1)

    # cosine_similarity returns a 2D array: [[score]]
    # We extract the single value with [0][0]
    raw_score: float = cosine_similarity(a, b)[0][0]

    # Convert 0.0–1.0 → 0.0–100.0
    percentage = float(raw_score) * 100.0

    # Safety clamp (should never be needed, but defensive programming)
    percentage = max(0.0, min(100.0, percentage))

    return round(percentage, 2)


def run_full_analysis(resume_text: str, jd_text: str) -> dict:
    """
    Orchestrates the complete NLP pipeline.

    Steps:
      1. Generate embedding for resume text
      2. Generate embedding for JD text
      3. Compute cosine similarity
      4. Return results + timing info

    Args:
        resume_text: Cleaned text from the resume PDF
        jd_text:     Cleaned text from the job description PDF

    Returns:
        dict with keys: similarity_score, resume_embedding,
                        jd_embedding, processing_time
    """
    t_start = time.time()

    logger.info("Generating resume embedding...")
    resume_emb = generate_embedding(resume_text)

    logger.info("Generating job description embedding...")
    jd_emb = generate_embedding(jd_text)

    logger.info("Computing cosine similarity...")
    score = compute_similarity(resume_emb, jd_emb)

    elapsed = time.time() - t_start
    logger.info(f"NLP analysis done: score={score:.2f}%, time={elapsed:.2f}s")

    return {
        "similarity_score":  score,
        "resume_embedding":  resume_emb,
        "jd_embedding":      jd_emb,
        "processing_time":   round(elapsed, 3)
    }

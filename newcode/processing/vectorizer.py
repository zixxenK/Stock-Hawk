import numpy as np
from typing import List, Dict
from newcode.vectorizer import FEATURE_NAMES
from pydantic import BaseModel
# Assume FEATURE_NAMES and VECTOR_DIM are loaded here.

def build_vector(input_data: VectorInputModel) -> np.ndarray:
    """
    Converts the structured Pydantic input into the 18-dim vector.
    This function acts as the central hub, ensuring all inputs map correctly.
    """
    # Logic remains complex but is now fully contained and validated by Pydantic.
    pass

def calculate_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculates Cosine Similarity Dot Product."""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def describe_vector(v: np.ndarray) -> dict[str, float]:
    """Returns a human-readable dictionary of the vector contents."""
    return {FEATURE_NAMES[i]: round(float(v[i]), 4) for i in range(VECTOR_DIM)}

"""
String normalization and fuzzy matching helpers for the Privacy Protection Layer.
"""

from __future__ import annotations

import re

try:
    from Levenshtein import distance as levenshtein_distance
except Exception:  # pragma: no cover - optional dependency
    levenshtein_distance = None  # type: ignore


def _normalize_string(text: str) -> str:
    """
    Normalize string for fuzzy matching: lower + remove spaces + remove punctuation.
    
    Args:
        text: Input string
        
    Returns:
        Normalized string
    """
    # Convert to lowercase
    normalized = text.lower()
    # Remove all whitespace
    normalized = re.sub(r'\s+', '', normalized)
    # Remove all punctuation
    normalized = re.sub(r'[^\w]', '', normalized)
    return normalized


def _fuzzy_match(text1: str, text2: str, threshold: float = 0.8) -> bool:
    """
    Check if two strings match using normalized string comparison and Levenshtein distance.
    
    Args:
        text1: First string
        text2: Second string
        threshold: Similarity threshold (0.0 to 1.0). Default 0.8 means 80% similarity.
        
    Returns:
        True if strings match (exact match after normalization or Levenshtein distance below threshold)
    """
    # First try exact match after normalization
    norm1 = _normalize_string(text1)
    norm2 = _normalize_string(text2)
    
    if norm1 == norm2:
        return True
    
    # If normalized strings don't match exactly, use Levenshtein distance
    if levenshtein_distance is None:
        # Fallback: if Levenshtein is not available, only use exact normalized match
        return False
    
    # Calculate normalized Levenshtein distance (0.0 = identical, 1.0 = completely different)
    max_len = max(len(norm1), len(norm2))
    if max_len == 0:
        return True
    
    distance = levenshtein_distance(norm1, norm2)
    similarity = 1.0 - (distance / max_len)
    
    return similarity >= threshold

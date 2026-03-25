"""
Privacy Protection Layer package for Android-Lab.

Re-exports the public API so that consumers can do:
    from utils_mobile.privacy import PrivacyProtectionLayer, get_privacy_layer, ...
"""

from .constants import GLINER_PII_LABELS, GLINER_DETECTION_THRESHOLD, PII_FIXED_PLACEHOLDER
from .layer import (
    PrivacyProtectionLayer,
    get_privacy_layer,
    set_privacy_layer,
)

__all__ = [
    "PrivacyProtectionLayer",
    "get_privacy_layer",
    "set_privacy_layer",
    "GLINER_PII_LABELS",
    "GLINER_DETECTION_THRESHOLD",
    "PII_FIXED_PLACEHOLDER",
]

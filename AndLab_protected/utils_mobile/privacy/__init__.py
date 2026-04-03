"""
Privacy Protection Layer package for Android-Lab.

Re-exports the public API so that consumers can do:
    from utils_mobile.privacy import PrivacyProtectionLayer, get_privacy_layer, ...
"""

from .constants import GLINER_PII_LABELS, GLINER_DETECTION_THRESHOLD
from .runtime import (
    BasePrivacyProtectionLayer,
    NoPrivacyProtectionLayer,
    PrivacyConfig,
    create_privacy_layer,
    get_privacy_layer,
    register_privacy_strategy,
    set_privacy_layer,
)
from .layer import (
    PrivacyProtectionLayer,
    cloud_agent_compute_with_tokens,
)

__all__ = [
    "BasePrivacyProtectionLayer",
    "NoPrivacyProtectionLayer",
    "PrivacyConfig",
    "PrivacyProtectionLayer",
    "create_privacy_layer",
    "get_privacy_layer",
    "register_privacy_strategy",
    "set_privacy_layer",
    "cloud_agent_compute_with_tokens",
    "GLINER_PII_LABELS",
    "GLINER_DETECTION_THRESHOLD",
]

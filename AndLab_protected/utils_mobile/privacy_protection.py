"""
Privacy Protection Layer for Android-Lab — backward-compatible re-export shim.

All implementation has been moved to the ``utils_mobile.privacy`` package.
This file exists solely to preserve existing import paths such as:

    from utils_mobile.privacy_protection import get_privacy_layer
    from utils_mobile.privacy_protection import cloud_agent_compute_with_tokens
    from utils_mobile.privacy_protection import PrivacyProtectionLayer, set_privacy_layer
"""

from utils_mobile.privacy import (  # noqa: F401
    PrivacyProtectionLayer,
    get_privacy_layer,
    set_privacy_layer,
    cloud_agent_compute_with_tokens,
    GLINER_PII_LABELS,
    GLINER_DETECTION_THRESHOLD,
)

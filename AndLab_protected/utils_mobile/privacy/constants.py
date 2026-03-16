"""
Constants for the Privacy Protection Layer.

Includes hash alphabet, GLiNER PII detection labels, detection threshold,
and XML structural keywords exempt from anonymization.
"""

# A small alphabet to keep hashes in [0-9a-z]
_HASH_ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyz"

# GLiNER PII detection labels - all PII categories
# These labels are used for GLiNER model to detect PII entities
GLINER_PII_LABELS = [
    # Personal information
    "name",                       # Full names
    "first name",                 # First names  
    "last name",                  # Last names
    "name medical professional",  # Healthcare provider names
    "person name",                # Person names (alternative)
    "dob",                        # Date of birth
    "age",                        # Age information
    "gender",                     # Gender identifiers
    "marital status",             # Marital status
    # Contact information
    "email",                      # Email addresses
    "email address",              # Email addresses (alternative)
    "phone number",               # Phone numbers
    "ip address",                 # IP addresses
    "url",                        # URLs
    "address",                    # Addresses
    "location address",           # Street addresses
    "location street",            # Street names
    "location city",              # City names
    "location state",             # State/province names
    "location country",           # Country names
    "location zip",               # ZIP/postal codes
    # Financial information
    "account number",             # Account numbers
    "bank account",               # Bank account numbers
    "routing number",             # Routing numbers
    "credit card",                # Credit card numbers
    "credit card expiration",     # Card expiration dates  
    "cvv",                        # CVV/security codes
    "ssn",                        # Social Security Numbers
    "money",                      # Monetary amounts
    # Healthcare information
    "condition",                  # Medical conditions
    "medical process",            # Medical procedures
    "drug",                       # Drugs
    "dose",                       # Dosage information
    "blood type",                 # Blood types
    "injury",                     # Injuries
    "organization medical facility", # Healthcare facility names
    "healthcare number",          # Healthcare numbers
    "medical code",               # Medical codes
    # ID information
    "passport number",            # Passport numbers
    "driver license",             # Driver's license numbers
    "username",                   # Usernames
    "password",                   # Passwords
    "vehicle id",                 # Vehicle IDs
]

# GLiNER detection threshold (configurable)
GLINER_DETECTION_THRESHOLD = 0.5

# XML keywords and structural elements that should never be masked
# These are common in compressed XML format like: [id] url#class ;click ; ;;text:
# Note: Only structural elements are exempted, not user-visible text content
_XML_EXEMPT_KEYWORDS = {
    # Structural symbols (single chars)
    "[", "]", ";", ":", "#",
    # Multi-char structural elements
    "bounds", ";;", "url#",
    # Common XML/Android component class names (structural, not content)
    "TextView", "Button", "ImageButton", "ImageView", "Layout", 
    "LinearLayout", "RelativeLayout", "FrameLayout", "ViewGroup", "View",
    "RecyclerView", "ScrollView", "EditText", "CheckBox", "RadioButton",
    # XML attribute names (structural, not values)
    "click", "clickable", "focusable", "selected", "checked", "enabled",
    "scrollable", "long-clickable", "password", "focused", "checkable",
    "NAF", "index", "text", "resource-id", "class", "package", "content-desc",
    # Android package prefixes (structural)
    "android.widget", "android.view", "androidx",
    # Common separators and formatting
    "The current screenshot's description is shown",
}

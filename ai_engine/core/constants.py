"""
QBITEL Engine - Constants and Configuration Limits

This module defines system-wide constants including size limits, timeouts,
and other configurable thresholds for security and performance.
"""

# =============================================================================
# INPUT SIZE LIMITS (bytes)
# =============================================================================

# Maximum payload sizes
MAX_PAYLOAD_SIZE_DEFAULT = 10 * 1024 * 1024      # 10 MB - default for most endpoints
MAX_PAYLOAD_SIZE_JSON = 5 * 1024 * 1024          # 5 MB - JSON payloads
MAX_PAYLOAD_SIZE_FORM = 1 * 1024 * 1024          # 1 MB - form data
MAX_PAYLOAD_SIZE_FILE = 50 * 1024 * 1024         # 50 MB - file uploads

# Protocol discovery limits
MAX_PROTOCOL_MESSAGE_SIZE = 10 * 1024 * 1024     # 10 MB - single protocol message
MAX_PROTOCOL_BATCH_SIZE = 100                     # Max messages per batch
MAX_PROTOCOL_SAMPLE_COUNT = 1000                  # Max samples for discovery

# Field detection limits
MAX_FIELD_DETECTION_SIZE = 1 * 1024 * 1024       # 1 MB - field detection input
MAX_SEQUENCE_LENGTH = 512                         # BiLSTM-CRF max sequence

# LLM/Copilot limits
MAX_COPILOT_QUERY_LENGTH = 10_000                # Characters in copilot query
MAX_LLM_CONTEXT_LENGTH = 128_000                 # Tokens (varies by model)
MAX_RAG_DOCUMENTS = 10                            # Documents per RAG query

# API limits
MAX_QUERY_PARAM_LENGTH = 2048                    # URL query parameter length
MAX_HEADER_SIZE = 8192                           # HTTP header size
MAX_URI_LENGTH = 8192                            # Maximum URI length


# =============================================================================
# RATE LIMITING DEFAULTS
# =============================================================================

# Per-minute rate limits by endpoint type
RATE_LIMIT_DEFAULT = 100                          # requests/minute
RATE_LIMIT_DISCOVERY = 10                         # Heavy compute endpoints
RATE_LIMIT_DETECTION = 30                         # Medium compute endpoints
RATE_LIMIT_COPILOT = 60                           # LLM-based endpoints
RATE_LIMIT_HEALTH = 300                           # Lightweight endpoints
RATE_LIMIT_BURST = 200                            # Burst allowance

# Per-hour limits
RATE_LIMIT_HOURLY_DEFAULT = 1000
RATE_LIMIT_HOURLY_ENTERPRISE = 10000


# =============================================================================
# TIMEOUT DEFAULTS (seconds)
# =============================================================================

# Request timeouts
REQUEST_TIMEOUT_DEFAULT = 30.0
REQUEST_TIMEOUT_DISCOVERY = 120.0                 # Protocol discovery
REQUEST_TIMEOUT_DETECTION = 60.0                  # Field detection
REQUEST_TIMEOUT_LLM = 60.0                        # LLM calls
REQUEST_TIMEOUT_TRANSLATION = 300.0               # API/SDK generation

# Database timeouts
DB_CONNECTION_TIMEOUT = 30.0
DB_QUERY_TIMEOUT = 60.0
DB_POOL_TIMEOUT = 30.0

# Cache timeouts
CACHE_CONNECTION_TIMEOUT = 5.0
CACHE_OPERATION_TIMEOUT = 10.0

# External service timeouts
EXTERNAL_SERVICE_TIMEOUT = 30.0


# =============================================================================
# RETRY CONFIGURATION
# =============================================================================

MAX_RETRIES_DEFAULT = 3
MAX_RETRIES_LLM = 2
MAX_RETRIES_DB = 3
MAX_RETRIES_CACHE = 2

RETRY_BASE_DELAY = 1.0                            # seconds
RETRY_MAX_DELAY = 60.0                            # seconds
RETRY_EXPONENTIAL_BASE = 2.0


# =============================================================================
# CIRCUIT BREAKER CONFIGURATION
# =============================================================================

CIRCUIT_BREAKER_FAILURE_THRESHOLD = 5
CIRCUIT_BREAKER_RECOVERY_TIMEOUT = 60.0           # seconds
CIRCUIT_BREAKER_HALF_OPEN_MAX_CALLS = 3


# =============================================================================
# SECURITY CONSTANTS
# =============================================================================

# Password requirements
PASSWORD_MIN_LENGTH = 12
PASSWORD_MAX_LENGTH = 128
PASSWORD_HASH_ROUNDS = 12

# Token settings
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7
API_KEY_PREFIX_LENGTH = 8

# Session settings
SESSION_TIMEOUT_MINUTES = 60
MAX_SESSIONS_PER_USER = 5

# MFA settings
MFA_CODE_LENGTH = 6
MFA_CODE_VALIDITY_SECONDS = 300


# =============================================================================
# CACHE SETTINGS
# =============================================================================

CACHE_TTL_DEFAULT = 3600                          # 1 hour
CACHE_TTL_DISCOVERY = 7200                        # 2 hours
CACHE_TTL_SESSION = 3600                          # 1 hour
CACHE_TTL_RATE_LIMIT = 60                         # 1 minute

CACHE_MAX_SIZE_LRU = 10000                        # Max items in LRU cache


# =============================================================================
# MODEL CONSTANTS
# =============================================================================

# BiLSTM-CRF field detection
FIELD_DETECTOR_EMBEDDING_DIM = 128
FIELD_DETECTOR_HIDDEN_DIM = 256
FIELD_DETECTOR_NUM_LAYERS = 2

# Anomaly detection
ANOMALY_THRESHOLD_DEFAULT = 0.5
ANOMALY_ENSEMBLE_SIZE = 3


# =============================================================================
# LOGGING CONSTANTS
# =============================================================================

LOG_MAX_MESSAGE_LENGTH = 10000
LOG_TRUNCATION_SUFFIX = "...[truncated]"


# =============================================================================
# API VERSION
# =============================================================================

API_VERSION = "2.0.0"
API_VERSION_PREFIX = "/api/v1"


# =============================================================================
# HEALTH CHECK CONSTANTS
# =============================================================================

HEALTH_CHECK_INTERVAL = 30                        # seconds
HEALTH_CHECK_TIMEOUT = 10                         # seconds
UNHEALTHY_THRESHOLD_ERRORS_PER_MINUTE = 1.0
DEGRADED_THRESHOLD_ERRORS_PER_MINUTE = 0.5

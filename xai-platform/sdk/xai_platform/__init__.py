"""
XAI Platform Python SDK

A client library for interacting with the XAI Platform API.
"""

__version__ = "1.0.0"
__author__ = "XAI Platform Team"

from .client import XAIClient
from .exceptions import (
    XAIClientError,
    XAIAuthError,
    XAIRateLimitError,
    XAINotFoundError,
    XAIValidationError,
    XAIExplanationError,
)

__all__ = [
    "XAIClient",
    "XAIClientError",
    "XAIAuthError",
    "XAIRateLimitError",
    "XAINotFoundError",
    "XAIValidationError",
    "XAIExplanationError",
]

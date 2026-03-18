"""
Custom exceptions for XAI Platform SDK.
"""

class XAIClientError(Exception):
    """Base exception for all XAI Platform errors."""
    pass

class XAIAuthError(XAIClientError):
    """Authentication or authorization error."""
    pass

class XAIRateLimitError(XAIClientError):
    """Rate limit exceeded error."""
    pass

class XAINotFoundError(XAIClientError):
    """Resource not found error."""
    pass

class XAIValidationError(XAIClientError):
    """Validation or input error."""
    pass

class XAIExplanationError(XAIClientError):
    """Explanation generation error."""
    pass

class XAITaskTimeoutError(XAIClientError):
    """Async task timeout error."""
    pass

"""
Audit logging utility for tracking important actions.
"""

from typing import Optional, Dict, Any
from datetime import datetime
from app.db.mongo import get_db
from app.db.repositories.audit_repository import AuditRepository
from fastapi import Request


async def log_action(
    user_id: str,
    action: str,
    resource_type: str,
    resource_id: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    request: Optional[Request] = None
):
    """
    Log an audit event.

    Args:
        user_id: ID of the user performing the action
        action: Action type (e.g., 'model.upload', 'prediction.create')
        resource_type: Type of resource (model, prediction, explanation, etc.)
        resource_id: Optional ID of the affected resource
        details: Optional additional context data
        request: Optional FastAPI Request object to extract IP/user-agent
    """
    try:
        db = get_db()
        ip_address = None
        user_agent = None

        if request:
            ip_address = request.client.host if request.client else None
            user_agent = request.headers.get("user-agent")

        audit_data = {
            "user_id": user_id,
            "action": action,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "details": details or {},
            "ip_address": ip_address,
            "user_agent": user_agent
        }

        await AuditRepository.create(db, audit_data)
    except Exception as e:
        # Don't fail the main operation if audit logging fails
        print(f"Audit logging failed: {str(e)}")


# Predefined action constants for consistency
class AuditActions:
    # Model actions
    MODEL_UPLOAD = "model.upload"
    MODEL_DELETE = "model.delete"
    MODEL_VIEW = "model.view"

    # Prediction actions
    PREDICTION_CREATE = "prediction.create"
    PREDICTION_VIEW = "prediction.view"

    # Explanation actions
    EXPLANATION_CREATE = "explanation.create"
    EXPLANATION_VIEW = "explanation.view"
    EXPLANATION_EXPORT = "explanation.export"

    # Bias actions
    BIAS_ANALYZE = "bias.analyze"
    BIAS_REPORT_GENERATE = "bias.report_generate"

    # API Key actions
    API_KEY_CREATE = "api_key.create"
    API_KEY_DELETE = "api_key.delete"

    # User actions
    USER_LOGIN = "user.login"
    USER_LOGOUT = "user.logout"
    USER_REGISTER = "user.register"

    # Audit actions
    AUDIT_VIEW = "audit.view"

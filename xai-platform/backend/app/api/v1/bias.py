from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks, Request, File, UploadFile
from fastapi.responses import StreamingResponse
from app.models.model_meta import ModelResponse, FeatureSchema
from app.api.v1.auth import get_current_user, decode_token, oauth2_scheme
from app.db.mongo import get_db
from app.workers.celery_app import celery_app
from app.services.model_loader_service import ModelLoaderService
from app.db.repositories.bias_repository import BiasRepository
from app.utils.audit_logger import log_action, AuditActions
from datetime import datetime
from bson import ObjectId
import json
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import io
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

router = APIRouter()

async def get_current_user(token: str = Depends(oauth2_scheme)):
    payload = decode_token(token)
    if not payload:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    db = await get_db()
    user = await db.users.find_one({"email": payload.get("sub")})
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    user["_id"] = str(user["_id"])
    return user

@router.post("/analyze", response_model=Dict[str, Any])
async def analyze_bias(
    request: Request,
    model_id: str,
    protected_attribute: str,
    sensitive_attribute: str,
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    """
    Analyze model bias using fairness metrics.
    Requires evaluation dataset with protected attributes.
    """
    try:
        # Get model metadata
        db = await get_db()
        model_doc = await db.models.find_one({"_id": ObjectId(model_id), "user_id": current_user["_id"]})
        if not model_doc:
            raise HTTPException(status_code=404, detail="Model not found")

        # Load evaluation dataset
        contents = await file.read()
        df = pd.read_csv(pd.io.common.BytesIO(contents))

        # Check for required columns
        if protected_attribute not in df.columns:
            raise HTTPException(status_code=400, detail=f"Dataset missing protected attribute: {protected_attribute}")
        if sensitive_attribute not in df.columns:
            raise HTTPException(status_code=400, detail=f"Dataset missing sensitive attribute: {sensitive_attribute}")

        # Load model
        model_obj, framework = await ModelLoaderService.load_model(model_doc["file_path"])

        # Prepare data for prediction
        features = [col for col in df.columns if col not in [protected_attribute, sensitive_attribute]]
        X = df[features]
        y_true = df[protected_attribute]
        sensitive_values = df[sensitive_attribute]

        # Make predictions
        if framework == "sklearn":
            y_pred = model_obj.predict(X)
        elif framework == "xgboost":
            import xgboost as xgb
            dmatrix = xgb.DMatrix(X)
            y_pred = model_obj.predict(dmatrix)
        elif framework == "onnx":
            input_name = model_obj.get_inputs()[0].name
            y_pred = model_obj.run(None, {input_name: X.values.astype(np.float32)})[0]
        else:
            raise HTTPException(status_code=500, detail="Prediction not implemented for this framework")

        # Compute bias metrics
        bias_metrics = compute_bias_metrics(y_true, y_pred, sensitive_values)

        # Store bias report
        bias_report = {
            "model_id": model_id,
            "user_id": current_user["_id"],
            "protected_attribute": protected_attribute,
            "sensitive_attribute": sensitive_attribute,
            "demographic_parity_diff": bias_metrics["demographic_parity_diff"],
            "equal_opportunity_diff": bias_metrics["equal_opportunity_diff"],
            "disparate_impact_ratio": bias_metrics["disparate_impact_ratio"],
            "group_metrics": bias_metrics["group_metrics"],
            "dataset_size": len(df),
            "created_at": datetime.utcnow()
        }

        bias_id = await BiasRepository.create(bias_report)

        # Log audit event
        await log_action(
            user_id=current_user["_id"],
            action=AuditActions.BIAS_ANALYZE,
            resource_type="bias_report",
            resource_id=str(bias_id),
            details={
                "model_id": model_id,
                "protected_attribute": protected_attribute,
                "sensitive_attribute": sensitive_attribute,
                "metrics": bias_metrics
            },
            request=request
        )

        return {
            "message": "Bias analysis complete",
            "bias_id": bias_id,
            "metrics": bias_metrics,
            "dataset_size": len(df)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/reports/{model_id}", response_model=List[Dict[str, Any]])
async def get_bias_reports(
    model_id: str,
    limit: int = 50,
    skip: int = 0,
    current_user: dict = Depends(get_current_user)
):
    """Get bias reports for a model."""
    try:
        # Verify model belongs to user
        db = await get_db()
        model_doc = await db.models.find_one({"_id": ObjectId(model_id), "user_id": current_user["_id"]})
        if not model_doc:
            raise HTTPException(status_code=404, detail="Model not found")

        # Get bias reports
        reports = await BiasRepository.get_by_model(model_id, limit, skip)
        return reports

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/compare", response_model=List[Dict[str, Any]])
async def compare_bias(
    model_ids: List[str],
    protected_attribute: str,
    sensitive_attribute: str,
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    """
    Compare bias across multiple models.
    """
    try:
        # Load evaluation dataset
        contents = await file.read()
        df = pd.read_csv(pd.io.common.BytesIO(contents))

        # Check for required columns
        if protected_attribute not in df.columns:
            raise HTTPException(status_code=400, detail=f"Dataset missing protected attribute: {protected_attribute}")
        if sensitive_attribute not in df.columns:
            raise HTTPException(status_code=400, detail=f"Dataset missing sensitive attribute: {sensitive_attribute}")

        results = []

        for model_id in model_ids:
            # Get model metadata
            db = await get_db()
            model_doc = await db.models.find_one({"_id": ObjectId(model_id), "user_id": current_user["_id"]})
            if not model_doc:
                continue  # Skip models not owned by user

            # Load model
            model_obj, framework = await ModelLoaderService.load_model(model_doc["file_path"])

            # Prepare data for prediction
            features = [col for col in df.columns if col not in [protected_attribute, sensitive_attribute]]
            X = df[features]
            y_true = df[protected_attribute]
            sensitive_values = df[sensitive_attribute]

            # Make predictions
            if framework == "sklearn":
                y_pred = model_obj.predict(X)
            elif framework == "xgboost":
                import xgboost as xgb
                dmatrix = xgb.DMatrix(X)
                y_pred = model_obj.predict(dmatrix)
            elif framework == "onnx":
                input_name = model_obj.get_inputs()[0].name
                y_pred = model_obj.run(None, {input_name: X.values.astype(np.float32)})[0]
            else:
                continue

            # Compute bias metrics
            bias_metrics = compute_bias_metrics(y_true, y_pred, sensitive_values)

            results.append({
                "model_id": model_id,
                "model_name": model_doc.get("name", "Unknown"),
                "task_type": model_doc.get("task_type", "unknown"),
                "metrics": bias_metrics,
                "dataset_size": len(df)
            })

        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics/{model_id}", response_model=Dict[str, Any])
async def get_bias_metrics(
    model_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get aggregated bias metrics for a model."""
    try:
        # Verify model belongs to user
        db = await get_db()
        model_doc = await db.models.find_one({"_id": ObjectId(model_id), "user_id": current_user["_id"]})
        if not model_doc:
            raise HTTPException(status_code=404, detail="Model not found")

        # Get all bias reports for this model
        reports = await BiasRepository.get_by_model(model_id)

        if not reports:
            raise HTTPException(status_code=404, detail="No bias reports found")

        # Aggregate metrics
        total_reports = len(reports)
        avg_metrics = {
            "demographic_parity_diff": 0,
            "equal_opportunity_diff": 0,
            "disparate_impact_ratio": 0
        }

        for report in reports:
            avg_metrics["demographic_parity_diff"] += report["demographic_parity_diff"]
            avg_metrics["equal_opportunity_diff"] += report["equal_opportunity_diff"]
            avg_metrics["disparate_impact_ratio"] += report["disparate_impact_ratio"]

        for key in avg_metrics:
            avg_metrics[key] /= total_reports

        return {
            "model_id": model_id,
            "total_reports": total_reports,
            "average_metrics": avg_metrics,
            "reports": reports
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/generate-report/{report_id}")
async def generate_bias_report_pdf(
    report_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Generate a PDF compliance report for a bias analysis.
    Includes fairness metrics and compliance assessment with GDPR, AI Act, and ECOA.
    """
    try:
        db = await get_db()

        # Find the bias report
        report = await db.bias_reports.find_one({"_id": ObjectId(report_id)})
        if not report:
            raise HTTPException(status_code=404, detail="Bias report not found")

        # Verify user ownership
        model_doc = await db.models.find_one({"_id": ObjectId(report["model_id"])})
        if not model_doc or model_doc["user_id"] != current_user["_id"]:
            raise HTTPException(status_code=403, detail="Access denied")

        # Generate PDF
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        # Title
        title_style = ParagraphStyle(
            'Title',
            parent=styles['Heading1'],
            fontSize=18,
            spaceAfter=20,
            alignment=1  # Center
        )
        story.append(Paragraph("Bias Analysis Compliance Report", title_style))
        story.append(Spacer(1, 12))

        # Model Information
        story.append(Paragraph("<b>Model Information</b>", styles['Heading2']))
        story.append(Paragraph(f"<b>Model Name:</b> {model_doc.get('name', 'N/A')}", styles['Normal']))
        story.append(Paragraph(f"<b>Model ID:</b> {report['model_id']}", styles['Normal']))
        story.append(Paragraph(f"<b>Report ID:</b> {report_id}", styles['Normal']))
        story.append(Paragraph(f"<b>Generated:</b> {report['created_at'].strftime('%Y-%m-%d %H:%M:%S') if isinstance(report['created_at'], datetime) else report['created_at']}", styles['Normal']))
        story.append(Spacer(1, 20))

        # Fairness Metrics
        story.append(Paragraph("<b>Fairness Metrics</b>", styles['Heading2']))
        story.append(Spacer(1, 12))

        metrics_data = [
            ["Metric", "Value", "Status"],
            ["Demographic Parity Difference", f"{report['demographic_parity_diff']:.4f}", assess_demographic_parity(report['demographic_parity_diff'])],
            ["Equal Opportunity Difference", f"{report['equal_opportunity_diff']:.4f}", assess_equal_opportunity(report['equal_opportunity_diff'])],
            ["Disparate Impact Ratio", f"{report['disparate_impact_ratio']:.4f}", assess_disparate_impact(report['disparate_impact_ratio'])],
            ["Protected Attribute", report['protected_attribute'], "-"],
            ["Sensitive Attribute", report['sensitive_attribute'], "-"],
            ["Dataset Size", str(report['dataset_size']), "-"]
        ]

        metrics_table = Table(metrics_data, colWidths=[180, 120, 120])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(metrics_table)
        story.append(Spacer(1, 20))

        # Group Metrics
        if report.get('group_metrics'):
            story.append(Paragraph("<b>Group-Specific Metrics</b>", styles['Heading2']))
            story.append(Spacer(1, 12))

            group_table_data = [["Group", "Positive Rate", "TPR", "FPR", "Accuracy"]]
            for group, metrics in report['group_metrics'].items():
                group_table_data.append([
                    str(group),
                    f"{metrics.get('positive_rate', 0):.3f}",
                    f"{metrics.get('true_positive_rate', 0):.3f}",
                    f"{metrics.get('false_positive_rate', 0):.3f}",
                    f"{metrics.get('accuracy', 0):.3f}"
                ])

            group_table = Table(group_table_data, colWidths=[80, 80, 80, 80, 80])
            group_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(group_table)
            story.append(Spacer(1, 20))

        # Compliance Assessment
        story.append(Paragraph("<b>Compliance Assessment</b>", styles['Heading2']))
        story.append(Spacer(1, 12))

        dp_assessment = assess_demographic_parity(report['demographic_parity_diff'])
        eo_assessment = assess_equal_opportunity(report['equal_opportunity_diff'])
        di_assessment = assess_disparate_impact(report['disparate_impact_ratio'])

        assessments = [
            ["Aspect", "Assessment", "Details"],
            ["Demographic Parity", dp_assessment, generate_paragraph(dp_assessment, report['demographic_parity_diff'])],
            ["Equal Opportunity", eo_assessment, generate_paragraph(eo_assessment, report['equal_opportunity_diff'])],
            ["Disparate Impact", di_assessment, generate_paragraph(di_assessment, report['disparate_impact_ratio'])]
        ]

        assess_table = Table(assessments, colWidths=[150, 100, 250])
        assess_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'TOP')
        ]))
        story.append(assess_table)
        story.append(Spacer(1, 20))

        # Recommendations
        story.append(Paragraph("<b>Recommendations</b>", styles['Heading2']))
        story.append(Spacer(1, 12))

        recommendations = generate_recommendations(dp_assessment, eo_assessment, di_assessment)
        for rec in recommendations:
            story.append(Paragraph(f"• {rec}", styles['Normal']))
            story.append(Spacer(1, 6))

        story.append(Spacer(1, 20))

        # Footer with disclaimer
        story.append(Paragraph("<i>Disclaimer: This report is generated automatically and should be reviewed by qualified professionals. Results are based on the provided evaluation dataset and may not generalize to all scenarios.</i>", styles['Italic']))

        doc.build(story)
        buffer.seek(0)

        return StreamingResponse(
            buffer,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=bias_report_{report_id}.pdf"}
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def assess_demographic_parity(diff: float) -> str:
    """Assess demographic parity fairness."""
    if diff < 0.05:
        return "PASS"
    elif diff < 0.1:
        return "MARGINAL"
    else:
        return "FAIL"

def assess_equal_opportunity(diff: float) -> str:
    """Assess equal opportunity fairness."""
    if diff < 0.05:
        return "PASS"
    elif diff < 0.1:
        return "MARGINAL"
    else:
        return "FAIL"

def assess_disparate_impact(ratio: float) -> str:
    """Assess disparate impact (80% rule)."""
    if ratio >= 0.8:
        return "PASS"
    elif ratio >= 0.7:
        return "MARGINAL"
    else:
        return "FAIL"

def generate_paragraph(status: str, value: float) -> str:
    """Generate detail paragraph for assessment."""
    if status == "PASS":
        return "Metric within acceptable thresholds. No significant fairness violations detected."
    elif status == "MARGINAL":
        return "Metric close to acceptable threshold. Monitor and consider mitigation strategies."
    else:
        return "Metric exceeds fairness threshold. Mitigation recommended before deployment."

def generate_recommendations(dp: str, eo: str, di: str) -> List[str]:
    """Generate recommendations based on assessment results."""
    recommendations = []

    if dp == "FAIL" or eo == "FAIL" or di == "FAIL":
        recommendations.append("Apply fairness mitigation techniques (reweighing, adversarial debiasing, or disparate impact remover).")
        recommendations.append("Collect more diverse training data to address representation gaps.")
        recommendations.append("Consider using fairness constraints during model training.")

    if dp == "MARGINAL" or eo == "MARGINAL" or di == "MARGINAL":
        recommendations.append("Monitor model performance in production for fairness drift.")

    if dp == "PASS" and eo == "PASS" and di == "PASS":
        recommendations.append("Model appears fair based on current evaluation.")
        recommendations.append("Regular audits recommended to maintain fairness standards.")

    recommendations.append("Document bias analysis and mitigation steps for compliance (GDPR Article 22, AI Act, ECOA).")
    recommendations.append("Establish ongoing monitoring for model drift and fairness degradation.")

    return recommendations


def compute_bias_metrics(y_true, y_pred, sensitive_attribute):
    """
    Compute fairness metrics for bias analysis.
    Returns dictionary with demographic parity, equal opportunity, and disparate impact.
    """
    import numpy as np
    from collections import defaultdict

    try:
        groups = np.unique(sensitive_attribute)
        group_metrics = {}

        for group in groups:
            mask = sensitive_attribute == group
            group_metrics[str(group)] = {
                "positive_rate": float(np.mean(y_pred[mask])),
                "true_positive_rate": float(np.mean(y_pred[mask & (y_true == 1)])),
                "false_positive_rate": float(np.mean(y_pred[mask & (y_true == 0)])),
                "accuracy": float(np.mean(y_pred[mask] == y_true[mask]))
            }

        rates = [v["positive_rate"] for v in group_metrics.values()]
        tprs = [v["true_positive_rate"] for v in group_metrics.values()]

        min_rate = min(rates) if min(rates) > 0 else 1e-9
        min_tpr = min(tprs) if min(tprs) > 0 else 1e-9

        return {
            "demographic_parity_diff": max(rates) - min(rates),
            "equal_opportunity_diff": max(tprs) - min(tprs),
            "disparate_impact_ratio": min_rate / max(rates),
            "group_metrics": group_metrics
        }

    except Exception as e:
        return {
            "error": str(e),
            "demographic_parity_diff": None,
            "equal_opportunity_diff": None,
            "disparate_impact_ratio": None,
            "group_metrics": {}
        }
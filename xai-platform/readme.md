# Explainable AI (XAI) Platform — Complete Implementation Plan

> **Version:** 1.0 | **Last Updated:** March 2026  
> **Stack:** FastAPI · React · MongoDB · SHAP · LIME · scikit-learn · XGBoost

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Technology Stack](#3-technology-stack)
4. [Project Structure](#4-project-structure)
5. [Phase-wise Implementation Plan](#5-phase-wise-implementation-plan)
6. [Backend Implementation](#6-backend-implementation)
7. [Frontend Implementation](#7-frontend-implementation)
8. [ML & Explainability Engine](#8-ml--explainability-engine)
9. [Database Design](#9-database-design)
10. [API Reference](#10-api-reference)
11. [Core Features Deep Dive](#11-core-features-deep-dive)
12. [Extra / Advanced Features](#12-extra--advanced-features)
13. [Security & Compliance](#13-security--compliance)
14. [Deployment & DevOps](#14-deployment--devops)
15. [Testing Strategy](#15-testing-strategy)
16. [Milestones & Timeline](#16-milestones--timeline)
17. [Future Roadmap](#17-future-roadmap)

---

## 1. Project Overview

### Mission Statement
Build a production-grade, domain-agnostic Explainable AI (XAI) platform that transforms opaque machine learning models into transparent, auditable, and trustworthy decision systems — suitable for high-stakes domains like finance, healthcare, and recruitment.

### Core Problems Solved

| Problem | Solution |
|---|---|
| Black-box model decisions | SHAP + LIME local/global explanations |
| Lack of model auditability | Full prediction + explanation logging |
| No bias visibility | Fairness metrics across demographic groups |
| Model comparison difficulty | Side-by-side model analysis dashboard |
| Technical-only audiences | NLP-based plain-language explanation generator |

### Supported Model Types
- **Tree-based:** Random Forest, XGBoost, LightGBM, Decision Trees
- **Linear:** Logistic Regression, Linear Regression, Lasso, Ridge
- **Neural Networks:** PyTorch, TensorFlow/Keras (via ONNX or native)
- **Custom APIs:** Any model exposed via REST endpoint

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLIENT LAYER                             │
│   React/Next.js Dashboard  │  Mobile App  │  External API       │
└───────────────┬─────────────────────────────────────────────────┘
                │ HTTPS / WebSocket
┌───────────────▼─────────────────────────────────────────────────┐
│                      API GATEWAY (Nginx)                        │
│            Rate Limiting  │  Auth  │  Load Balancing            │
└───────────────┬─────────────────────────────────────────────────┘
                │
┌───────────────▼─────────────────────────────────────────────────┐
│                    FASTAPI BACKEND                               │
│  ┌────────────┐  ┌──────────────┐  ┌────────────────────────┐  │
│  │ Auth Svc   │  │ Model Svc    │  │  Prediction Svc        │  │
│  └────────────┘  └──────────────┘  └────────────────────────┘  │
│  ┌────────────┐  ┌──────────────┐  ┌────────────────────────┐  │
│  │ Explain Svc│  │ Bias Svc     │  │  NLG Svc               │  │
│  └────────────┘  └──────────────┘  └────────────────────────┘  │
└──────────┬────────────────────┬────────────────────────────────-┘
           │                    │
┌──────────▼──────┐   ┌─────────▼──────────────────────────────┐
│   CELERY QUEUE  │   │       ML EXPLAINABILITY ENGINE          │
│  (Async Tasks)  │   │  SHAP  │  LIME  │  Custom Interpreters  │
└──────────┬──────┘   └────────────────────────────────────────┘
           │
┌──────────▼──────────────────────────────────────────────────────┐
│                      DATA LAYER                                 │
│  MongoDB (metadata/logs)  │  Redis (cache)  │  S3/MinIO (files) │
└─────────────────────────────────────────────────────────────────┘
```

### Key Design Principles
- **Microservice-inspired monolith** for simplicity with clear service boundaries
- **Async-first** — all long-running tasks (SHAP computation) offloaded to Celery workers
- **Event-driven logging** — every prediction and explanation is stored automatically
- **API-first** — backend fully decoupled from frontend

---

## 3. Technology Stack

### Backend
| Component | Technology | Reason |
|---|---|---|
| API Framework | FastAPI | Async support, auto OpenAPI docs, fast |
| ML Libraries | scikit-learn, XGBoost, LightGBM | Industry standard model support |
| Explainability | SHAP, LIME, Captum (PyTorch) | Best-in-class XAI toolkits |
| NLP Explanations | OpenAI API / LLaMA (local) | Natural language generation |
| Task Queue | Celery + Redis | Async SHAP computations |
| File Storage | MinIO (S3-compatible) | Model file management |
| Auth | JWT + OAuth2 (via FastAPI-Users) | Secure, standard auth |

### Frontend
| Component | Technology | Reason |
|---|---|---|
| Framework | Next.js 14 (App Router) | SSR, routing, performance |
| UI Library | Tailwind CSS + shadcn/ui | Rapid, consistent UI |
| Charts | Recharts + D3.js | SHAP waterfall, force plots |
| State Management | Zustand | Lightweight global state |
| API Client | TanStack Query | Caching, sync, async state |
| Forms | React Hook Form + Zod | Validation, file uploads |

### Infrastructure
| Component | Technology |
|---|---|
| Database | MongoDB Atlas / self-hosted |
| Cache | Redis |
| Container | Docker + Docker Compose |
| Orchestration | Kubernetes (production) |
| CI/CD | GitHub Actions |
| Monitoring | Prometheus + Grafana |

---

## 4. Project Structure

```
xai-platform/
├── backend/
│   ├── app/
│   │   ├── main.py                    # FastAPI app entry point
│   │   ├── config.py                  # Settings (Pydantic BaseSettings)
│   │   ├── dependencies.py            # Shared DI (DB, auth)
│   │   ├── api/
│   │   │   ├── v1/
│   │   │   │   ├── auth.py
│   │   │   │   ├── models.py          # Model upload/management
│   │   │   │   ├── predictions.py     # Predict endpoint
│   │   │   │   ├── explanations.py    # SHAP/LIME endpoints
│   │   │   │   ├── bias.py            # Fairness analysis
│   │   │   │   └── compare.py        # Model comparison
│   │   ├── services/
│   │   │   ├── model_service.py       # Load, validate, store models
│   │   │   ├── prediction_service.py  # Run inference
│   │   │   ├── shap_service.py        # SHAP explanations
│   │   │   ├── lime_service.py        # LIME explanations
│   │   │   ├── bias_service.py        # Fairness metrics
│   │   │   └── nlg_service.py         # NL explanation generation
│   │   ├── models/                    # Pydantic schemas
│   │   │   ├── model_meta.py
│   │   │   ├── prediction.py
│   │   │   └── explanation.py
│   │   ├── db/
│   │   │   ├── mongo.py               # Motor async client
│   │   │   └── repositories/
│   │   ├── workers/
│   │   │   └── celery_app.py          # Async task definitions
│   │   └── utils/
│   │       ├── file_handler.py
│   │       └── model_loader.py
│   ├── tests/
│   ├── Dockerfile
│   └── requirements.txt
│
├── frontend/
│   ├── app/
│   │   ├── layout.tsx
│   │   ├── page.tsx                   # Landing / Dashboard
│   │   ├── models/
│   │   │   ├── page.tsx               # Model list
│   │   │   ├── upload/page.tsx        # Upload wizard
│   │   │   └── [id]/page.tsx         # Model detail
│   │   ├── predict/
│   │   │   └── [modelId]/page.tsx    # Prediction form
│   │   ├── explain/
│   │   │   ├── local/page.tsx        # Individual prediction explain
│   │   │   └── global/page.tsx       # Global feature importance
│   │   ├── compare/page.tsx          # Model comparison view
│   │   └── bias/page.tsx             # Bias & fairness dashboard
│   ├── components/
│   │   ├── charts/
│   │   │   ├── SHAPWaterfall.tsx
│   │   │   ├── SHAPBeeswarm.tsx
│   │   │   ├── FeatureImportanceBar.tsx
│   │   │   ├── LIMEPlot.tsx
│   │   │   └── BiasRadarChart.tsx
│   │   ├── forms/
│   │   │   ├── PredictionForm.tsx
│   │   │   └── ModelUploadWizard.tsx
│   │   └── ui/                       # shadcn components
│   ├── lib/
│   │   ├── api.ts                    # API client
│   │   └── store.ts                  # Zustand store
│   └── Dockerfile
│
├── docker-compose.yml
├── docker-compose.prod.yml
└── README.md
```

---

## 5. Phase-wise Implementation Plan

### Phase 1 — Foundation (Weeks 1–3)
**Goal:** Working backend + frontend scaffold with auth and model upload

- [ ] Project scaffolding (monorepo setup, Docker Compose)
- [ ] FastAPI app with health check, CORS, logging middleware
- [ ] MongoDB connection with Motor async driver
- [ ] JWT Authentication (register, login, refresh token)
- [ ] Model upload endpoint (`.pkl`, `.joblib`, `.onnx`, `.h5`)
- [ ] MinIO/S3 integration for model file storage
- [ ] Model metadata schema and CRUD endpoints
- [ ] Next.js frontend with auth pages (login/register)
- [ ] Dashboard shell with sidebar navigation
- [ ] Model upload wizard (drag & drop, validation)

**Deliverable:** Users can register, log in, and upload models.

---

### Phase 2 — Prediction Engine (Weeks 4–5)
**Goal:** Real-time inference with input validation

- [ ] Model loader service (auto-detect framework: sklearn, xgboost, keras, ONNX)
- [ ] Prediction endpoint with structured input schema
- [ ] Dynamic input form generation from model metadata
- [ ] Prediction result storage in MongoDB
- [ ] Prediction history page in frontend
- [ ] WebSocket support for real-time prediction feedback
- [ ] Batch prediction support (CSV upload → predictions)
- [ ] Prediction confidence scores / probability outputs

**Deliverable:** Users can submit inputs and receive predictions with confidence scores.

---

### Phase 3 — Explainability Engine (Weeks 6–9)
**Goal:** SHAP and LIME integration with full visualization suite

- [ ] SHAP integration for tree-based models (TreeExplainer)
- [ ] SHAP integration for linear models (LinearExplainer)
- [ ] SHAP integration for deep models (DeepExplainer / KernelExplainer)
- [ ] Celery async tasks for SHAP computation (background jobs)
- [ ] LIME integration for tabular data (LimeTabularExplainer)
- [ ] LIME integration for text (LimeTextExplainer)
- [ ] LIME integration for images (LimeImageExplainer)
- [ ] Global feature importance endpoint + bar chart visualization
- [ ] Local explanation endpoint + SHAP waterfall chart
- [ ] SHAP beeswarm plot (global summary)
- [ ] SHAP dependence plots
- [ ] LIME explanation visualization
- [ ] Explanation caching in Redis (avoid redundant computation)

**Deliverable:** Full SHAP + LIME explanation suite with interactive charts.

---

### Phase 4 — Advanced Analytics (Weeks 10–12)
**Goal:** Bias detection, model comparison, NL explanations

- [ ] Bias detection service (disparate impact, demographic parity, equal opportunity)
- [ ] Protected attribute configuration per model
- [ ] Bias dashboard with radar chart and group comparison
- [ ] Model comparison endpoint (side-by-side SHAP importance)
- [ ] NLG service — LLM-powered plain-language explanation of SHAP output
- [ ] Explanation export (PDF report, JSON, CSV)
- [ ] Audit log viewer in frontend
- [ ] Notification system (Celery task completion alerts)

**Deliverable:** Bias analysis, model comparison, and human-readable explanations.

---

### Phase 5 — API Layer & Polish (Weeks 13–15)
**Goal:** External API, docs, production-readiness

- [ ] API key management for external developers
- [ ] Rate limiting per API key (Redis-based)
- [ ] Auto-generated OpenAPI/Swagger documentation
- [ ] SDK scaffold (Python client library)
- [ ] Frontend performance optimization (lazy loading, virtualization)
- [ ] Accessibility audit (WCAG 2.1 AA)
- [ ] End-to-end tests (Playwright)
- [ ] Load testing (Locust)
- [ ] Kubernetes Helm chart
- [ ] Production Docker Compose with Nginx + SSL

**Deliverable:** Production-ready platform with external API access.

---

## 6. Backend Implementation

### 6.1 FastAPI Application Setup

```python
# backend/app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.db.mongo import connect_db, close_db
from app.api.v1 import auth, models, predictions, explanations, bias

@asynccontextmanager
async def lifespan(app: FastAPI):
    await connect_db()
    yield
    await close_db()

app = FastAPI(
    title="XAI Platform API",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(CORSMiddleware, allow_origins=["*"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

app.include_router(auth.router, prefix="/api/v1/auth", tags=["Auth"])
app.include_router(models.router, prefix="/api/v1/models", tags=["Models"])
app.include_router(predictions.router, prefix="/api/v1/predict", tags=["Predictions"])
app.include_router(explanations.router, prefix="/api/v1/explain", tags=["Explanations"])
app.include_router(bias.router, prefix="/api/v1/bias", tags=["Bias"])
```

### 6.2 Model Upload Service

```python
# backend/app/services/model_service.py
import joblib, pickle, onnxruntime
from pathlib import Path
from app.utils.file_handler import upload_to_storage

SUPPORTED_FORMATS = {
    ".pkl": "sklearn",
    ".joblib": "sklearn",
    ".json": "xgboost",
    ".onnx": "onnx",
    ".h5": "keras",
    ".pt": "pytorch"
}

async def load_model(file_path: str, framework: str):
    if framework == "sklearn":
        return joblib.load(file_path)
    elif framework == "xgboost":
        import xgboost as xgb
        model = xgb.Booster()
        model.load_model(file_path)
        return model
    elif framework == "onnx":
        return onnxruntime.InferenceSession(file_path)
    elif framework == "keras":
        from tensorflow import keras
        return keras.models.load_model(file_path)
```

### 6.3 SHAP Service

```python
# backend/app/services/shap_service.py
import shap
import numpy as np
from typing import Any

def get_explainer(model, framework: str, background_data=None):
    if framework in ["sklearn", "xgboost", "lightgbm"]:
        return shap.TreeExplainer(model)
    elif framework == "linear":
        return shap.LinearExplainer(model, background_data)
    else:
        return shap.KernelExplainer(model.predict, background_data)

def compute_shap_values(model, input_data: np.ndarray, framework: str, background_data=None):
    explainer = get_explainer(model, framework, background_data)
    shap_values = explainer.shap_values(input_data)
    expected_value = explainer.expected_value
    return {
        "shap_values": shap_values.tolist() if isinstance(shap_values, np.ndarray)
                       else [sv.tolist() for sv in shap_values],
        "expected_value": expected_value if isinstance(expected_value, float)
                          else expected_value.tolist(),
        "feature_names": list(input_data.columns) if hasattr(input_data, 'columns') else []
    }

def compute_global_importance(shap_values: np.ndarray) -> dict:
    mean_abs = np.abs(shap_values).mean(axis=0)
    return {
        "feature_importance": mean_abs.tolist(),
        "ranking": np.argsort(mean_abs)[::-1].tolist()
    }
```

### 6.4 LIME Service

```python
# backend/app/services/lime_service.py
import lime
import lime.lime_tabular
import numpy as np

def compute_lime_explanation(model, input_instance: np.ndarray,
                              training_data: np.ndarray, feature_names: list,
                              mode: str = "classification", num_features: int = 10):
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=training_data,
        feature_names=feature_names,
        mode=mode,
        discretize_continuous=True
    )
    predict_fn = model.predict_proba if mode == "classification" else model.predict
    explanation = explainer.explain_instance(
        data_row=input_instance,
        predict_fn=predict_fn,
        num_features=num_features
    )
    return {
        "local_explanation": explanation.as_list(),
        "score": explanation.score,
        "intercept": explanation.intercept[1] if mode == "classification" else explanation.intercept,
        "predicted_value": explanation.predicted_value if hasattr(explanation, 'predicted_value') else None
    }
```

### 6.5 Bias Detection Service

```python
# backend/app/services/bias_service.py
import numpy as np
from dataclasses import dataclass

@dataclass
class BiasReport:
    demographic_parity_diff: float
    equal_opportunity_diff: float
    disparate_impact_ratio: float
    group_metrics: dict

def compute_bias_metrics(y_true, y_pred, sensitive_attribute, group_labels) -> BiasReport:
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

    return BiasReport(
        demographic_parity_diff=max(rates) - min(rates),
        equal_opportunity_diff=max(tprs) - min(tprs),
        disparate_impact_ratio=min(rates) / max(rates),
        group_metrics=group_metrics
    )
```

---

## 7. Frontend Implementation

### 7.1 SHAP Waterfall Chart Component

```tsx
// frontend/components/charts/SHAPWaterfall.tsx
import { BarChart, Bar, XAxis, YAxis, Tooltip, Cell, ReferenceLine, ResponsiveContainer } from "recharts";

interface SHAPWaterfallProps {
  shapValues: number[];
  featureNames: string[];
  baseValue: number;
  prediction: number;
}

export function SHAPWaterfall({ shapValues, featureNames, baseValue, prediction }: SHAPWaterfallProps) {
  const sorted = shapValues
    .map((val, i) => ({ feature: featureNames[i], value: val }))
    .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
    .slice(0, 15);

  return (
    <div className="w-full bg-white rounded-xl shadow p-6">
      <h3 className="text-lg font-semibold mb-1">Local Explanation (SHAP)</h3>
      <p className="text-sm text-gray-500 mb-4">
        Base value: <strong>{baseValue.toFixed(4)}</strong> → Prediction: <strong>{prediction.toFixed(4)}</strong>
      </p>
      <ResponsiveContainer width="100%" height={400}>
        <BarChart data={sorted} layout="vertical" margin={{ left: 120 }}>
          <XAxis type="number" domain={["auto", "auto"]} />
          <YAxis type="category" dataKey="feature" width={110} tick={{ fontSize: 12 }} />
          <Tooltip formatter={(v: number) => v.toFixed(4)} />
          <ReferenceLine x={0} stroke="#6b7280" />
          <Bar dataKey="value" radius={[0, 4, 4, 0]}>
            {sorted.map((entry, index) => (
              <Cell key={index} fill={entry.value >= 0 ? "#ef4444" : "#3b82f6"} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <div className="flex gap-4 mt-3 text-sm">
        <span className="flex items-center gap-1"><span className="w-3 h-3 rounded bg-red-500 inline-block"/>Increases prediction</span>
        <span className="flex items-center gap-1"><span className="w-3 h-3 rounded bg-blue-500 inline-block"/>Decreases prediction</span>
      </div>
    </div>
  );
}
```

### 7.2 Global Feature Importance

```tsx
// frontend/components/charts/FeatureImportanceBar.tsx
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from "recharts";

interface Props {
  features: { name: string; importance: number }[];
}

export function FeatureImportanceBar({ features }: Props) {
  const sorted = [...features].sort((a, b) => b.importance - a.importance).slice(0, 20);
  return (
    <div className="w-full bg-white rounded-xl shadow p-6">
      <h3 className="text-lg font-semibold mb-4">Global Feature Importance</h3>
      <ResponsiveContainer width="100%" height={500}>
        <BarChart data={sorted} layout="vertical" margin={{ left: 130 }}>
          <CartesianGrid strokeDasharray="3 3" horizontal={false} />
          <XAxis type="number" tickFormatter={(v) => v.toFixed(3)} />
          <YAxis type="category" dataKey="name" width={125} tick={{ fontSize: 12 }} />
          <Tooltip formatter={(v: number) => v.toFixed(4)} />
          <Bar dataKey="importance" fill="#6366f1" radius={[0, 4, 4, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
```

### 7.3 Prediction Form with Dynamic Fields

```tsx
// frontend/components/forms/PredictionForm.tsx
"use client";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";

interface FeatureSchema { name: string; type: "numeric" | "categorical"; options?: string[] }

export function PredictionForm({ features, onSubmit }: { features: FeatureSchema[]; onSubmit: (data: any) => void }) {
  const { register, handleSubmit } = useForm();

  return (
    <form onSubmit={handleSubmit(onSubmit)} className="grid grid-cols-2 gap-4">
      {features.map((f) => (
        <div key={f.name} className="flex flex-col gap-1">
          <label className="text-sm font-medium text-gray-700">{f.name}</label>
          {f.type === "categorical" ? (
            <select {...register(f.name)} className="border rounded-lg px-3 py-2 text-sm focus:ring-2 ring-indigo-400">
              {f.options?.map((opt) => <option key={opt} value={opt}>{opt}</option>)}
            </select>
          ) : (
            <input type="number" step="any" {...register(f.name, { valueAsNumber: true })}
              className="border rounded-lg px-3 py-2 text-sm focus:ring-2 ring-indigo-400" />
          )}
        </div>
      ))}
      <div className="col-span-2">
        <button type="submit" className="w-full bg-indigo-600 text-white py-2.5 rounded-lg font-semibold hover:bg-indigo-700 transition">
          Run Prediction + Explain
        </button>
      </div>
    </form>
  );
}
```

---

## 8. ML & Explainability Engine

### 8.1 Explanation Method Selection Guide

| Model Type | Recommended Method | Fallback |
|---|---|---|
| Random Forest | SHAP TreeExplainer | SHAP KernelExplainer |
| XGBoost / LightGBM | SHAP TreeExplainer | LIME Tabular |
| Logistic Regression | SHAP LinearExplainer | LIME Tabular |
| Neural Network (Keras) | SHAP DeepExplainer | SHAP KernelExplainer |
| Neural Network (PyTorch) | Captum IntegratedGradients | SHAP DeepExplainer |
| Unknown / External API | LIME Tabular | SHAP KernelExplainer |

### 8.2 Explanation Types

**Local Explanations (per prediction):**
- SHAP waterfall plot — shows feature contribution to a single prediction
- SHAP force plot — interactive visualization of feature push/pull
- LIME explanation — locally linear approximation around the input

**Global Explanations (model-wide):**
- SHAP beeswarm summary — distribution of SHAP values across dataset
- SHAP mean absolute importance — ranked feature bar chart
- SHAP dependence plots — feature interaction with another feature
- Partial Dependence Plots (PDP) — marginal effect of a feature

### 8.3 Celery Async Task Flow

```
User requests explanation
         │
         ▼
POST /explain → returns task_id immediately
         │
         ▼
Celery worker picks up task
         │
         ▼
Compute SHAP values (may take 10–60s for large datasets)
         │
         ▼
Store result in MongoDB + Redis cache
         │
         ▼
WebSocket pushes completion event to frontend
         │
         ▼
Frontend fetches result via GET /explain/{task_id}
```

---

## 9. Database Design

### MongoDB Collections

#### `users`
```json
{
  "_id": "ObjectId",
  "email": "string (unique)",
  "hashed_password": "string",
  "name": "string",
  "role": "admin | user | viewer",
  "api_keys": ["string"],
  "created_at": "datetime",
  "plan": "free | pro | enterprise"
}
```

#### `models`
```json
{
  "_id": "ObjectId",
  "user_id": "ObjectId",
  "name": "string",
  "description": "string",
  "framework": "sklearn | xgboost | keras | onnx | api",
  "task_type": "classification | regression",
  "feature_schema": [{ "name": "string", "type": "numeric | categorical", "options": [] }],
  "file_path": "string (S3/MinIO key)",
  "background_data_path": "string",
  "protected_attributes": ["string"],
  "tags": ["string"],
  "version": "string",
  "metrics": { "accuracy": 0.95, "f1": 0.94, "auc": 0.97 },
  "created_at": "datetime",
  "updated_at": "datetime"
}
```

#### `predictions`
```json
{
  "_id": "ObjectId",
  "model_id": "ObjectId",
  "user_id": "ObjectId",
  "input_data": {},
  "prediction": "number | string",
  "probability": [0.12, 0.88],
  "explanation_id": "ObjectId",
  "latency_ms": 45,
  "created_at": "datetime"
}
```

#### `explanations`
```json
{
  "_id": "ObjectId",
  "prediction_id": "ObjectId",
  "model_id": "ObjectId",
  "method": "shap | lime",
  "explanation_type": "local | global",
  "shap_values": [],
  "expected_value": 0.5,
  "feature_names": [],
  "lime_weights": [],
  "nl_explanation": "string",
  "task_status": "pending | complete | failed",
  "created_at": "datetime"
}
```

#### `bias_reports`
```json
{
  "_id": "ObjectId",
  "model_id": "ObjectId",
  "user_id": "ObjectId",
  "protected_attribute": "string",
  "demographic_parity_diff": 0.12,
  "equal_opportunity_diff": 0.08,
  "disparate_impact_ratio": 0.87,
  "group_metrics": {},
  "dataset_size": 1000,
  "created_at": "datetime"
}
```

#### `audit_logs`
```json
{
  "_id": "ObjectId",
  "user_id": "ObjectId",
  "action": "model_upload | prediction | explanation | bias_check | export",
  "resource_type": "string",
  "resource_id": "ObjectId",
  "metadata": {},
  "ip_address": "string",
  "created_at": "datetime"
}
```

---

## 10. API Reference

### Authentication
| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/v1/auth/register` | Create account |
| POST | `/api/v1/auth/login` | Get JWT token |
| POST | `/api/v1/auth/refresh` | Refresh token |
| POST | `/api/v1/auth/api-keys` | Generate API key |

### Models
| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/v1/models` | List user's models |
| POST | `/api/v1/models/upload` | Upload model file |
| POST | `/api/v1/models/connect` | Connect external API model |
| GET | `/api/v1/models/{id}` | Get model details |
| PUT | `/api/v1/models/{id}` | Update model metadata |
| DELETE | `/api/v1/models/{id}` | Delete model |

### Predictions
| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/v1/predict/{model_id}` | Single prediction |
| POST | `/api/v1/predict/{model_id}/batch` | Batch prediction (CSV) |
| GET | `/api/v1/predict/history` | Prediction history |
| GET | `/api/v1/predict/{prediction_id}` | Get prediction result |

### Explanations
| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/v1/explain/local` | Local SHAP explanation |
| POST | `/api/v1/explain/global/{model_id}` | Global SHAP summary |
| POST | `/api/v1/explain/lime` | LIME local explanation |
| GET | `/api/v1/explain/{task_id}` | Get async explanation result |
| POST | `/api/v1/explain/{id}/export` | Export explanation as PDF/JSON |

### Bias & Fairness
| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/v1/bias/analyze` | Run bias analysis |
| GET | `/api/v1/bias/reports/{model_id}` | Get bias report history |
| GET | `/api/v1/bias/compare` | Compare bias across models |

### Model Comparison
| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/v1/compare` | Compare two or more models |
| GET | `/api/v1/compare/{comparison_id}` | Get comparison result |

---

## 11. Core Features Deep Dive

### 11.1 Model Upload Wizard (Frontend)
The upload wizard is a 4-step form:
1. **Basic Info** — Name, description, task type (classification/regression), tags
2. **File Upload** — Drag & drop model file, auto-detect framework from extension
3. **Feature Schema** — Define input features: name, data type, allowed ranges/categories
4. **Background Data** — Optional: upload a sample dataset used for SHAP KernelExplainer reference

### 11.2 Prediction + Explain Flow
When a user submits a prediction form:
1. Frontend POSTs to `/predict/{model_id}` → returns prediction + probability immediately
2. Frontend simultaneously POSTs to `/explain/local` → returns `task_id`
3. WebSocket subscription on `task_id` waits for SHAP completion
4. On completion, waterfall chart and NL explanation populate in real time

### 11.3 NL Explanation Generation
The NLG service receives SHAP output and produces human-readable text using an LLM:

```
Prompt Template:
"The model predicted [CLASS] with [CONFIDENCE]% confidence.
The top contributing factors were:
- [FEATURE_1] = [VALUE_1] (impact: +[SHAP_1]) — this [increased/decreased] the prediction
- [FEATURE_2] = [VALUE_2] (impact: [SHAP_2]) ...

Generate a clear, 2–3 sentence explanation for a non-technical user."
```

### 11.4 Global SHAP Dashboard
- Displays mean absolute SHAP values for top 20 features
- Interactive beeswarm plot showing value distribution per feature
- Dependence plot: select any two features to visualize their interaction
- Filter by: date range, input value thresholds, prediction classes

### 11.5 Bias Detection Flow
1. User selects a model + provides a labeled evaluation dataset + designates protected attribute (e.g., gender, race)
2. Platform runs predictions on entire dataset
3. Computes: Demographic Parity Difference, Equal Opportunity Difference, Disparate Impact Ratio
4. Renders radar chart comparing metrics across groups
5. Highlights features most correlated with protected attribute via SHAP

---

## 12. Extra / Advanced Features

### 12.1 Counterfactual Explanations
**What it does:** For a given prediction, shows the *minimal input changes* needed to flip the outcome.
**Example:** "If your income were $5,000 higher and your credit history were 2 years longer, your loan would have been approved."
**Implementation:** Use `DiCE` (Diverse Counterfactual Explanations) library — `pip install dice-ml`
**Frontend:** Side-by-side input comparison card showing original vs. counterfactual

### 12.2 What-If Analysis Tool
An interactive sandbox where users can adjust individual input feature values via sliders and instantly see how predictions and SHAP values change — without re-uploading or retraining the model. Powered by real-time API calls with debounced slider input.

### 12.3 Model Monitoring & Drift Detection
- Track prediction distributions over time
- Detect **data drift** using KL divergence / Population Stability Index (PSI)
- Detect **concept drift** via sliding window accuracy monitoring
- Automated alerts via email/Slack webhook when drift exceeds threshold
- **Library:** `evidently` for drift reports

### 12.4 Explanation Audit Trail & Compliance Reports
Generate full audit-ready PDF reports for regulatory compliance (GDPR Article 22, EU AI Act, ECOA/FCRA in finance):
- Model card (description, metrics, intended use, limitations)
- Sample of predictions + explanations for review period
- Bias analysis results
- Data provenance summary
- Digital signature for tamper evidence

### 12.5 Multi-Modal Support
Extend beyond tabular data:
- **Image models** — LIME image explainer + Grad-CAM heatmaps
- **Text/NLP models** — LIME text explainer, SHAP for transformers (via `shap.Explainer` with HuggingFace)
- **Time series models** — SHAP with sliding window features

### 12.6 Collaborative Workspace
- **Teams & Organizations:** Multi-user access with role-based permissions (Admin, Analyst, Viewer)
- **Model annotations:** Users can annotate specific predictions with comments
- **Explanation reviews:** Mark explanations as "reviewed" or "flagged for audit"
- **Version control:** Upload multiple versions of the same model and compare explanations across versions

### 12.7 AutoML Integration
Integrate with AutoML frameworks so users can train models directly within the platform:
- Train using `auto-sklearn` or `TPOT` on uploaded datasets
- Automatically generate model cards on completion
- Explanations generated automatically post-training

### 12.8 Causal Inference Layer
Go beyond correlational SHAP values toward causal understanding:
- Integrate `DoWhy` library for causal effect estimation
- Show users which features have causal vs. merely correlational impact
- Especially valuable in healthcare and policy applications

### 12.9 Prediction Confidence & Uncertainty Quantification
- For probabilistic models, display calibration curves (reliability diagrams)
- Monte Carlo Dropout for neural network uncertainty estimation
- Conformal prediction intervals — display a range of plausible outputs rather than a point estimate
- Visual uncertainty bars on prediction charts

### 12.10 Federated Explanation Mode
For privacy-sensitive domains (healthcare, HR):
- Model stays on the client's server — never uploaded to the platform
- Platform connects via a secure API to run explanations remotely
- Explanation results are returned without raw data leaving the client environment
- Compliant with HIPAA, GDPR data residency requirements

### 12.11 Interactive Tutorial & Onboarding System
- Guided tours for first-time users (using `react-joyride`)
- Sample datasets + pre-loaded demo models (credit scoring, disease prediction)
- Built-in "Explain the Explanation" tooltips on all charts
- Video walkthroughs embedded in help panel

### 12.12 Explanation Marketplace / Community
- Users can share anonymized model explanations publicly
- Upvote interesting explanation patterns
- Community-curated "explainability playbooks" by industry (finance, HR, healthcare)

### 12.13 Real-Time Streaming Explanations
For production models processing live data streams (Kafka / Kinesis):
- Connect to event stream source
- Compute SHAP values on-the-fly with a sliding window
- Dashboard shows live explanation feed with anomaly highlighting

### 12.14 No-Code Model Connector
Visual drag-and-drop interface to connect external REST API models without writing code:
- Define endpoint URL, auth header, input/output field mappings
- Platform automatically wraps the API to support SHAP KernelExplainer
- Useful for proprietary models that cannot be uploaded

---

## 13. Security & Compliance

### Authentication & Authorization
- JWT access tokens (15 min expiry) + refresh tokens (7 days)
- Role-Based Access Control: Admin / Analyst / Viewer
- API key scoping (read-only, predict-only, full-access)
- OAuth2 social login (Google, GitHub) via NextAuth.js

### Data Security
- All model files encrypted at rest (AES-256 via MinIO/S3)
- Database fields containing PII encrypted with application-level encryption
- HTTPS enforced — HTTP → HTTPS redirect via Nginx
- Input validation on all endpoints via Pydantic schemas

### Compliance Considerations
| Regulation | Coverage |
|---|---|
| GDPR Art. 22 | Explanation audit logs, right-to-explanation reports |
| EU AI Act | High-risk model documentation, bias reports |
| ECOA / FCRA | Credit model bias detection, adverse action notice templates |
| HIPAA | Federated mode, no raw patient data stored |

---

## 14. Deployment & DevOps

### Docker Compose (Development)

```yaml
# docker-compose.yml
version: "3.9"
services:
  backend:
    build: ./backend
    ports: ["8000:8000"]
    environment:
      - MONGODB_URL=mongodb://mongo:27017/xai
      - REDIS_URL=redis://redis:6379
      - MINIO_ENDPOINT=minio:9000
    depends_on: [mongo, redis, minio]

  frontend:
    build: ./frontend
    ports: ["3000:3000"]
    environment:
      - NEXT_PUBLIC_API_URL=http://localhost:8000

  worker:
    build: ./backend
    command: celery -A app.workers.celery_app worker --loglevel=info
    depends_on: [redis, backend]

  mongo:
    image: mongo:7
    volumes: [mongo_data:/data/db]

  redis:
    image: redis:7-alpine

  minio:
    image: minio/minio
    command: server /data --console-address ":9001"
    volumes: [minio_data:/data]

volumes:
  mongo_data:
  minio_data:
```

### Production Kubernetes Setup
- Backend: 3 replicas with HPA (scale on CPU > 70%)
- Celery workers: 5 replicas (scale on Redis queue depth)
- MongoDB: Atlas managed cluster (M10+)
- Redis: ElastiCache (AWS) or Upstash
- MinIO: replaced with AWS S3 / GCS

### CI/CD Pipeline (GitHub Actions)
```
Push to main
    │
    ├── Lint (Ruff, ESLint)
    ├── Unit Tests (pytest, Vitest)
    ├── Integration Tests (Playwright)
    ├── Build Docker images
    ├── Push to Container Registry
    └── Deploy to Kubernetes (Helm upgrade)
```

---

## 15. Testing Strategy

### Backend Tests

| Layer | Tool | Coverage |
|---|---|---|
| Unit tests | pytest | All services, 80%+ coverage |
| Integration | pytest + Motor | API endpoints, DB ops |
| ML tests | pytest | SHAP/LIME output validation |
| Load tests | Locust | 100 concurrent prediction requests |

### Frontend Tests

| Layer | Tool | Coverage |
|---|---|---|
| Component | Vitest + React Testing Library | All chart + form components |
| E2E | Playwright | Full prediction + explain flows |
| Visual regression | Chromatic (Storybook) | Chart rendering |

### Key Test Scenarios
- Upload sklearn model → predict → get SHAP explanation → verify waterfall values
- Upload XGBoost model → run bias analysis → verify disparate impact ratio
- Async SHAP task → verify Celery job completion → verify WebSocket event fired
- API key auth → verify rate limiting enforcement
- Batch prediction CSV → verify all rows returned with valid predictions

---

## 16. Milestones & Timeline

| Phase | Duration | Key Deliverable |
|---|---|---|
| Phase 1: Foundation | Weeks 1–3 | Auth + model upload working |
| Phase 2: Prediction | Weeks 4–5 | Live predictions with history |
| Phase 3: Explainability | Weeks 6–9 | Full SHAP + LIME with charts |
| Phase 4: Analytics | Weeks 10–12 | Bias detection + NL explanations |
| Phase 5: API + Polish | Weeks 13–15 | External API + production deploy |
| **Total** | **~15 weeks** | **Production-ready platform** |

---

## 17. Future Roadmap

### 6–12 Month Horizon
- **LLM-native explanations** — use GPT-4o / Claude to provide conversational explanations ("Ask your model anything")
- **Explanation fine-tuning** — allow domain experts to rate explanations and fine-tune the NLG model
- **Plugin ecosystem** — third-party explanation method plugins (custom explainers)
- **Mobile app** — React Native dashboard for monitoring predictions on the go
- **Enterprise SSO** — SAML 2.0 / LDAP integration

### 12–24 Month Horizon
- **Causal ML studio** — full causal graph editor integrated with DoWhy
- **Synthetic data generation** — generate privacy-safe training data with explanations
- **Regulation assistant** — AI-powered assistant that maps model behavior to regulatory requirements automatically
- **On-premise enterprise edition** — fully air-gapped deployment with Helm chart + license server

---

## Appendix: Key Libraries & Versions

```
# Python (backend/requirements.txt)
fastapi==0.115.0
uvicorn[standard]==0.32.0
motor==3.6.0               # Async MongoDB
celery[redis]==5.4.0
shap==0.46.0
lime==0.2.0.1
scikit-learn==1.5.0
xgboost==2.1.0
lightgbm==4.5.0
onnxruntime==1.19.0
dice-ml==0.11
evidently==0.4.40
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
boto3==1.35.0              # S3/MinIO
pydantic-settings==2.5.0

# JavaScript (frontend/package.json)
next: ^14.2.0
react: ^18.3.0
recharts: ^2.13.0
d3: ^7.9.0
tailwindcss: ^3.4.0
@tanstack/react-query: ^5.59.0
zustand: ^5.0.0
react-hook-form: ^7.53.0
zod: ^3.23.0
```

---

*This document is a living specification. Update it as architectural decisions evolve.*
# XAI Platform — Framework Extensions Documentation

This document details the integration of **InterpretML**, **Alibi Explain**, and **AIX360** into the XAI Platform. These extensions provide advanced explainability techniques beyond standard SHAP and LIME.

## 1. Supported Frameworks

| Framework | Core Technique | Output Type | Best For |
| :--- | :--- | :--- | :--- |
| **InterpretML** | Explainable Boosting Machines (EBM) | Feature Importance | High-accuracy glass-box models. |
| **Alibi Explain** | Anchors | Boolean Rules (IF-THEN) | Precision-focused local explanations. |
| **AIX360** | Boolean Rule Column Generation (BRCG) | Rule Sets | Human-readable audited decisions. |

---

## 2. Architecture Overview

### Backend Integration
The frameworks are integrated as **Asynchronous Celery Workers**.
- **Trigger**: `POST /api/v1/explain/{framework}/{model_id}`.
- **Task ID Storage**: When a task is triggered, its Celery `task_id` is saved to the **Prediction** document in MongoDB (`interpretml_task_id`, `alibi_task_id`, `aix360_task_id`).
- **Unified Polling**: The generic endpoint `GET /api/v1/explain/prediction/{id}?method={framework}` checks for these specific task IDs and returns the status (`pending`, `complete`, `failed`).

### Frontend Integration
Specialized React components handle the diverse data structures returned by these libraries:
- **`AIX360RuleDisplay`**: Renders a numbered list of Boolean rules with AND/OR highlighting.
- **`AlibiRuleDisplay`**: Displays "Anchor" conditions that guarantee the model's prediction.
- **`FeatureImportanceBar`**: A reusable chart for InterpretML's local feature contributions.

---

## 3. Deployment & Dependencies

These frameworks have strict dependency requirements (e.g., specific versions of `scikit-learn` and `numpy`).

To ensure stability, dependencies are managed in:
- `notebooks/requirements-xai-frameworks.txt`
- The `xai-platform-worker` Docker image.

### Optimization Note
The workers use `--pool=solo` to avoid memory fragmentation when loading large XAI models (like Alibi's Anchor Tabular) alongside the primary ML models.

---

## 4. Troubleshooting

### "Framework Unavailable" Error
If you see this in the UI, it typically means the `worker` container is missing the required library.
1. Check worker logs: `docker logs xai-platform-worker-1`
2. Ensure consistent Python environments: The worker must be built with the packages in `requirements-xai-frameworks.txt`.

### Polling Timeout
Complex explanations (especially Alibi Anchors) can take 10-30 seconds. The frontend polls every 2 seconds. This is normal behavior for high-precision XAI techniques.

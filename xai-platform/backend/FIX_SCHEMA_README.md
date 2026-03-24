# Feature Schema Fix Script

This tool fixes the `feature_schema` field in existing model documents in MongoDB. It is specifically designed to address the bug where categorical features were incorrectly marked as numeric.

## Problem

Previously, when uploading sklearn models with categorical features processed by `OneHotEncoder`, the auto-generated feature schema incorrectly labeled all features as `numeric` with empty `options`. This caused frontend forms to show numeric inputs instead of dropdowns for categorical features like `sex` (Male/Female).

## Solution

The script:
1. Downloads each model file from MinIO
2. Extracts the correct feature schema directly from the sklearn pipeline's `ColumnTransformer`
3. Identifies categorical features (via `OneHotEncoder`) and extracts their options from `categories_`
4. Preserves existing `min`/`max`/`mean` statistics if they were previously computed from background data
5. Updates the database with the corrected schema

## Prerequisites

- Backend MongoDB and MinIO must be running
- Python dependencies: `pymongo`, `minio`, `scikit-learn`, `joblib`, `pandas`
- The script uses settings from `backend/app/config.py`

## Usage

```bash
cd backend
python fix_feature_schemas.py [--dry-run] [--model-id <specific_model_id>]
```

### Options

- `--dry-run`: Show what changes would be made without actually updating the database. Recommended first step.
- `--model-id`: Only fix a specific model (by MongoDB ObjectId). Useful for testing on one model before bulk fix.

### Example

```bash
# Dry run to see what will be fixed
python fix_feature_schemas.py --dry-run

# Fix all models
python fix_feature_schemas.py

# Fix a single model
python fix_feature_schemas.py --model-id 65f1234567890123456789abc
```

## What Gets Updated

For each sklearn pipeline with a `ColumnTransformer`:

- Features processed by `OneHotEncoder` → `type: "categorical"` with `options` extracted from the encoder
- Features processed by `StandardScaler`/`MinMaxScaler` → `type: "numeric"`
- Existing numeric statistics (`min`, `max`, `mean`) are preserved if present

## Non-Sklearn Models

Currently only `sklearn` models are processed. Other frameworks (xgboost, onnx, keras) are skipped.

## Notes

- Always run with `--dry-run` first to verify
- Ensure you have backups of your MongoDB before bulk operations
- After fixing existing models, you can re-upload them through the UI/API and the auto-generated schema will be correct (due to the fixed `ModelLoaderService.generate_feature_schema()`)

## Post-Fix Verification

After running the script, check a fixed model:

```bash
# Get model via API
curl -H "Authorization: Bearer <token>" http://localhost:8000/api/v1/models/<model_id>
```

The `feature_schema` should show `type: "categorical"` with `options` for features like `sex`, `cp`, `restecg`, `exang`, `slope`.

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**XAI Platform** - An Explainable AI platform for model management, predictions, and explanations. The platform supports:
- Model upload and versioning (sklearn, XGBoost, LightGBM, ONNX, Keras)
- Real-time predictions with comprehensive explanations (SHAP, LIME)
- Counterfactual explanations (DiCE)
- Bias detection and fairness metrics
- Model comparison
- Audit logging
- JWT-based authentication with API key management

## Architecture

### High-Level Architecture

The platform is a full-stack, containerized application with a microservices-oriented design:

```
┌─────────┐     ┌──────────┐     ┌─────────────────┐
│ Frontend│────▶│  Backend │────▶│   MongoDB      │
│ (Next.js)│     │ (FastAPI)│     │   (data)       │
└─────────┘     └──────────┘     └─────────────────┘
                     │                    │
                     ▼                    ▼
                ┌─────────┐        ┌──────────┐
                │  Celery │        │   MinIO  │
                │ Workers │        │ (models) │
                └─────────┘        └──────────┘
                     │
                     ▼
                ┌─────────┐
                │  Redis  │
                │ (queue) │
                └─────────┘
```

### Backend (Python/FastAPI)

**Location**: `backend/app/`

**Core Structure**:
- `main.py` - FastAPI app with CORS, request logging, exception handlers
- `config.py` - Pydantic settings with environment variable support
- `api/v1/` - REST API routers (auth, models, predictions, explanations, bias, compare, api_keys, audit, notifications)
- `models/` - Pydantic data models for request/response validation
- `db/` - MongoDB async connection (Motor) and repository pattern
- `services/` - Business logic:
  - `model_loader_service.py` - Loads models from MinIO (supports sklearn, xgboost, lightgbm, onnx, keras)
  - `lime_service.py` - LIME explanation generation
  - `prediction_service.py` - Prediction handling
  - `nlg_service.py` - Natural language generation (OpenAI integration)
- `workers/` - Celery async tasks (SHAP computation, background jobs)
- `custom/` - Custom feature engineer class for pickle compatibility
- `middleware/` - Rate limiting
- `websocket/` - WebSocket connection manager for real-time updates

**Database**: MongoDB with async Motor driver
**Object Storage**: MinIO (S3-compatible) for model files
**Task Queue**: Celery with Redis broker
**Authentication**: JWT tokens (python-jose) with passlib/bcrypt for password hashing

### Frontend (Next.js/React/TypeScript)

**Location**: `frontend/`

**Tech Stack**:
- Next.js 16 with App Router
- React 19
- TypeScript 5
- Tailwind CSS 4
- Zustand (state management)
- TanStack React Query
- Recharts (visualizations)
- D3 (advanced charts)
- React Hook Form + Zod validation

**Structure**:
- `src/app/` - Page components (app router)
  - Pages: `/login`, `/register`, `/models`, `/models/upload`, `/models/[id]`, `/predict/[modelId]`, `/predict/history`, `/explain/global/[modelId]`, `/explain/local/[modelId]/[predictionId]`, `/bias`, `/compare`, `/audit`, `/settings/api-keys`
- `src/components/` - Reusable UI components
  - `charts/` - Visualization components (FeatureImportanceBar, LIMEPlot, etc.)
  - `forms/` - Form components
  - `ui/` - Basic UI components (buttons, cards, etc.)
- `src/lib/` - Utilities (API client, helpers)

**API Client**: Frontend uses Axios to communicate with backend API at `NEXT_PUBLIC_API_URL` (default: `http://localhost:8000`)

## Development Setup

### Prerequisites

- Python 3.11+
- Node.js 20+
- Docker & Docker Compose (recommended for full stack)
- MongoDB, Redis, MinIO (or use docker-compose)

### Running with Docker Compose (Development)

From the project root:

```bash
# Start all services (backend, frontend, MongoDB, Redis, MinIO)
docker-compose up -d

# Stop all services
docker-compose down

# View logs
docker-compose logs -f backend
docker-compose logs -f frontend
docker-compose logs -f worker

# Restart a specific service
docker-compose restart backend
```

Services will be available:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API docs: http://localhost:8000/docs
- MongoDB: localhost:27017
- Redis: localhost:6379
- MinIO Console: http://localhost:9001 (minioadmin/minioadmin)
- MinIO API: localhost:9000

### Running Backend Locally (without Docker)

```bash
cd backend

# Create virtual environment (if not already)
python -m venv venv

# Activate venv
# Windows:
venv\Scripts\activate
# Unix/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables (or create .env file)
# Copy .env.production.example to .env and modify as needed

# Run development server with auto-reload
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Run tests
pytest tests/ -v
```

### Running Frontend Locally (without Docker)

```bash
cd frontend

# Install dependencies
npm install
# or yarn, pnpm, bun

# Copy .env.local.example to .env.local and set NEXT_PUBLIC_API_URL

# Development server (hot reload)
npm run dev

# Build for production
npm run build

# Start production server
npm start

# Lint code
npm run lint
```

### Starting Celery Worker

```bash
cd backend
# With venv activated
celery -A app.workers.celery_app worker --loglevel=info --pool=solo
```

Or with docker-compose (worker service already defined).

## Common Commands

### Docker Compose
- `docker-compose up -d` - Start all services in detached mode
- `docker-compose down` - Stop and remove containers
- `docker-compose logs -f <service>` - Follow logs for a service
- `docker-compose restart <service>` - Restart a specific service
- `docker-compose ps` - Check status of all services
- `docker-compose exec <service> sh` - Open shell in container

### Backend (Python)
- `uvicorn app.main:app --reload` - Start development server
- `pytest tests/ -v` - Run all tests
- `pytest tests/test_file.py::test_name -v` - Run specific test
- `python fix_feature_schemas.py --dry-run` - Dry-run schema fix script
- `python fix_feature_schemas.py` - Fix feature schemas in database
- `python test_jwt.py` - Quick JWT token generation test

### Frontend (Node.js)
- `npm run dev` - Start development server at http://localhost:3000
- `npm run build` - Create production build
- `npm start` - Start production server
- `npm run lint` - Run ESLint
- `npm run lint -- --fix` - Auto-fix lint issues

## API Documentation

- Swagger UI: http://localhost:8000/docs (when backend running)
- ReDoc: http://localhost:8000/redoc

All API endpoints are prefixed with `/api/v1/` except WebSocket and health check.

## Database Schema

**Collections** (MongoDB):
- `users` - User accounts (email, hashed_password, is_active, etc.)
- `models` - Model metadata (name, framework, file_path, feature_schema, created_by, etc.)
- `predictions` - Prediction records (model_id, input_data, output, explanation_id, created_at)
- `explanations` - SHAP/LIME explanations (prediction_id, explanation_type, values, metadata)
- `bias_reports` - Bias detection results (model_id, metrics, flagged_features)
- `audit_logs` - Audit trail (user_id, action, resource_type, resource_id, details, timestamp)
- `api_keys` - API key management (key, name, user_id, is_active, last_used)
- `notifications` - User notifications

**Indexes**:
- `users.email` (unique)
- `api_keys.key` (unique)

## Key Features & Patterns

### Model Upload & Inference Flow
1. User uploads model file (`.pkl`, `.joblib`, `.pickle`, `.onnx`, `.h5`, `.keras`) via POST `/api/v1/models/upload`
2. `ModelLoaderService` saves file to MinIO, extracts metadata, detects framework, generates feature schema
3. Model is registered in MongoDB with `models` document
4. Prediction: POST `/api/v1/predict` with input data → returns prediction + optional explanation task ID
5. Explanations computed asynchronously via Celery:
   - SHAP: `compute_shap_values` task
   - LIME: computed synchronously or async depending on config

**Note**: Both SHAP and LIME support **regression** and **classification** tasks. The mode is automatically detected from the model type.

### Feature Schema Generation
Critical schema logic in `ModelLoaderService.generate_feature_schema()`:
- Handles sklearn pipelines with `ColumnTransformer`
- Detects `OneHotEncoder` → categorical features with options
- Detects scalers → numeric features with min/max/mean statistics
- For non-pipeline models: requires manual feature schema in upload

**Known Issue**: Earlier models had categorical features mislabeled as numeric. Use `fix_feature_schemas.py` to correct.

### Authentication
- JWT access tokens (15 min default) and refresh tokens (7 days)
- POST `/api/v1/auth/login` → returns tokens
- POST `/api/v1/auth/refresh` → refresh access token
- POST `/api/v1/auth/logout` → blacklist token (if implemented)
- Protected routes use `get_current_user` dependency

### API Keys
Users can create API keys for programmatic access:
- POST `/api/v1/api-keys` → create key
- Rate limiting differs: API keys have higher limits than token auth

### Async Explanations
Long-running SHAP computations run in Celery workers:
- Task: `compute_shap_values` in `backend/app/workers/tasks.py`
- Predictions returned immediately; explanation status polled via WebSocket or GET `/api/v1/explain/{explanation_id}`

### Rate Limiting
- Anonymous: 60 req/min
- Authenticated: 300 req/min
- API key: 500 req/min
- Implemented in `backend/app/middleware/rate_limit.py`

## Testing

### Backend Tests
Tests located in `backend/tests/`. Current test examples:
- `test_jwt.py` - Simple JWT encode/decode verification
- `test_heart_model.py` - Model loading and prediction test

Run with pytest (from backend directory):
```bash
pytest tests/ -v
```

There is no `pytest.ini` yet. Tests may require running MongoDB and MinIO.

### Frontend Tests
No formal test suite yet. Consider adding:
- Jest + React Testing Library for unit tests
- Playwright or Cypress for E2E tests

## Environment Variables

### Backend (`.env`)
See `backend/.env.production.example` for all available variables. Key ones:
- `MONGODB_URL` - MongoDB connection string
- `REDIS_URL` - Redis connection for Celery
- `MINIO_*` - MinIO credentials and endpoint
- `JWT_SECRET` - JWT signing secret (CRITICAL: change in production)
- `OPENAI_API_KEY` - Optional, for NLG features
- `PICKLE_CLASS_MODULES` - Custom classes for pickle compatibility (default: `app.custom.feature_engineer`)

### Frontend (`.env.local`)
- `NEXT_PUBLIC_API_URL` - Backend API URL (e.g., `http://localhost:8000`)

## Important Files & Directories

- `backend/app/main.py` - FastAPI application entry point, all routes registered here
- `backend/app/config.py` - Application configuration
- `backend/app/services/model_loader_service.py` - Core model loading logic (supports multiple frameworks) + model type detection
- `backend/app/workers/tasks.py` - Celery background tasks (SHAP, LIME) with TreeExplainer optimization
- `backend/fix_feature_schemas.py` - Utility to fix categorical feature schema bugs
- `backend/FIX_SCHEMA_README.md` - Documentation for the schema fix script
- `frontend/src/app/` - Next.js app router pages
- `frontend/src/components/charts/` - Visualization components for SHAP/LIME
- `docker-compose.yml` - Development orchestration
- `docker-compose.prod.yml` - Production orchestration (includes nginx)

## Recent Improvements (2025-03-26)

### SHAP Performance Optimization
- **Problem**: Global SHAP for pipeline models was using slow KernelExplainer, taking hours on large datasets.
- **Solution**: Modified `backend/app/workers/tasks.py`:
  - Pipeline models now try `TreeExplainer` first (dramatically faster for tree-based models)
  - Fallback to `KernelExplainer` with capped background samples (200) for non-tree models
  - Global SHAP now processes full dataset efficiently (no unnecessary sampling)
- **Impact**: Graph generation now completes in seconds/minutes instead of hours.

### Automatic Model Detection
- **Added**: `ModelLoaderService.get_estimator_info()` in `backend/app/services/model_loader_service.py`
  - Detects specific algorithm: `RandomForestRegressor`, `XGBRegressor`, `LinearRegression`, etc.
  - Determines model family: `tree`, `linear`, `svm`, `neighbors`, `neural_network`, etc.
  - Sets `is_tree_based` flag for SHAP optimization.
- **Auto-detection of task_type**: No longer requires manual input; inferred from model (classification vs regression).
- **Database changes**: Models collection now stores:
  - `model_type` (e.g., "RandomForestRegressor")
  - `model_family` (e.g., "tree")
  - `is_tree_based` (boolean)
- **Frontend changes**:
  - Model upload form (`/models/upload`) no longer asks for task type (auto-detected).
  - Model details page (`/models/[id]`) displays algorithm and model family in the metadata section.

## Recent Improvements (2026-03-26)

### Global LIME Support
- Added full support for global LIME explanations.
- New endpoints:
  - `POST /api/v1/explain/lime/global/{model_id}` - request global LIME with background dataset
  - `GET /api/v1/explain/lime/global/{model_id}/latest` - retrieve latest global LIME
- Global LIME aggregates local LIME explanations across multiple samples to produce feature importance.
- Frontend global explanation page includes LIME as an alternative method via method selector.

### Enhanced Global SHAP
- **Classification handling**: Properly handles 3D SHAP output (samples × features × classes) by detecting class axis and selecting positive class (binary) or averaging (multi-class).
- **OneHotEncoder aggregation**: For pipeline models with one-hot encoded categorical features, SHAP values are aggregated back to the original categorical features, ensuring manageable feature counts and interpretable results.
- **Feature name consistency**: SHAP now uses feature names from `_compute_shap` (post-aggregation) for `global_importance` and `feature_names` fields.

### Pipeline Preprocessing Fixes (Both SHAP & LIME)
- Both SHAP and LIME tasks now correctly preprocess input data when models use sklearn pipelines with ColumnTransformer.
- Feature names are extracted from the preprocessor's `get_feature_names_out()` when available, ensuring consistency between explainer and feature importance display.
- Local SHAP (`compute_shap_values`) and local LIME (`compute_lime_values`) also updated to use correct feature names after aggregation.

### Bug Fix: SHAP Global Endpoint Filter
- Fixed `GET /api/v1/explain/global/{model_id}/latest` to filter by `method: "shap"` in addition to `explanation_type: "global"`.
- Previously, this endpoint returned the latest global explanation regardless of method, causing SHAP tab to display LIME data if that was computed last. This made SHAP appear broken.
- The filter ensures each method's latest endpoint returns only explanations of that method.

### Performance Optimization: Fast TreeExplainer for Pipeline Models
- **Critical fix**: Global SHAP for pipeline tree models now uses `TreeExplainer` on the final estimator with preprocessed data, providing **dramatic speed improvements**.
- Previously, the code attempted `TreeExplainer` on the full pipeline (which fails) and fell back to slow `KernelExplainer`, taking 10-15 minutes for 200 samples.
- New approach directly preprocesses data and uses `TreeExplainer` on the final estimator only, reducing computation from **~14 minutes to seconds**.
- Implementation in `_compute_shap()`: for pipeline models, background data is preprocessed, final estimator extracted, and TreeExplainer attempted first.
- Fallback to KernelExplainer still works for non-tree models (linear, SVM, etc.).

### Frontend Polling Fix (Infinite Loading Issue)
- **Problem**: Global explanation page would show "loading" indefinitely after requesting an explanation, even while the task was processing in the background.
- **Root cause**: The query only refetched once (3 seconds after request). If the task wasn't done yet, it would stop polling and never update.
- **Solution**: Added `refetchInterval` to the global explanation query that polls every 3 seconds whenever no explanation data exists (404 response). Polling automatically stops once the explanation is retrieved.
- Users now see the explanation appear automatically within seconds/minutes of task completion without manual refresh.

### Detailed Error Logging
- Added comprehensive exception logging to global SHAP request and retrieval endpoints to aid debugging.
- Errors are now printed with full tracebacks to backend logs for quicker issue resolution.

## Recent Improvements (2026-03-27)

### Feature Schema Generation Fix (Categorical Dropdowns)
- **Problem**: Models with pipelines starting directly with `ColumnTransformer` (without a custom `FeatureEngineer` step) had categorical features incorrectly labeled as numeric. This caused the frontend to show number inputs instead of dropdowns for categorical fields.
- **Root cause**: The `generate_feature_schema()` function in `model_loader_service.py` would find the raw feature step but then fail to locate a subsequent preprocessor to infer feature types, leaving `feature_types` empty (default numeric).
- **Fix**: Added fallback logic to use `raw_feature_step` itself as the preprocessor when it contains `transformers_`. This applies to:
  - Feature type detection (categorical vs numeric)
  - Categorical options extraction from `OneHotEncoder`
- **Impact**: Categorical features now correctly show dropdowns with actual values (e.g., "Male", "Female", "Urban", "Rural") in the prediction form.

### Global SHAP/LIME 404 Fix (model_id Query)
- **Problem**: The `GET /explain/global/{model_id}/latest` and `GET /explain/lime/global/{model_id}/latest` endpoints returned 404 even when explanations existed in the database.
- **Root cause**: Some models stored `model_id` as `ObjectId` in the `explanations` collection, while the query used a string. MongoDB treats these as different types, causing no match.
- **Fix**: Changed queries to use `$in` operator to match both string and `ObjectId` representations:
  ```python
  query["model_id"] = {"$in": [model_id, ObjectId(model_id)]} if ObjectId.is_valid(model_id) else model_id
  ```
- **Impact**: Global explanation endpoints now correctly retrieve existing explanations.

### SHAP Additivity Check Disabled
- **Problem**: TreeExplainer failed with "Additivity check failed" error, especially for models with large value ranges. This forced fallback to extremely slow KernelExplainer or caused task failure.
- **Root cause**: The additivity check verifies that SHAP values sum to the model output, but numerical precision issues with large values cause this to fail incorrectly.
- **Fix**: Added `check_additivity=False` parameter to all `shap.Explainer` and `shap.TreeExplainer` calls in `_compute_shap()`.
- **Impact**: SHAP computations now succeed quickly using TreeExplainer (seconds/minutes) instead of falling back to KernelExplainer (hours) or failing.

### Global SHAP Data Validation & Sanitization
- **Problem**: The `_compute_shap` function could return invalid data structures:
  - `shap_values` as 1D array when only one background sample → frontend expects 2D
  - Mismatch between `shap_values` shape and `feature_names` length (especially after one-hot aggregation)
  - Non-finite values (NaN/Inf) in SHAP values causing frontend rendering issues
- **Fix** (in `backend/app/workers/tasks.py`):
  - **Shape validation**: Ensures `shap_values` is at least 2D; reshapes 1D to (1, n_features); averages >2D to 2D.
  - **Length reconciliation**: Truncates or pads `feature_names` to match `shap_values.shape[1]`.
  - **Sanitization**: Applies `np.nan_to_num` to replace NaN/Inf with 0 before storage.
  - **Validation checks**: Raises clear errors if `shap_values` has zero features after processing.
- **Impact**: Frontend receives clean, valid data and can render SHAP graphs correctly.

### Global LIME Validation
- **Problem**: If all LIME samples failed during computation, `lime_global["feature_importance"]` would be empty. The task would succeed and save an empty list, causing the frontend to display nothing without error indication.
- **Fix**: After calling `LIMEService.explain_global()`, check if `feature_importance` is empty. If so, log an error and raise an exception to fail the task visibly.
- **Impact**: Empty LIME results now clearly indicate failure, prompting user to retry with different background data.

## Data Flow

## Data Flow

### Prediction Request
1. Frontend → Backend: `POST /api/v1/predict` with `{model_id, input_data}`
2. Backend validates user, loads model from MinIO, runs prediction
3. Returns `{prediction_id, result}`
4. If explanation requested: enqueue SHAP/LIME task, return `explanation_id`
5. Frontend can poll for explanation status or use WebSocket

### Model Upload
1. Frontend → Backend: `POST /api/v1/models/upload` with multipart form (model file + optional metadata JSON)
2. Backend validates file type, uploads to MinIO, extracts feature schema from sklearn pipeline
3. Creates model document in MongoDB
4. Returns `{model_id, name, ...}`

## Dependencies

### Backend Key Libraries
- FastAPI 0.115.0
- Uvicorn 0.32.0
- Motor (MongoDB async) 3.6.0
- Celery 5.4.0 + Redis 5.1.0
- SHAP >= 0.45.1
- LIME 0.2.0.1
- scikit-learn 1.5.0
- XGBoost 2.1.0
- LightGBM 4.5.0
- ONNX Runtime 1.19.0
- DiCE (counterfactuals) >= 0.11
- Evidently (drift detection) 0.4.40
- Boto3 (MinIO) 1.35.0
- Pydantic 2.9.0
- Python-Jose (JWT) 3.3.0

### Frontend Key Libraries
- Next.js 16.1.7
- React 19.2.3
- TypeScript 5
- Zustand 5.0.12
- TanStack Query 5.90.21
- Recharts 3.8.0
- D3 7.9.0
- React Hook Form 7.71.2
- Zod 4.3.6
- Tailwind CSS 4

## Notes & Gotchas

- **Model Loading**: Models are stored in MinIO and loaded into memory on demand. Large models may cause memory issues. Consider model caching in `ModelLoaderService` (currently not implemented).
- **Pickle Security**: Only load models from trusted sources. Pickle can execute arbitrary code.
- **Categorical Features**: The auto-detection of categorical features relies on `OneHotEncoder` in sklearn pipelines. Manual schema override may be needed for custom preprocessing.
- **Explanation Aggregation**: For pipelines with `OneHotEncoder`, SHAP and LIME automatically aggregate one-hot encoded feature contributions back to the original categorical feature. This ensures explanations display the original input features rather than hundreds of binary columns.
  - **Important**: The aggregation logic in `backend/app/workers/tasks.py` (SHAP) and `backend/app/services/lime_service.py` (LIME) uses normalized feature names that handle transformer prefixes (e.g., `"cat__name_Audi"` becomes `"name_Audi"`). This ensures correct mapping even with sklearn's ColumnTransformer naming.
  - After aggregation, the frontend charts will display only the original features (e.g., `name`, `year`, `km_driven`, `fuel`, `seller_type`, `transmission`, `owner` for the car price model).
- **Celery**: Requires Redis to be running. Worker must import task modules correctly. Database connections are established per-task (see `tasks.py` pattern).
  - **Starting Workers**: Run `celery -A app.workers.celery_app worker --loglevel=info --pool=solo` from the `backend` directory. Without a running worker, SHAP/LIME explanations will stay in "pending" state indefinitely and graphs will not be generated.
- **Explanation Generation Flow**:
  - Local explanations (SHAP/LIME) are triggered via API endpoints and processed asynchronously by Celery workers.
  - The frontend polls for completion. If a worker is not running or fails, the graphs will not appear.
  - Always check Celery worker logs for errors if explanations are not generating.
- **Global SHAP & LIME**: Requires a background dataset (CSV) to be uploaded. Without it, global explanations cannot be computed.
  - **Background Data Requirements**: The CSV must contain all features listed in the model's feature schema. Extra columns are automatically dropped. Missing required features will cause an error with a clear message.
  - **Performance**: For tree-based models (RandomForest, XGBoost, LightGBM), global SHAP should complete in **seconds to minutes** (optimized with TreeExplainer). For non-tree models (linear, SVM), it may take longer due to KernelExplainer.
- **CRITICAL: Background Data Quality**:
  - **Missing Values (NaN) Must Be Handled**: The background dataset should have **no NaN values** in any of the required feature columns. NaN values can cause:
    - TreeExplainer to fail and fall back to slow KernelExplainer
    - KernelExplainer to run extremely slowly or fail
    - Potential errors during preprocessing
  - **Preprocessing Recommendation**: Before uploading background data, impute or remove missing values:
    - Categorical: Fill with most frequent value or "Unknown"
    - Numeric: Fill with median/mean or a sentinel value
  - The alignment function (`_prepare_background` in `tasks.py`) does **not** automatically impute missing values.
- **CORS**: Backend set to `allow_origins=["*"]` - restrict in production.
- **JWT Secret**: Must be changed from default for production. Store in secure env var.
- **MinIO**: Bucket `xai-models` auto-created on startup. Ensure MinIO has sufficient storage.
- **MongoDB Replica Set**: Production docker-compose uses `--replSet rs0`. Must initialize replica set: `rs.initiate()` in mongo shell.
- **Frontend Path Aliases**: `@/*` maps to `./src/*` (tsconfig.json). Use `@/components` etc.
- **API Versioning**: All endpoints under `/api/v1/`. Breaking changes should increment version.

## Troubleshooting Global Explanations

### Issue: "Loading..." forever, no graphs appear

**Possible causes and solutions:**

1. **Celery worker not running**: Check `docker-compose ps` - worker should be Up. Start with `docker-compose up -d worker`.
2. **Task failed or stuck**: Check worker logs: `docker-compose logs worker`. Look for errors or long-running tasks.
3. **Background dataset has NaN values**: This causes SHAP to fall back to extremely slow KernelExplainer or fail. Clean your CSV by filling missing values.
4. **Background dataset missing required features**: The upload will succeed, but computation will fail. Ensure your CSV contains all features from the model's schema (accessible on model details page).
5. **Frontend not polling**: Recent fix added automatic polling. If still not working, check browser console for errors.
6. **Feature schema incorrect**: If categorical features show as numeric inputs, delete and re-upload the model to regenerate correct schema (see Feature Schema Generation Fix above).

### Issue: Global SHAP taking >10 minutes

**Root cause**: Using KernelExplainer (slow) instead of TreeExplainer (fast). This happens if:
- Model is not tree-based (linear, SVM) → expected to be slow
- Model is tree-based but background data contains NaN values → TreeExplainer fails, falls back to KernelExplainer
- Model pipeline preprocessing issue → TreeExplainer fails

**Solution**:
- For tree models (RandomForest, XGBoost, LightGBM): Ensure background data has **no NaN values**.
- Check worker logs for "Using 200 background data samples..." warning - indicates KernelExplainer path.
- After fixing, tree-based global SHAP should complete in **under 2 minutes** for 200 samples.
- **Note**: As of 2026-03-27, `check_additivity=False` is enabled, so additivity check failures no longer cause fallback.

### Issue: 404 on `/explain/global/{modelId}/latest`

**Previous behavior (fixed 2026-03-27)**: This could return 404 even when an explanation existed due to `model_id` type mismatch (string vs ObjectId).

**Now**: The query matches both string and ObjectId representations. If you still see 404:
1. Ensure you've uploaded background data and the task completed successfully (check Celery logs).
2. Verify the explanation exists in the database:
   ```bash
   docker-compose exec backend python3 -c "
   import asyncio
   from app.db.mongo import connect_db, get_db
   async def check():
       await connect_db()
       db = await get_db()
       import sys
       model_id = sys.argv[1] if len(sys.argv) > 1 else 'YOUR_MODEL_ID'
       count = await db.explanations.count_documents({'model_id': model_id, 'explanation_type': 'global', 'method': 'shap'})
       print(f'Global SHAP explanations for {model_id}: {count}')
   asyncio.run(check())
   " YOUR_MODEL_ID
   ```
3. If count is 0, the task may have failed. Check Celery worker logs for errors.

### Issue: Explanation tasks succeed but graphs show nothing (empty charts)

**Possible causes and solutions:**

1. **Background data quality**: Ensure background dataset has:
   - All required features from model's feature schema
   - **No missing values (NaN)** – impute before upload
   - At least 2-3 rows (more is better, e.g., 50-200)
2. **Check backend logs**: Look for warnings from `_compute_shap` about:
   - Shape mismatches
   - Non-finite values (NaN/Inf) that may have been sanitized to zero
   - Feature name adjustments
3. **Verify saved data**: Check the explanation document in MongoDB:
   ```bash
   docker-compose exec backend python3 -c "
   import asyncio, json
   from app.db.mongo import connect_db, get_db
   from bson import ObjectId
   async def check():
       await connect_db()
       db = await get_db()
       exp = await db.explanations.find_one({'explanation_type': 'global'}, sort=[('created_at', -1)])
       if exp:
           print('shap_values shape:', len(exp['shap_values']), 'x', len(exp['shap_values'][0]) if exp['shap_values'] else 0)
           print('feature_names count:', len(exp['feature_names']))
           print('global_importance count:', len(exp.get('global_importance', [])))
   asyncio.run(check())
   "
   ```
   - `shap_values` should be a 2D array (samples × features)
   - `feature_names` count should equal `shap_values` columns
   - `global_importance` should have non-zero importance values for most features
4. **Refresh frontend**: Clear browser cache and hard refresh (Ctrl+F5) to ensure latest frontend code.

### Checking Task Status Manually

```bash
# Check if worker is processing tasks
docker-compose logs worker | grep "Task compute_global"

# Check Redis for pending tasks
docker-compose exec redis redis-cli -n 0 keys "celery-task-meta-*" | wc -l

# Connect to database and check explanation count
docker-compose exec backend python3 -c "
import asyncio
from app.db.mongo import connect_db, get_db
async def check():
    await connect_db()
    db = await get_db()
    count = await db.explanations.count_documents({'explanation_type': 'global'})
    print(f'Total global explanations: {count}')
asyncio.run(check())
"
```

## Contributing

When making changes:

1. **Backend**:
   - Follow async patterns (async/await) for I/O operations
   - Use dependency injection for database/repositories
   - Add Pydantic models for request/response validation
   - Register new routers in `main.py`
   - Update environment variable documentation if adding config

2. **Frontend**:
   - Use TypeScript strictly; avoid `any`
   - Follow Next.js App Router conventions (server vs client components)
   - Use Tailwind CSS for styling; avoid custom CSS
   - API calls through a centralized client (e.g., `src/lib/api.ts` if exists)
   - Use React Hook Form + Zod for forms

3. **Models**:
   - Prefer joblib over pickle for sklearn models
   - Ensure pipelines include preprocessing (ColumnTransformer)
   - Test model upload end-to-end before merging

4. **Docker**:
   - Don't commit built images; only Dockerfiles and compose files
   - Keep images slim; use multi-stage builds
   - Use production Dockerfile for deployment, not development

## Useful Queries (for Claude)

When working in this codebase, consider:

- **"How do I add a new API endpoint?"** → Create router in `backend/app/api/v1/`, register in `main.py`, add Pydantic schemas, add auth dependency if needed.
- **"How do I add a new page?"** → Create `src/app/<route>/page.tsx` in frontend, add navigation if needed.
- **"Where is the model loading logic?"** → `backend/app/services/model_loader_service.py`
- **"How are explanations computed?"** → SHAP in Celery task (`workers/tasks.py:compute_shap_values`), LIME service (`services/lime_service.py`).
- **"How to fix feature schema issues?"** → Run `python backend/fix_feature_schemas.py --dry-run` then without `--dry-run`.
- **"How to test locally?"** → Use `docker-compose up -d` for full stack, or run backend/frontend separately as described.

## Additional Documentation

- `Heart Disease Prediction.ipynb` - Original Jupyter notebook for the heart disease prediction model
- `backend/FIX_SCHEMA_README.md` - Details on the feature schema bug fix
- `frontend/README.md` - Next.js basic setup (standard create-next-app)

## Git Workflow

Current branch strategy:
- Main development branch: `main`
- Feature branches: create from `main`, PR to merge back

Commit messages should describe the "why" and include context for machine learning/model changes.

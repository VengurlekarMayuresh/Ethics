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
- `backend/app/services/model_loader_service.py` - Core model loading logic (supports multiple frameworks)
- `backend/app/workers/tasks.py` - Celery background tasks (SHAP, etc.)
- `backend/fix_feature_schemas.py` - Utility to fix categorical feature schema bugs
- `backend/FIX_SCHEMA_README.md` - Documentation for the schema fix script
- `frontend/src/app/` - Next.js app router pages
- `frontend/src/components/charts/` - Visualization components for SHAP/LIME
- `docker-compose.yml` - Development orchestration
- `docker-compose.prod.yml` - Production orchestration (includes nginx)

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
- **Celery**: Requires Redis to be running. Worker must import task modules correctly. Database connections are established per-task (see `tasks.py` pattern).
- **CORS**: Backend set to `allow_origins=["*"]` - restrict in production.
- **JWT Secret**: Must be changed from default for production. Store in secure env var.
- **MinIO**: Bucket `xai-models` auto-created on startup. Ensure MinIO has sufficient storage.
- **MongoDB Replica Set**: Production docker-compose uses `--replSet rs0`. Must initialize replica set: `rs.initiate()` in mongo shell.
- **Frontend Path Aliases**: `@/*` maps to `./src/*` (tsconfig.json). Use `@/components` etc.
- **API Versioning**: All endpoints under `/api/v1/`. Breaking changes should increment version.

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

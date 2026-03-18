# Phase 3: Explainability Engine Implementation Guide

## Backend Implementation

### 1. LIME Integration
- [ ] Implement LIME explainer in `compute_shap_values` and `compute_global_shap` Celery tasks
- [ ] Add LIME-specific API endpoints in `explanations.py`:
  - `POST /api/v1/explain/lime/{model_id}`
  - `GET /api/v1/explain/lime/latest`
- [ ] Update database schema to store LIME explanations

### 2. NLG Service
- [ ] Enhance `nlg_service.py` for explanation generation
- [ ] Cache NLG outputs in Redis
- [ ] Add error handling for LLM failures

### 3. API Endpoints
- [ ] Update `explanations.py` to handle LIME requests
- [ ] Add pagination for explanation history endpoints

## Frontend Development

### 1. Visualization Components
- [ ] SHAP Waterfall chart (`SHAPWaterfall.tsx`)
- [ ] SHAP Beeswarm plot (`SHAPBeeswarm.tsx`)
- [ ] LIME interactive plot (`LIMEPlot.tsx`)

### 2. Explanation Pages
- [ ] `/explain/local/[modelId]/[predictionId]` page
- [ ] `/explain/global/[modelId]` page
- [ ] `/compare/explanations` page

## Testing Plan
- [ ] Unit tests for NLG/export functionality
- [ ] Integration tests for full explanation workflow
- [ ] Performance testing under load

## Dependencies
- [ ] Obtain OpenAI API key for NLG (or set up local Llama)
- [ ] Install charting libraries (Recharts/ApexCharts)

---
## Critical Files
1. `backend/app/services/lime_service.py`
2. `backend/app/api/v1/explanations.py`
3. `frontend/src/components/charts/LIMEPlot.tsx`
# Open Issues & Planned Improvements

This document tracks known issues, technical debt, and planned enhancements for the XAI Platform.

---

## 🚨 Critical Issues (Blocking/User-Facing)

### Issue #1: Background Dataset with NaN Values Causes Extremely Slow SHAP

**Severity**: High (affects user experience)
**Status**: Open (workaround exists, needs permanent fix)
**Affected Component**: Backend - `backend/app/workers/tasks.py`

**Problem**:
- Users upload background datasets containing NaN (missing) values
- SHAP TreeExplainer fails when encountering NaN after preprocessing
- System falls back to KernelExplainer, which is **10-100x slower** (14+ minutes vs seconds)
- KernelExplainer may also struggle with NaN, leading to hangs or failures

**Current Workaround**:
- Users must manually clean their CSV files before uploading (impute missing values)
- Documentation explains this requirement in CLAUDE.md

**Proposed Fix**:
Add automatic NaN handling in the `_prepare_background` function:
1. Detect NaN values in required feature columns
2. For **numeric** columns: fill with median (or mean) of the column
3. For **categorical** columns: fill with most frequent value or "Unknown"
4. Log a warning to inform users that NaNs were auto-filled
5. Proceed with computation

**Implementation Location**:
`backend/app/workers/tasks.py` → `_prepare_background()` function (lines 555-605)

**Estimated Effort**: 1 hour

---

### Issue #2: Frontend Production Build Fails Due to Missing `next-auth`

**Severity**: Medium (affects deployment, not development)
**Status**: Open
**Affected Component**: Frontend - `frontend/package.json` & auth setup

**Problem**:
- `frontend/src/app/api/auth/[...nextauth]/route.ts` imports `next-auth` and providers
- `package.json` does **not** include `next-auth` dependency
- Production build (`npm run build`) fails with "Module not found"
- Development mode works because unused routes aren't compiled

**Root Cause**:
The project uses JWT-based authentication (custom implementation in `backend/app/api/v1/auth.py`), NOT NextAuth. The NextAuth route file appears to be leftover scaffolding or an incomplete feature.

**Proposed Solutions** (choose one):

**Option A: Remove NextAuth Route** (Recommended)
- Delete `frontend/src/app/api/auth/[...nextauth]/route.ts`
- This is the simplest fix; the platform doesn't use NextAuth
- Confirms authentication is handled entirely by backend JWT

**Option B: Add next-auth Dependency**
- If NextAuth is needed for future OAuth integration, add it to `package.json`
- But currently no other code uses it, so this adds unnecessary bloat

**Implementation**:
Remove the unused route file and ensure no other references exist.

**Estimated Effort**: 15 minutes

---

## 📝 Minor Issues & Technical Debt

### Issue #3: Model `model_type`, `model_family`, `is_tree_based` Fields May Be `None`

**Severity**: Low (doesn't break functionality)
**Status**: Open
**Affected Component**: Backend - `ModelLoaderService.get_estimator_info()` & model registration

**Problem**:
- Some models in the database have `null` for `model_type`, `model_family`, `is_tree_based`
- This metadata is important for optimization decisions (e.g., whether to use TreeExplainer)
- Likely caused by older models uploaded before this feature was added, or `get_estimator_info()` failing for certain model types

**Proposed Fix**:
- Backfill these fields for existing models via a migration script
- Improve `get_estimator_info()` to handle edge cases (e.g., wrapped models, custom estimators)
- Log warnings when detection fails, so admins can manually set

**Estimated Effort**: 2 hours

---

### Issue #4: Pydantic Model Field Name Conflicts

**Severity**: Low (cosmetic, shows warnings in logs)
**Status**: Open
**Affected Component**: Backend - Pydantic models (e.g., `ModelResponse`)

**Problem**:
- Logs show warnings:
  ```
  Field "model_type" in ModelResponse has conflict with protected namespace "model_".
  Field "model_family" in ModelResponse has conflict with protected namespace "model_".
  ```
- Pydantic v2 reserves fields starting with `model_` for internal use
- Our model fields (`model_type`, `model_family`, `model_ids`) trigger these warnings

**Proposed Fix**:
- Rename fields to avoid `model_` prefix:
  - `model_type` → `type` or `algorithm`
  - `model_family` → `family`
  - `model_ids` → `ids` (in compare endpoints)
- Update all Pydantic models, API responses, and frontend TypeScript interfaces
- This is a breaking change for API consumers, so requires version bump

**Estimated Effort**: 4 hours (due to widespread usage)

---

### Issue #5: LIME Global Importance Calculation May Be Simplified

**Severity**: Low (correctness concern)
**Status**: Open
**Affected Component**: Backend - `LIMEService.explain_global()`

**Problem**:
- Current implementation in `lime_service.py` (lines 420-467):
  - Runs LIME on up to 50 samples
  - Aggregates absolute weights by feature
  - **BUT**: The loop logic seems to collect ALL weights from `local_exp` but might not correctly handle multi-class or feature mapping
  - It uses `exp_data["local_exp"].get("0", exp_data["local_exp"].get("1", []))` which assumes binary classification
  - May produce incorrect global importance for multi-class LIME

**Proposed Fix**:
- Review LIME global aggregation logic
- Ensure it handles both regression and classification (binary + multi-class)
- Test with known model to validate output
- Consider using LIME's built-in `explain_global()` if available, or improve custom implementation

**Estimated Effort**: 3 hours

---

### Issue #6: SHAP Dependence Endpoint Uses Full Dataset Without Sampling

**Severity**: Medium (performance)
**Status**: Open
**Affected Component**: Backend - `GET /api/v1/explain/dependence/{model_id}`

**Problem**:
- The SHAP dependence plot endpoint (`explanations.py` lines 726-875) computes SHAP values for the **entire** uploaded background dataset
- If user uploads a large CSV (10,000 rows), SHAP computation will be extremely slow or run out of memory
- The endpoint should sample the background data to a reasonable limit (e.g., `MAX_GLOBAL_SHAP_SAMPLES`)

**Proposed Fix**:
- Before computing SHAP, sample background data if it exceeds `MAX_GLOBAL_SHAP_SAMPLES` (default 200)
- Add a warning log if sampling occurs
- Document in API that large datasets will be downsampled

**Estimated Effort**: 30 minutes

---

### Issue #7: Local SHAP/LIME Feature Names May Not Match After Aggregation

**Severity**: Low (display issue)
**Status**: Open
**Affected Component**: Backend - `compute_shap_values()` & `compute_lime_values()`

**Problem**:
- After one-hot aggregation for pipelines, the feature names returned in the explanation should reflect the **original categorical features**, not the encoded ones
- Currently, local SHAP uses `feature_names` from `_normalize_shap_local`, which may still contain preprocessed names
- Need to ensure `feature_names` in the explanation document are the **final aggregated feature names**

**Proposed Fix**:
- In `compute_shap_values()` and `compute_lime_values()`, after calling `_normalize_shap_local()` or `LIMEService.explain_instance()`, ensure the `feature_names` field matches the aggregated `local_exp` keys
- For SHAP: Use `final_feature_names` returned from `_compute_shap()` (already aggregated if pipeline)
- For LIME: Extract feature names from aggregated `local_exp` keys

**Estimated Effort**: 1 hour

---

## ✅ Resolved Issues

### Issue #8: SHAP Global Explanation Taking 14+ Minutes (Optimized)
- **Fixed**: Modified `_compute_shap()` to use TreeExplainer on pipeline final estimator + preprocessed data
- **Result**: Fast tree-based SHAP now completes in seconds/minutes
- **Date**: 2026-03-26

### Issue #9: Frontend Infinite Loading (No Polling)
- **Fixed**: Added `refetchInterval` to global explanation query to poll every 3s while pending
- **Result**: Explanation appears automatically when ready
- **Date**: 2026-03-26

### Issue #10: SHAP Global Endpoint Returns Wrong Method Data
- **Fixed**: Added `{"method": "shap"}` filter to `GET /api/v1/explain/global/{model_id}/latest`
- **Result**: SHAP and LIME tabs show correct respective data
- **Date**: 2026-03-26

---

## Priority Roadmap

### Immediate (Next Sprint)
1. ✅ Fix Issue #1: Auto-impute NaN in background data (CRITICAL for UX)
2. ✅ Fix Issue #2: Remove unused NextAuth route (unblock production builds)

### Short Term (1-2 Weeks)
3. ✅ Fix Issue #6: Sample background data in dependence endpoint
4. ✅ Fix Issue #7: Align local explanation feature names with aggregated output

### Medium Term (1 Month)
5. Investigate Issue #3: Backfill model metadata for existing models
6. Address Issue #4: Resolve Pydantic field name conflicts
7. Improve Issue #5: Validate LIME global calculation

---

## How to Report New Issues

When you discover a bug or improvement opportunity:

1. **Check this document** first to avoid duplicates
2. **Add a new section** with:
   - Clear title and severity
   - Steps to reproduce
   - Expected vs actual behavior
   - Screenshots/logs if applicable
3. Update the status as work progresses

---

**Last Updated**: 2026-03-26
**Maintained by**: Development Team

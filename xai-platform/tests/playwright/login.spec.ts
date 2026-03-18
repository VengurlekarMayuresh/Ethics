import { test, expect } from '@playwright/test';

test.describe('XAI Platform E2E Tests', () => {

  test('should login successfully', async ({ page }) => {
    await page.goto('/login');

    // Fill in credentials
    await page.fill('[data-testid="email-input"]', 'test@example.com');
    await page.fill('[data-testid="password-input"]', 'password123');
    await page.click('[data-testid="login-button"]');

    // Should redirect to dashboard
    await expect(page).toHaveURL('/');
    await expect(page.locator('text=Dashboard')).toBeVisible();
  });

  test('should upload a model', async ({ page }) => {
    await page.goto('/models');

    // Click upload button
    await page.click('[data-testid="upload-model-button"]');

    // Fill model details
    await page.fill('[data-testid="model-name"]', 'Test Model');
    await page.fill('[data-testid="model-description"]', 'Test description');

    // Upload file (mocked in test environment)
    await page.setInputFiles('[data-testid="model-file"]', 'test-model.pkl');

    // Submit
    await page.click('[data-testid="submit-upload"]');

    // Should show success
    await expect(page.locator('text=Model uploaded successfully')).toBeVisible();
  });

  test('should create a prediction', async ({ page }) => {
    await page.goto('/predict/history');

    // Click new prediction
    await page.click('[data-testid="new-prediction-button"]');

    // Select model
    await page.selectOption('[data-testid="model-select"]', 'test-model-id');

    // Fill features
    await page.fill('[data-testid="feature-age"]', '25');
    await page.fill('[data-testid="feature-income"]', '50000');

    // Submit
    await page.click('[data-testid="predict-button"]');

    // Should show result
    await expect(page.locator('[data-testid="prediction-result"]')).toBeVisible();
  });

  test('should request SHAP explanation', async ({ page }) => {
    await page.goto('/explain/local/test-model/test-prediction');

    // Request explanation
    await page.click('[data-testid="request-shap-button"]');

    // Should show task ID
    await expect(page.locator('[data-testid="task-id"]')).toBeVisible();

    // Poll for completion (or wait for WebSocket)
    await expect(page.locator('[data-testid="explanation-complete"]')).toBeVisible({ timeout: 30000 });
  });

});

# XAI Platform E2E Tests

Uses Playwright for end-to-end testing.

## Setup

```bash
npm install -D @playwright/test
npx playwright install
```

## Running Tests

```bash
# Start the app in one terminal
npm start

# Run tests in another
npx playwright test
```

## CI Integration

Add to GitHub Actions:

```yaml
- name: Run E2E tests
  run: |
    npm start &
    npx wait-on http://localhost:3000
    npx playwright test --reporter=html
```

# Load Testing

Uses Locust for load testing the API.

## Setup

```bash
pip install locust
```

## Running Tests

```bash
# Start the backend (assumes running on localhost:8000)
cd backend
uvicorn app.main:app --reload

# Run load tests in another terminal
cd tests/locust
locust -f locustfile.py --host=http://localhost:8000
```

Then open http://localhost:8089 to configure and run load tests.

## Simulating Users

- Logged-in users: tasks for models, predictions, API calls
- Anonymous users: health checks, public pages

Adjust number of users and spawn rate in the web UI.

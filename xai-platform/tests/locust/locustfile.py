"""
Load testing for XAI Platform API.
"""

from locust import HttpUser, task, between
import random
import json


class XAIPlatformUser(HttpUser):
    wait_time = between(1, 5)

    def on_start(self):
        """Login and get JWT token."""
        response = self.client.post("/api/v1/auth/login", json={
            "username": "test@example.com",
            "password": "testpassword"
        })
        if response.status_code == 200:
            self.token = response.json()["access_token"]
            self.headers = {"Authorization": f"Bearer {self.token}"}
        else:
            self.token = None
            self.headers = {}

    @task(4)
    def list_models(self):
        """List user's models."""
        if self.token:
            self.client.get("/api/v1/models", headers=self.headers)

    @task(3)
    def get_model_detail(self):
        """Get specific model details."""
        if self.token:
            # In real scenario, would use actual model ID from user's list
            model_id = "64d5f8a7b1c2e3f456789abc"  # Mock ID
            self.client.get(f"/api/v1/models/{model_id}", headers=self.headers)

    @task(2)
    def predict(self):
        """Make a prediction."""
        if self.token:
            payload = {
                "input_data": {
                    "feature1": random.uniform(0, 10),
                    "feature2": random.uniform(0, 100),
                }
            }
            self.client.post(
                "/api/v1/predict/mock-model-id",
                json=payload,
                headers=self.headers
            )

    @task(1)
    def health_check(self):
        """Check API health."""
        self.client.get("/health")

    @task(1)
    def get_prediction_history(self):
        """Get prediction history."""
        if self.token:
            self.client.get("/api/v1/predict/history", headers=self.headers)

    @task(1)
    def list_api_keys(self):
        """List API keys."""
        if self.token:
            self.client.get("/api/v1/api-keys/", headers=self.headers)


class AnonymousUser(HttpUser):
    wait_time = between(2, 10)

    @task(5)
    def health_check(self):
        self.client.get("/health")

    @task(3)
    def login_page(self):
        self.client.get("/login")

    @task(1)
    def register_page(self):
        self.client.get("/register")

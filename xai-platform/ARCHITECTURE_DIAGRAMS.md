# 🗺️ XAI Platform: Technical Architecture Diagrams

These diagrams explain the "How it Works" visually. You can describe these during your viva to show you understand the full stack.

---

## 1. System Component Hierarchy
This diagram shows how the platform handles high-performance tasks and storage.

```mermaid
graph TD
    User((User/Examiner))
    Web[Next.js Frontend]
    API[FastAPI Backend]
    Worker[Celery Worker]
    Redis[(Redis Queue)]
    Mongo[(MongoDB - Metadata)]
    MinIO[(MinIO - Model Store)]

    User <--> Web
    Web <--> API
    API <--> Mongo
    API <--> MinIO
    API <--> Redis
    Redis <--> Worker
    Worker <--> MinIO
    Worker --> Mongo
```

---

## 2. The Model Life Cycle (Training to Upload)
Explain this flow to show how a script becomes an active model on the dashboard.

```mermaid
sequenceDiagram
    participant S as Training Script (Python)
    participant A as XAI API
    participant S3 as MinIO (Object Storage)
    participant M as MongoDB

    S->>S: pipeline = Pipeline(steps...)
    S->>S: joblib.dump(pipeline, 'model.pkl')
    S->>A: POST /models/upload
    A->>A: ModelLoader: Extract Feature Schema
    A->>S3: Stream model.pkl to S3
    A->>M: Create Model Meta (Schema, Task Type)
    A-->>S: Success (Return Model ID)
```

---

## 3. The Prediction & Explanation Request Flow
This is the most critical flow to explain during a Viva.

```mermaid
sequenceDiagram
    participant U as User (Next.js Dashboard)
    participant B as FastAPI Backend
    participant W as Celery Worker (XAI Engine)
    participant S3 as MinIO (S3)

    U->>B: POST /predict (Raw JSON)
    B->>S3: Fetch model.pkl
    B->>B: joblib.load(model)
    B->>B: pipeline.predict(data)
    B-->>U: Return JSON Prediction
    
    U->>B: POST /explain (Request SHAP)
    B->>W: Push Task to Redis
    W->>S3: Load model.pkl
    W->>W: Compute Shapley Values
    W-->>B: Explanation JSON
    B-->>U: Render D3.js Graphs
```

---

## 4. Feature Space vs. Raw Space
Explain how the platform bridges the gap between raw user input and model tensors.

```mermaid
flowchart LR
    Raw[Raw Input: "Married: Yes"] --> PE[Pipeline: FeatureEngineer]
    PE --> Enc[One-Hot Encoding: [1, 0]]
    Enc --> Scale[StandardScaler: 0.85]
    Scale --> Pred[XGBoost Prediction]
    Pred --> Inverse[XAI Engine: Aggregated Logic]
    Inverse --> UI[UI Result: "Marriage Impact"]
```

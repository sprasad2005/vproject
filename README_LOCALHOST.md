# RiceGuard — Localhost Application Guide

This guide provides instructions for launching and testing the **RiceGuard** local web application.

> [!NOTE]
> **Strict Localhost Deployment**: This application runs entirely on your local machine (`localhost`). It does not require any cloud service or external API.

---

## 1. System Requirements

- **Python**: 3.10+ (Current: Python 3.11.9)
- **PyTorch**: 2.0+ with CUDA support (Current: PyTorch 2.6.0+cu124)
- **Node.js & npm**: Node v18+ (Current: Node v24.11.1, npm 11.6.2)
- **Target Hardware**: NVIDIA GPU (e.g. GeForce GTX 1650 4GB) or CPU fallback

---

## 2. Quick Start (Two Terminal Windows)

### Terminal 1: Launch FastAPI Backend Server

```powershell
# Navigate to project root
cd D:\vproj

# Activate virtual environment
.venv\Scripts\activate

# Start backend on localhost:8000
python -m uvicorn app.backend.main:app --host 127.0.0.1 --port 8000
```

- **Health Check**: [http://127.0.0.1:8000/api/health](http://127.0.0.1:8000/api/health)
- **Interactive Swagger Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

### Terminal 2: Launch React Frontend Dashboard

```powershell
# Navigate to frontend directory
cd D:\vproj\app\frontend

# Install dependencies (if not already installed)
npm install

# Start Vite development server
npm run dev
```

Open your browser and navigate to:
👉 **[http://127.0.0.1:5173](http://127.0.0.1:5173)**

---

## 3. Application Features & Architecture

```
                    ┌─────────────────────────┐
                    │      React Frontend     │
                    │   http://127.0.0.1:5173 │
                    └────────────┬────────────┘
                                 │ HTTP API (multipart/json)
                                 ▼
                    ┌─────────────────────────┐
                    │     FastAPI Backend     │
                    │   http://127.0.0.1:8000 │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ RiceGuard Engine (CUDA) │
                    └────────────┬────────────┘
                                 │
                 ┌───────────────┴───────────────┐
                 ▼                               ▼
       Phase 3B Multi-Task Model          Grad-CAM Explainer
        (Frozen Conf ≥ 0.60, Top-K 3)      (Backbone Features)
                 │                               │
                 └───────────────┬───────────────┘
                                 ▼
                     Structured Predictions
                     • Disease Diagnosis
                     • Probabilities Distribution
                     • Lesion Bounding Boxes
                     • Grad-CAM Heatmap Overlay
                     • Combined Multi-Modal View
```

---

## 4. API Endpoints Reference

| Method | Endpoint | Description | Sample Output |
|---|---|---|---|
| `GET` | `/api/health` | System health & active device | `{"status": "healthy", "device": "cuda:0", "model": "RiceGuard Phase 3B"}` |
| `GET` | `/api/samples` | List of quick-test sample keys | `[{"id": "blast", "name": "Blast Leaf Sample"}, ...]` |
| `GET` | `/api/samples/{id}` | Serve sample image bytes | Binary JPEG image file |
| `POST` | `/api/predict` | Multi-task analysis endpoint | Full structured prediction JSON + 4 Base64 visualizations |

---

## 5. Automated Tests

To run the application test suite:

```powershell
# Run backend application tests
pytest tests/test_phase6_app.py -v

# Run full project test suite (Phases 1 through 6)
pytest
```

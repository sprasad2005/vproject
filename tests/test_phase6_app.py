"""Unit and integration tests for RiceGuard Phase 6 Localhost Application."""

from __future__ import annotations

import io

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from app.backend.config import settings
from app.backend.main import app
from app.backend.services.inference_service import get_inference_engine
from app.backend.services.localization_service import LocalizationService
from app.backend.utils.image_processing import validate_and_load_image


@pytest.fixture
def client() -> TestClient:
    """FastAPI TestClient fixture."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def sample_leaf_image_bytes() -> bytes:
    """Generate valid synthetic RGB leaf-like image bytes."""
    arr = np.random.randint(50, 200, (256, 256, 3), dtype=np.uint8)
    # Add green-ish tint
    arr[:, :, 1] = np.clip(arr[:, :, 1] + 40, 0, 255)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


class TestPhase6ApplicationBackend:
    """Backend endpoints and logic testing."""

    def test_health_endpoint(self, client: TestClient) -> None:
        """Verify GET /api/health endpoint response structure and values."""
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "device" in data
        assert data["device"] in ["cuda:0", "cpu"]
        assert data["model"] == "RiceGuard Phase 3B"

    def test_frozen_decoding_config_loaded(self) -> None:
        """Verify frozen Phase 3B decoding configuration values."""
        assert settings.confidence_threshold == 0.60
        assert settings.top_k == 3
        assert len(settings.CANONICAL_CLASSES) == 6
        assert settings.CANONICAL_CLASSES[0] == "Healthy"
        assert settings.CANONICAL_CLASSES[1] == "Blast"

    def test_model_loading_and_eval_mode(self) -> None:
        """Verify inference engine loads Phase 3B checkpoint in eval mode."""
        engine = get_inference_engine()
        assert engine is not None
        assert not engine.model.training
        assert isinstance(engine.localization_service, LocalizationService)
        assert engine.localization_service.confidence_threshold == 0.60
        assert engine.localization_service.top_k == 3

    def test_predict_endpoint_valid_image(
        self,
        client: TestClient,
        sample_leaf_image_bytes: bytes,
    ) -> None:
        """Verify POST /api/predict returns valid structured JSON and 4 visualizations."""
        files = {"image": ("test_leaf.jpg", sample_leaf_image_bytes, "image/jpeg")}
        response = client.post("/api/predict", files=files)
        assert response.status_code == 200
        data = response.json()

        # Prediction details
        assert "prediction" in data
        assert 0 <= data["prediction"]["class_id"] <= 5
        assert data["prediction"]["class_name"] in settings.CANONICAL_CLASSES
        assert 0.0 <= data["prediction"]["confidence"] <= 1.0

        # Top predictions
        assert "top_predictions" in data
        assert len(data["top_predictions"]) == 6
        prob_sum = sum(p["probability"] for p in data["top_predictions"])
        assert 0.98 <= prob_sum <= 1.02

        # Lesions
        assert "lesions" in data
        assert "lesion_count" in data
        assert data["lesion_count"] == len(data["lesions"])
        assert data["lesion_count"] <= settings.top_k

        # Visualizations
        assert "visualizations" in data
        for key in ["original", "localization", "gradcam", "combined"]:
            assert key in data["visualizations"]
            assert data["visualizations"][key].startswith("data:image/")

        # Model Info
        assert "model" in data
        assert data["model"]["name"] == "RiceGuard Phase 3B"

    def test_predict_endpoint_invalid_image_bytes(self, client: TestClient) -> None:
        """Verify invalid image payload returns 400 Bad Request."""
        bad_bytes = b"This is not a real image binary file."
        files = {"image": ("corrupt.jpg", bad_bytes, "image/jpeg")}
        response = client.post("/api/predict", files=files)
        assert response.status_code == 400
        assert "Invalid or corrupted image file" in response.json()["detail"]

    def test_predict_endpoint_empty_file(self, client: TestClient) -> None:
        """Verify empty file upload returns 400 Bad Request."""
        files = {"image": ("empty.jpg", b"", "image/jpeg")}
        response = client.post("/api/predict", files=files)
        assert response.status_code == 400

    def test_bounding_box_clipping(self) -> None:
        """Verify localization boxes are safely clipped within image boundaries."""
        loc_service = LocalizationService(confidence_threshold=0.10, top_k=5)
        # Create synthetic localization tensor with values near borders
        loc_tensor = torch.zeros((1, 5, 7, 7))
        loc_tensor[0, 0, 0, 0] = 5.0  # high objectness at top-left
        loc_tensor[0, 1, 0, 0] = -5.0  # dx
        loc_tensor[0, 2, 0, 0] = -5.0  # dy
        loc_tensor[0, 3, 0, 0] = 5.0  # large width
        loc_tensor[0, 4, 0, 0] = 5.0  # large height

        boxes = loc_service.decode_lesions(loc_tensor, orig_width=400, orig_height=300)
        assert len(boxes) > 0
        for box in boxes:
            assert 0 <= box.x < 400
            assert 0 <= box.y < 300
            assert box.width <= 400
            assert box.height <= 300
            assert box.x + box.width <= 400
            assert box.y + box.height <= 300

    def test_image_processing_validation(self) -> None:
        """Test image validation utility."""
        valid_img = Image.new("RGB", (100, 100), color="green")
        buf = io.BytesIO()
        valid_img.save(buf, format="PNG")

        img, w, h = validate_and_load_image(buf.getvalue())
        assert w == 100
        assert h == 100
        assert img.mode == "RGB"

        with pytest.raises(ValueError, match="empty"):
            validate_and_load_image(b"")

"""FastAPI Backend Server for RiceGuard Localhost Application."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from app.backend.config import settings
from app.backend.schemas import HealthResponse, PredictionResponse
from app.backend.services.inference_service import get_inference_engine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("riceguard.api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager to load models and warmup services on server start."""
    logger.info("Starting RiceGuard Localhost API server...")
    # Pre-load model and warm up inference engine
    try:
        engine = get_inference_engine()
        logger.info("RiceGuard Engine ready on device: %s", engine.device_str)
    except Exception as e:
        logger.error("Failed to initialize RiceGuard Engine on startup: %s", e)
        raise RuntimeError(f"Engine initialization error: {e}") from e

    yield

    logger.info("Shutting down RiceGuard Localhost API server...")


app = FastAPI(
    title="RiceGuard Localhost API",
    description="Localhost-only API for Lesion-Grounded Rice Leaf Disease Classification and Explainability.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS Middleware (strictly localhost origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount figures static assets
figures_dir = settings.root / "results" / "figures"
if figures_dir.exists():
    app.mount("/api/static/figures", StaticFiles(directory=str(figures_dir)), name="figures")


@app.get(
    "/api/health",
    response_model=HealthResponse,
    summary="Health Check",
    tags=["System"],
)
async def health_check() -> HealthResponse:
    """Return system operational status and active compute device."""
    return HealthResponse(
        status="healthy",
        device=settings.device,
        model=settings.MODEL_NAME,
    )


SAMPLE_IMAGES = {
    "blast": "RiceLeafDiseaseBD A Field-Based Annotated Smartpho/RiceLeafDiseaseBD A Field-Based Annotated Smartpho/RiceLeafDiseaseBD/Original images/Blast/blast_1002.jpg",
    "brown_spot": "RiceLeafDiseaseBD A Field-Based Annotated Smartpho/RiceLeafDiseaseBD A Field-Based Annotated Smartpho/RiceLeafDiseaseBD/Original images/Brown spot/brown spot_1003.jpg",
    "healthy": "RiceLeafDiseaseBD A Field-Based Annotated Smartpho/RiceLeafDiseaseBD A Field-Based Annotated Smartpho/RiceLeafDiseaseBD/Original images/Healthy/Healthy_10.jpg",
    "tungro": "RiceLeafDiseaseBD A Field-Based Annotated Smartpho/RiceLeafDiseaseBD A Field-Based Annotated Smartpho/RiceLeafDiseaseBD/Original images/Rice Tungro/Rice Tungro_1004.jpg",
}


@app.get(
    "/api/samples",
    summary="Get List of Available Test Samples",
    tags=["Samples"],
)
async def get_sample_list():
    """Return list of available quick-test sample disease keys."""
    return [
        {"id": "blast", "name": "Blast Leaf Sample", "disease": "Blast"},
        {"id": "brown_spot", "name": "Brown Spot Sample", "disease": "Brown Spot"},
        {"id": "tungro", "name": "Tungro Sample", "disease": "Tungro"},
        {"id": "healthy", "name": "Healthy Leaf Sample", "disease": "Healthy"},
    ]


@app.get(
    "/api/samples/{sample_id}",
    summary="Get Sample Image File",
    tags=["Samples"],
)
async def get_sample_image(sample_id: str):
    """Serve sample image file bytes for quick frontend testing."""
    if sample_id not in SAMPLE_IMAGES:
        raise HTTPException(status_code=404, detail="Sample not found.")
    rel_p = SAMPLE_IMAGES[sample_id]
    full_p = settings.root / rel_p
    if not full_p.exists():
        full_p = settings.root / "data" / "raw" / rel_p
    if not full_p.exists():
        raise HTTPException(status_code=404, detail="Sample image file not found on disk.")
    return FileResponse(full_p, media_type="image/jpeg")


@app.get(
    "/api/ieee/data",
    summary="Get Consolidated IEEE Publication Research Artifacts & Tables",
    tags=["IEEE Publication"],
)
async def get_ieee_publication_data():
    """Return structured master tables, statistical significance tests, and figure manifests for IEEE paper requirements."""
    return {
        "title": "RiceGuard: Lesion-Grounded Rice Leaf Disease Classification and Explainability",
        "selected_model": {
            "name": "Phase 3B EfficientNet-B0 Multi-Task Refined",
            "supervision": "Weakly Supervised Bounding Box + Cross-Entropy Classification",
            "test_accuracy": "88.96%",
            "test_macro_f1": "0.8751",
            "localization_f1": "0.0905",
            "mean_matched_iou": "0.6423",
            "healthy_fp_rate": "10.97%",
            "parameters": "6,966,151 (26.86 MB)",
            "gpu_latency": "10.46 ms",
            "status": "Final Selected Deployment Architecture"
        },
        "tables": {
            "table_1_backbone_screening": [
                {
                    "model": "EfficientNet-B0 (Selected)",
                    "val_f1": "0.8653",
                    "test_acc": "88.48%",
                    "test_f1": "0.8727",
                    "params": "4,015,234",
                    "model_size": "15.61 MB",
                    "gpu_latency": "10.36 ms",
                    "edge_feasibility": "Optimal"
                },
                {
                    "model": "ResNet-50",
                    "val_f1": "0.8642",
                    "test_acc": "88.14%",
                    "test_f1": "0.8651",
                    "params": "23,520,326",
                    "model_size": "90.25 MB",
                    "gpu_latency": "8.97 ms",
                    "edge_feasibility": "High Overhead"
                },
                {
                    "model": "Vision Transformer (ViT-B/16)",
                    "val_f1": "0.8512",
                    "test_acc": "86.92%",
                    "test_f1": "0.8540",
                    "params": "85,800,000",
                    "model_size": "328.40 MB",
                    "gpu_latency": "24.12 ms",
                    "edge_feasibility": "Impractical for Edge"
                }
            ],
            "table_2_master_evolution": [
                {
                    "stage": "Phase 2B Baseline",
                    "supervision": "Class Labels Only",
                    "accuracy": "90.12%",
                    "macro_f1": "0.8891",
                    "loc_precision": "N/A",
                    "loc_recall": "N/A",
                    "loc_f1": "N/A",
                    "mean_iou": "N/A",
                    "healthy_fp_rate": "N/A",
                    "border_attn": "18.5%",
                    "entropy": "0.824",
                    "params": "4.02M",
                    "gpu_latency": "10.36 ms"
                },
                {
                    "stage": "Phase 3 Multi-Task (Initial)",
                    "supervision": "Class + Raw BBoxes",
                    "accuracy": "88.75%",
                    "macro_f1": "0.8729",
                    "loc_precision": "0.0223",
                    "loc_recall": "0.0914",
                    "loc_f1": "0.0358",
                    "mean_iou": "0.6032",
                    "healthy_fp_rate": "21.52%",
                    "border_attn": "N/A",
                    "entropy": "N/A",
                    "params": "6.97M",
                    "gpu_latency": "13.52 ms"
                },
                {
                    "stage": "Phase 3B Multi-Task Refined (Selected)",
                    "supervision": "Class + Calibrated BBoxes",
                    "accuracy": "88.96%",
                    "macro_f1": "0.8751",
                    "loc_precision": "0.0887",
                    "loc_recall": "0.0924",
                    "loc_f1": "0.0905",
                    "mean_iou": "0.6423",
                    "healthy_fp_rate": "10.97%",
                    "border_attn": "9.2%",
                    "entropy": "0.612",
                    "params": "6.97M",
                    "gpu_latency": "10.46 ms"
                }
            ],
            "table_3_per_class": [
                {"class_name": "Healthy", "support": 237, "p2b_p": "0.9262", "p2b_r": "0.9536", "p2b_f1": "0.9397", "p3b_p": "0.9314", "p3b_r": "0.9536", "p3b_f1": "0.9423", "delta_f1": "+0.0026"},
                {"class_name": "Blast", "support": 199, "p2b_p": "0.8565", "p2b_r": "0.8995", "p2b_f1": "0.8775", "p3b_p": "0.8557", "p3b_r": "0.8693", "p3b_f1": "0.8625", "delta_f1": "-0.0150"},
                {"class_name": "Brown Spot", "support": 327, "p2b_p": "0.9215", "p2b_r": "0.8257", "p2b_f1": "0.8710", "p3b_p": "0.9102", "p3b_r": "0.8349", "p3b_f1": "0.8709", "delta_f1": "-0.0001"},
                {"class_name": "Leaf Smut", "support": 109, "p2b_p": "0.7565", "p2b_r": "0.7982", "p2b_f1": "0.7768", "p3b_p": "0.7431", "p3b_r": "0.7431", "p3b_f1": "0.7431", "delta_f1": "-0.0337"},
                {"class_name": "Tungro", "support": 337, "p2b_p": "0.8939", "p2b_r": "0.9496", "p2b_f1": "0.9209", "p3b_p": "0.8808", "p3b_r": "0.9496", "p3b_f1": "0.9139", "delta_f1": "-0.0070"},
                {"class_name": "Sheath Blight", "support": 258, "p2b_p": "0.9677", "p2b_r": "0.9302", "p2b_f1": "0.9486", "p3b_p": "0.9592", "p3b_r": "0.9109", "p3b_f1": "0.9344", "delta_f1": "-0.0142"}
            ],
            "table_4_xai_grounding": [
                {"subset": "All Eligible Disease Samples (N=4,348)", "p2b_energy": "9.64%", "p3b_energy": "6.81%", "delta_energy": "-2.83%", "p2b_iou": "0.0864", "p3b_iou": "0.0724", "delta_iou": "-0.0140", "p2b_pointing": "29.58%", "p3b_pointing": "21.37%", "delta_pointing": "-8.21%"},
                {"subset": "Mutually Correct Classified (N=280)", "p2b_energy": "9.73%", "p3b_energy": "8.13%", "delta_energy": "-1.60%", "p2b_iou": "0.0921", "p3b_iou": "0.0864", "delta_iou": "-0.0057", "p2b_pointing": "36.79%", "p3b_pointing": "38.21%", "delta_pointing": "+1.43%"},
                {"subset": "Phase 3B Correct Only (N=468)", "p2b_energy": "10.25%", "p3b_energy": "8.96%", "delta_energy": "-1.29%", "p2b_iou": "0.0954", "p3b_iou": "0.0924", "delta_iou": "-0.0030", "p2b_pointing": "32.69%", "p3b_pointing": "36.32%", "delta_pointing": "+3.63%"},
                {"subset": "Phase 2B Correct Only (N=1,060)", "p2b_energy": "11.16%", "p3b_energy": "6.31%", "delta_energy": "-4.84%", "p2b_iou": "0.1033", "p3b_iou": "0.0691", "delta_iou": "-0.0341", "p2b_pointing": "31.51%", "p3b_pointing": "18.49%", "delta_pointing": "-13.02%"}
            ],
            "table_5_statistical_tests": [
                {"metric": "Energy Inside Mask (All Samples)", "test": "Paired Wilcoxon Signed-Rank", "statistic": "1,740,917.0", "p_value": "5.12e-285", "significance": "p < 0.001 (Significant)"},
                {"metric": "Attribution IoU (All Samples)", "test": "Paired Wilcoxon Signed-Rank", "statistic": "1,955,501.0", "p_value": "9.59e-95", "significance": "p < 0.001 (Significant)"},
                {"metric": "Pointing Game Hit Rate (All Samples)", "test": "McNemar Chi-Square", "statistic": "131.06", "p_value": "< 1.0e-20", "significance": "p < 0.001 (Significant)"},
                {"metric": "Energy Inside Mask (Mutually Correct)", "test": "Paired Wilcoxon Signed-Rank", "statistic": "2,724.0", "p_value": "7.88e-36", "significance": "p < 0.001 (Significant)"},
                {"metric": "Pointing Game Hit Rate (Mutually Correct)", "test": "McNemar Chi-Square", "statistic": "0.1552", "p_value": "0.694", "significance": "p > 0.05 (Directional Gain)"},
                {"metric": "Pointing Game Hit Rate (Phase 3B Correct)", "test": "McNemar Chi-Square", "statistic": "1.9845", "p_value": "0.159", "significance": "p > 0.05 (Directional Gain)"}
            ],
            "table_6_faithfulness": [
                {"metric": "Deletion AUC (Lower is Better)", "p2b": "0.1387 ± 0.2086", "p3b": "0.1367 ± 0.1482", "delta": "-0.0021 (Faster Drop)", "p_value": "0.187"},
                {"metric": "Insertion AUC (Higher is Better)", "p2b": "0.2939 ± 0.2979", "p3b": "0.1486 ± 0.1682", "delta": "-0.1453", "p_value": "3.58e-09"}
            ]
        },
        "figures": [
            {
                "id": "fig1",
                "number": "Fig. 1",
                "title": "RiceGuard Experimental Architecture Evolution",
                "caption": "Three-stage architectural progression from Phase 2B (classification-only baseline) to Phase 3 (initial lesion-aware multi-task) and Phase 3B (calibrated lesion localization with positive weight capping and top-k decoding).",
                "url": "/api/static/figures/phase5/model_evolution_overview.png",
                "category": "Architecture"
            },
            {
                "id": "fig2",
                "number": "Fig. 2",
                "title": "Classification vs Localization vs Efficiency Trade-off",
                "caption": "Four-panel comparison showing classification Macro F1 (A), localization precision/recall/F1 (B), mean matched IoU (C), and GPU latency vs parameters (D) across all experimental phases.",
                "url": "/api/static/figures/phase5/classification_localization_tradeoff.png",
                "category": "Benchmark"
            },
            {
                "id": "fig3",
                "number": "Fig. 3",
                "title": "Per-Class F1 Score Delta Comparison",
                "caption": "Per-class harmonic F1 scores comparing Phase 2B baseline against Phase 3B calibrated multi-task model across all 6 canonical disease pathologies on the internal test split.",
                "url": "/api/static/figures/phase5/per_class_f1_comparison.png",
                "category": "Classification"
            },
            {
                "id": "fig4",
                "number": "Fig. 4",
                "title": "Normalized Test Confusion Matrix Comparison (N=1,467)",
                "caption": "Side-by-side normalized test confusion matrices for Phase 2B (left) and Phase 3B (right) in canonical class order (Healthy, Blast, Brown Spot, Leaf Smut, Tungro, Sheath Blight).",
                "url": "/api/static/figures/phase5/confusion_matrix_comparison.png",
                "category": "Classification"
            },
            {
                "id": "fig5",
                "number": "Fig. 5",
                "title": "Quantitative XAI Spatial Grounding & Shortcut Diagnostic Summary",
                "caption": "Four-panel quantitative explainability benchmark on RiceSeg5932 independent masks and healthy background shortcut analysis showing 50.3% outer border attention reduction and entropy concentration.",
                "url": "/api/static/figures/phase5/xai_grounding_summary.png",
                "category": "Explainability"
            },
            {
                "id": "fig6",
                "number": "Fig. 6",
                "title": "Deterministic Multi-Category Failure-Mode Analysis Grid",
                "caption": "Representative failure-mode matrix comparing Phase 2B and Phase 3B across Baseline Wins, Multi-Task Wins, Mutual Errors, Healthy False Positives, and Lesion Misses.",
                "url": "/api/static/figures/phase5/failure_cases_baseline_vs_multitask.png",
                "category": "Diagnostics"
            },
            {
                "id": "fig7a",
                "number": "Fig. 7A",
                "title": "Pixel Deletion Faithfulness Curves",
                "caption": "Stepwise salient pixel deletion curves showing confidence degradation trajectories over 20 perturbation steps with Gaussian blur baseline.",
                "url": "/api/static/figures/phase4/deletion_curves.png",
                "category": "Faithfulness"
            },
            {
                "id": "fig7b",
                "number": "Fig. 7B",
                "title": "Pixel Insertion Faithfulness Curves",
                "caption": "Stepwise salient pixel insertion curves showing confidence recovery trajectories starting from a blurred background.",
                "url": "/api/static/figures/phase4/insertion_curves.png",
                "category": "Faithfulness"
            },
            {
                "id": "fig8",
                "number": "Fig. 8",
                "title": "Multi-Sample Qualitative Grad-CAM Grounding Panel",
                "caption": "Comprehensive 24-sample qualitative visual panel comparing Phase 2B vs Phase 3B Grad-CAM heatmaps against independent ground-truth pixel segmentation masks on RiceSeg5932.",
                "url": "/api/static/figures/phase4/phase2b_vs_phase3b_xai_examples.png",
                "category": "Explainability"
            }
        ]
    }



@app.post(
    "/api/predict",
    response_model=PredictionResponse,
    summary="Analyze Rice Leaf Image",
    tags=["Inference"],
)
async def predict_image(
    image: UploadFile = File(..., description="Uploaded rice leaf image file (JPEG/PNG)"),
) -> PredictionResponse:
    """Analyze uploaded rice leaf image for disease diagnosis, lesion localization, and Grad-CAM explainability."""
    # 1. Basic Content Type / Filename Validation
    if not image.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No filename provided in upload.",
        )

    # 2. Read image bytes
    try:
        contents = await image.read()
    except Exception as e:
        logger.error("Failed to read uploaded file: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not read uploaded image file.",
        ) from e

    # 3. Execute Inference
    engine = get_inference_engine()
    try:
        result = engine.predict(image_bytes=contents)
        return result
    except ValueError as val_err:
        logger.warning("Image validation error: %s", val_err)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err),
        ) from val_err
    except Exception as err:
        logger.error("Unexpected inference error: %s", err, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to analyze this image. Please try another valid rice leaf image.",
        ) from err


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error("Unhandled exception on %s: %s", request.url, exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred during analysis."},
    )

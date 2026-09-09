"""Calibration module for RiceGuard."""

from src.calibration.temperature_scaling import ModelWithTemperature, compute_calibration_error

__all__ = [
    "ModelWithTemperature",
    "compute_calibration_error",
]

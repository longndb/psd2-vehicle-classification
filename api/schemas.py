"""Pydantic schemas for API request/response models."""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class ProbabilityDistribution(BaseModel):
    """Probability distribution across all classes."""
    bike: float = Field(..., ge=0, le=1, description="Probability of bike")
    car: float = Field(..., ge=0, le=1, description="Probability of car")


class SegmentPrediction(BaseModel):
    """Prediction for a single time segment."""
    segment_id: int = Field(..., description="Segment index (0-based)")
    time_start: float = Field(..., description="Segment start time in seconds")
    time_end: float = Field(..., description="Segment end time in seconds")
    label: str = Field(..., description="Predicted transportation mode")
    label_id: int = Field(..., ge=0, le=1, description="Label ID (0=bike, 1=car)")
    probabilities: ProbabilityDistribution = Field(..., description="Class probabilities")


class OverallPrediction(BaseModel):
    """Overall prediction based on majority voting."""
    label: str = Field(..., description="Most common transportation mode")
    label_id: int = Field(..., ge=0, le=1, description="Label ID of most common mode")
    confidence: float = Field(..., ge=0, le=1, description="Percentage of segments with this label")


class ModeCount(BaseModel):
    """Count and percentage for a transportation mode."""
    count: int = Field(..., ge=0, description="Number of segments")
    percentage: float = Field(..., ge=0, le=100, description="Percentage of total segments")


class ModeDistribution(BaseModel):
    """Distribution of predictions across all modes."""
    bike: ModeCount = Field(..., description="Bike statistics")
    car: ModeCount = Field(..., description="Car statistics")


class PredictionResponse(BaseModel):
    """Response model for successful prediction."""
    status: str = Field(default="success", description="Response status")
    window_seconds: float = Field(..., description="Window size in seconds used for segmentation")
    total_segments: int = Field(..., ge=1, description="Total number of time segments analyzed")
    overall_prediction: OverallPrediction = Field(..., description="Majority vote prediction")
    segment_predictions: List[SegmentPrediction] = Field(..., description="Per-segment predictions")
    mode_distribution: ModeDistribution = Field(..., description="Distribution across modes")

    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "success",
                "window_seconds": 2.0,
                "total_segments": 75,
                "overall_prediction": {
                    "label": "car",
                    "label_id": 1,
                    "confidence": 0.85
                },
                "segment_predictions": [
                    {
                        "segment_id": 0,
                        "time_start": 0.0,
                        "time_end": 2.0,
                        "label": "bike",
                        "label_id": 0,
                        "probabilities": {"bike": 0.90, "car": 0.10}
                    },
                    {
                        "segment_id": 1,
                        "time_start": 2.0,
                        "time_end": 4.0,
                        "label": "car",
                        "label_id": 1,
                        "probabilities": {"bike": 0.20, "car": 0.80}
                    }
                ],
                "mode_distribution": {
                    "bike": {"count": 5, "percentage": 6.67},
                    "car": {"count": 70, "percentage": 93.33}
                }
            }
        }
    }


class ErrorResponse(BaseModel):
    """Response model for errors."""
    status: str = Field(default="error", description="Response status")
    message: str = Field(..., description="Error message")
    details: Optional[str] = Field(None, description="Additional error details")

    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "error",
                "message": "Invalid CSV format",
                "details": "Missing required column: ax"
            }
        }
    }


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(default="healthy", description="Service status")
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    version: str = Field(..., description="API version")
    window_seconds: float = Field(..., description="Configured window size in seconds")


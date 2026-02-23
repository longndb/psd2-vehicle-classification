"""Model loading and prediction logic."""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter

import torch
import pandas as pd
import numpy as np

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.nn.model import ConvolutionalNeuralNetwork
from src.utils.data import WINDOW_SIZE, SENSOR_RATE

# Configure logger
logger = logging.getLogger("vehicle-api.predictor")

# Label mapping
LABEL_MAP = {0: "bike", 1: "car"}
LABEL_TO_ID = {"bike": 0, "car": 1}
REQUIRED_COLUMNS = ["time", "ax", "ay", "az"]

# Default model path
DEFAULT_MODEL_PATH = PROJECT_ROOT / "results" / "convolutional_neural_network" / "averaged_model" / "best_model.pth"

# Window configuration from environment variable
# WINDOW_SECONDS: Duration of each analysis window in seconds (default: 2.0)
DEFAULT_WINDOW_SECONDS = 2.0
WINDOW_SECONDS = float(os.environ.get("WINDOW_SECONDS", DEFAULT_WINDOW_SECONDS))


def process_data_with_window(
    df: pd.DataFrame,
    window_seconds: float = WINDOW_SECONDS,
) -> torch.Tensor:
    """Process CSV data with configurable window size.
    
    Args:
        df: DataFrame with time, ax, ay, az columns
        window_seconds: Duration of each analysis window in seconds
        
    Returns:
        Tensor of shape (num_segments, 3, WINDOW_SIZE)
    """
    logger.info(f"Processing data with window_seconds={window_seconds}")
    
    time = df['time'].to_numpy()
    ax = df['ax'].to_numpy()
    ay = df['ay'].to_numpy()
    az = df['az'].to_numpy()
    
    # Determine time range
    start_time = time.min()
    end_time = time.max()
    total_duration = end_time - start_time
    logger.info(f"Data time range: {start_time:.2f}s to {end_time:.2f}s (duration: {total_duration:.2f}s)")
    
    # Calculate number of complete windows
    num_segments = int(total_duration / window_seconds)
    logger.info(f"Dividing into {num_segments} segments of {window_seconds}s each")
    
    if num_segments < 1:
        raise ValueError(
            f"Data duration ({total_duration:.1f}s) is less than window size ({window_seconds}s)"
        )
    
    # Process each segment
    segments = []
    for i in range(num_segments):
        segment_start = start_time + i * window_seconds
        segment_end = segment_start + window_seconds
        
        # Create evaluation points for this segment (WINDOW_SIZE samples = 50)
        eval_time = np.linspace(segment_start, segment_end, WINDOW_SIZE)
        
        # Interpolate data to these points
        ax_interp = np.interp(eval_time, time, ax)
        ay_interp = np.interp(eval_time, time, ay)
        az_interp = np.interp(eval_time, time, az)
        
        # Stack into tensor (3, WINDOW_SIZE)
        segment_tensor = torch.stack([
            torch.tensor(ax_interp, dtype=torch.float32),
            torch.tensor(ay_interp, dtype=torch.float32),
            torch.tensor(az_interp, dtype=torch.float32),
        ], dim=0)
        
        segments.append(segment_tensor)
    
    # Stack all segments: (num_segments, 3, WINDOW_SIZE)
    result = torch.stack(segments, dim=0)
    logger.info(f"Data processed: output tensor shape {result.shape}")
    return result


class VehiclePredictor:
    """Handles model loading and predictions."""
    
    def __init__(self, model_path: str = None):
        """Initialize the predictor with a trained model.
        
        Args:
            model_path: Path to the trained model weights. If None, uses default.
        """
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the CNN model from disk."""
        logger.info(f"Loading model from {self.model_path}")
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        
        self.model = ConvolutionalNeuralNetwork(input_channels=3, num_classes=2)
        state_dict = torch.load(self.model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"Model loaded successfully on device: {self.device}")
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None
    
    def validate_csv(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """Validate CSV data has required columns.
        
        Args:
            df: DataFrame from CSV file
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {', '.join(missing_cols)}"
        
        # Check for numeric data
        for col in REQUIRED_COLUMNS:
            if not pd.api.types.is_numeric_dtype(df[col]):
                # Try to convert, replacing non-numeric with NaN
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except Exception:
                    return False, f"Column '{col}' must contain numeric values"
        
        # Check for sufficient data
        if len(df) < 50:  # Minimum window size
            return False, f"Insufficient data: need at least 50 rows, got {len(df)}"
        
        return True, ""
    
    def get_window_seconds(self) -> float:
        """Get the configured window size in seconds."""
        return WINDOW_SECONDS
    
    def predict(self, df: pd.DataFrame) -> Dict:
        """Run prediction on CSV data.
        
        Args:
            df: DataFrame with time, ax, ay, az columns
            
        Returns:
            Dictionary with prediction results
        """
        logger.info("Starting prediction pipeline")
        
        # Validate input
        logger.info("Validating CSV data...")
        is_valid, error_msg = self.validate_csv(df)
        if not is_valid:
            logger.error(f"Validation failed: {error_msg}")
            raise ValueError(error_msg)
        logger.info("CSV validation passed")
        
        # Clean data - replace non-numeric values
        logger.info("Cleaning data...")
        original_rows = len(df)
        for col in REQUIRED_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop rows with NaN values
        df = df.dropna(subset=REQUIRED_COLUMNS)
        cleaned_rows = len(df)
        if original_rows != cleaned_rows:
            logger.info(f"Dropped {original_rows - cleaned_rows} invalid rows")
        
        if len(df) < 50:
            raise ValueError("Insufficient valid data after cleaning")
        
        # Process data into tensor using configurable window size
        logger.info("Processing data into segments...")
        processed_data = process_data_with_window(df, WINDOW_SECONDS)
        processed_data = processed_data.to(self.device)
        
        # Run prediction
        logger.info("Running model inference...")
        with torch.no_grad():
            outputs = self.model(processed_data)
            probabilities = outputs.cpu().numpy()
            _, predicted = torch.max(outputs, 1)
            predicted_labels = predicted.cpu().numpy()
        logger.info("Model inference completed")
        
        # Build response
        total_segments = len(predicted_labels)
        logger.info(f"Building response for {total_segments} segments")
        
        # Segment predictions with time ranges
        segment_predictions = []
        time_col = df['time'].to_numpy()
        start_time = time_col.min()
        
        for i, (label_id, probs) in enumerate(zip(predicted_labels, probabilities)):
            segment_start = start_time + i * WINDOW_SECONDS
            segment_end = segment_start + WINDOW_SECONDS
            segment_predictions.append({
                "segment_id": i,
                "time_start": round(segment_start, 3),
                "time_end": round(segment_end, 3),
                "label": LABEL_MAP[int(label_id)],
                "label_id": int(label_id),
                "probabilities": {
                    "bike": float(probs[0]),
                    "car": float(probs[1])
                }
            })
        
        # Mode distribution
        label_counts = Counter(predicted_labels)
        mode_distribution = {}
        for label_name, label_id in LABEL_TO_ID.items():
            count = int(label_counts.get(label_id, 0))
            percentage = (count / total_segments) * 100 if total_segments > 0 else 0
            mode_distribution[label_name] = {
                "count": count,
                "percentage": round(percentage, 2)
            }
        
        # Overall prediction (majority vote)
        most_common_label_id = label_counts.most_common(1)[0][0]
        overall_confidence = label_counts[most_common_label_id] / total_segments
        
        return {
            "status": "success",
            "window_seconds": WINDOW_SECONDS,
            "total_segments": total_segments,
            "overall_prediction": {
                "label": LABEL_MAP[int(most_common_label_id)],
                "label_id": int(most_common_label_id),
                "confidence": round(overall_confidence, 4)
            },
            "segment_predictions": segment_predictions,
            "mode_distribution": mode_distribution
        }


# Global predictor instance (lazy loaded)
_predictor: VehiclePredictor = None


def get_predictor() -> VehiclePredictor:
    """Get or create the global predictor instance."""
    global _predictor
    if _predictor is None:
        _predictor = VehiclePredictor()
    return _predictor
